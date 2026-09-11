"""End-to-end validation on simulated multi-regime data with known ground truth.

The generator deliberately builds in the edge cases the package claims to handle:

  * K_true = 4 rotational/contractive regimes with distinct dynamics
  * a LATE-ONSET regime that only appears in the second half   -> tests birth
  * a DUPLICATE regime (same A, Q as another)                  -> tests merge
  * a RARE regime occupying <2% of timesteps                   -> tests delete
  * heterogeneous sequence lengths + trailing padding          -> tests valid mask
  * continuation chunks (is_first[:,0] == 0)                   -> tests boundary mask
  * an over-provisioned truncation K0 >> K_true

Recovery is scored on: number of active regimes, segmentation accuracy
(many-to-one + Hamming after Hungarian matching), dynamics MSE against the true
flow field, and monotonicity of the variational bound.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import shs_rssm.moves as M
from shs_rssm.offline_trainer import fit_offline_corpus
from shs_rssm.regime_head import RegimeHead


# ----------------------------------------------------------------- generator
def rot(theta, scale):
    c, s = np.cos(theta), np.sin(theta)
    return scale * np.array([[c, -s], [s, c]])


def make_regimes():
    """4 distinct + 1 duplicate of regime 0 + 1 rare."""
    A = [rot(0.25, 0.97),          # 0: slow CCW
         rot(-0.25, 0.97),         # 1: slow CW
         np.diag([0.99, 0.55]),    # 2: anisotropic contraction
         rot(0.60, 0.90),          # 3: fast CCW  (LATE ONSET)
         rot(0.25, 0.97),          # 4: DUPLICATE of 0
         np.diag([0.30, 0.30])]    # 5: RARE strong contraction
    b = [np.array([0.10, 0.00]), np.array([-0.10, 0.00]),
         np.array([0.00, 0.12]), np.array([0.00, -0.12]),
         np.array([0.10, 0.00]), np.array([0.00, 0.00])]
    Q = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    return A, b, Q


def simulate(n_seq=10, T=260, seed=0):
    rng = np.random.RandomState(seed)
    A, b, Q = make_regimes()
    Z, S, LEN = [], [], []
    for n in range(n_seq):
        T_n = int(T * rng.uniform(0.7, 1.0))        # heterogeneous lengths
        z = np.zeros((T_n, 2))
        s = np.zeros(T_n, dtype=int)
        x = rng.randn(2) * 0.3
        t = 0
        while t < T_n:
            late_ok = t > T_n // 2                   # regime 3 only appears late
            pool = [0, 1, 2, 4] + ([3] if late_ok else [])
            k = 5 if rng.rand() < 0.05 else pool[rng.randint(len(pool))]
            dur = max(8, int(rng.gamma(6.0, 5.0)))
            if k == 5:
                dur = max(3, dur // 6)               # rare regime is also short
            for _ in range(min(dur, T_n - t)):
                x = A[k] @ x + b[k] + np.sqrt(Q[k]) * rng.randn(2)
                z[t], s[t] = x, k
                t += 1
                if t >= T_n:
                    break
        Z.append(z); S.append(s); LEN.append(T_n)
    return Z, S, LEN, (A, b, Q)


def to_batch(Z, S, LEN, continuation_frac=0.3, seed=0):
    """Pack ragged sequences into a padded (B,T,L) batch with a valid mask.
    A fraction of rows are marked as CONTINUATION chunks (is_first[:,0]==0)."""
    rng = np.random.RandomState(seed)
    B, T = len(Z), max(LEN)
    z = torch.zeros(B, T, 2)
    valid = torch.zeros(B, T)
    isf = torch.zeros(B, T)
    strue = -np.ones((B, T), dtype=int)
    n_cont = int(round(continuation_frac * B))
    cont_rows = set(rng.choice(B, size=n_cont, replace=False).tolist())
    for i, (zz, ss, L_) in enumerate(zip(Z, S, LEN)):
        z[i, :L_] = torch.tensor(zz, dtype=torch.float32)
        valid[i, :L_] = 1.0
        strue[i, :L_] = ss
        isf[i, 0] = 0.0 if i in cont_rows else 1.0
    return z, valid, isf, strue


# ----------------------------------------------------------------- scoring
def score_segmentation(pred, true, mask):
    p, t = pred[mask], true[mask]
    ks, ts = np.unique(p), np.unique(t)
    Cm = np.zeros((len(ks), len(ts)))
    for i, a in enumerate(ks):
        for j, bb in enumerate(ts):
            Cm[i, j] = np.sum((p == a) & (t == bb))
    many_to_one = Cm.max(1).sum() / len(t)
    r, c = linear_sum_assignment(-Cm)
    hamming = 1.0 - Cm[r, c].sum() / len(t)
    return many_to_one, hamming


def dynamics_mse(head, A_true, b_true, pred, true, mask):
    """For each true regime, compare the fitted A of its dominant learned regime."""
    p, t = pred[mask], true[mask]
    Wt = head.regimes.M.detach().cpu().numpy()          # (K, L, G)
    errs = []
    for k in np.unique(t):
        sel = p[t == k]
        if len(sel) == 0:
            continue
        kk = np.bincount(sel).argmax()
        A_hat = Wt[kk][:, :2]
        errs.append(float(np.mean((A_hat - A_true[k]) ** 2)))
    return float(np.mean(errs)), errs


# ----------------------------------------------------------------- experiment
def run(K0=20, laps=25, sweep_every=3, seed=0, verbose=True):
    torch.manual_seed(seed)
    Z, S, LEN, (A_true, b_true, Q_true) = simulate(seed=seed)
    z, valid, isf, strue = to_batch(Z, S, LEN, seed=seed)
    B, T, L = z.shape
    d = torch.zeros(B, T, 2)                            # no deterministic carry
    zv = torch.full((B, T, L), 1e-3)

    n_cont = int((isf[:, 0] == 0).sum())
    if verbose:
        print(f"corpus: {B} sequences, T_max={T}, "
              f"{int(valid.sum())} valid steps ({int((1-valid).sum())} padded)")
        print(f"        {n_cont}/{B} rows are continuation chunks")
        occ = np.bincount(strue[strue >= 0], minlength=6) / max(1, (strue >= 0).sum())
        print(f"        true regime occupancy: "
              + " ".join(f"k{i}={o:.3f}" for i, o in enumerate(occ)))
        print(f"        K_true=6 (regime 4 duplicates 0; regime 5 is rare; "
              f"regime 3 is late-onset)")

    head = RegimeHead(stoch=L, deter=2, K=K0, proj_dim=None, action_dim=0,
                      a0=3.0, b0=0.1 * float(z[valid > 0].var()),
                      online_mode="memoized", expected_batches=1,
                      device=torch.device("cpu"))

    out = fit_offline_corpus(
        head, encode_fn=lambda: [("corpus", z, d, isf, zv, valid)],
        laps=laps, sweep_every=sweep_every, verbose=False,
        sweep_kwargs=dict(threshold=0.0, refine_iters=3))

    gam, _, _, _ = head.regime_inference(z, d, is_first=isf, z_var=zv, valid=valid)
    pred = gam.argmax(-1).cpu().numpy()
    mask = (valid.cpu().numpy() > 0) & (strue >= 0)
    # exclude the boundary step of continuation chunks: the model deliberately
    # drops it, so it carries no prediction
    for i in range(B):
        if float(isf[i, 0]) == 0.0:
            mask[i, 0] = False

    ident = strue.copy()
    ident[ident == 4] = 0                     # duplicate regime is unidentifiable
    m2o, ham = score_segmentation(pred, ident, mask)
    m2o_raw, _ = score_segmentation(pred, strue, mask)
    K_active = len(np.unique(pred[mask]))
    dmse, per_k = dynamics_mse(head, A_true, b_true, pred, ident, mask)
    n_ident = len(np.unique(ident[mask]))

    bounds = out["bounds"]
    scale = max(1.0, max(abs(x) for x in bounds))
    mono = all(bounds[i + 1] >= bounds[i] - 1e-6 * scale
               for i in range(len(bounds) - 1))

    if verbose:
        print(f"\nresult after {laps} laps")
        print(f"  K: {K0} -> {head.K} (truncation -> fitted), "
              f"{K_active} occupied  |  identifiable truth = {n_ident}")
        print(f"  K trace: {out['K_trace']}")
        print(f"  many-to-one accuracy : {m2o:.3f}  "
              f"(vs raw 6-label truth {m2o_raw:.3f})")
        print(f"  one-to-one Hamming   : {ham:.3f}")
        print(f"  dynamics MSE vs truth: {dmse:.5f}  per-regime "
              + " ".join(f"{e:.4f}" for e in per_k))
        print(f"  bound: {bounds[0]:.1f} -> {bounds[-1]:.1f}  monotone={mono}")
        seg = np.diff(np.flatnonzero(np.r_[True, pred[0][1:] != pred[0][:-1], True]))
        tseg = np.diff(np.flatnonzero(np.r_[True, strue[0][1:] != strue[0][:-1], True]))
        print(f"  seq0 segments: pred {len(seg)} (median {np.median(seg):.0f}) "
              f"vs true {len(tseg)} (median {np.median(tseg):.0f})")
    return dict(K=int(head.K), K_active=K_active, m2o=m2o, hamming=ham,
                dmse=dmse, bounds=bounds, monotone=mono, K_trace=out["K_trace"])


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or [0]
    rows = []
    for s in seeds:
        print("=" * 72)
        print(f"SEED {s}")
        print("=" * 72)
        rows.append(run(seed=s))
        print()
    if len(rows) > 1:
        print("=" * 72)
        print(f"{'seed':>5} {'K':>4} {'occ':>4} {'m2o':>7} {'hamming':>8} "
              f"{'dyn MSE':>9} {'monotone':>9}")
        for s, r in zip(seeds, rows):
            print(f"{s:5d} {r['K']:4d} {r['K_active']:4d} {r['m2o']:7.3f} "
                  f"{r['hamming']:8.3f} {r['dmse']:9.5f} {str(r['monotone']):>9}")

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from shs_rssm.regime_head import RegimeHead
from shs_rssm.offline_trainer import fit_offline_corpus
from shs_rssm.init_data import init_from_random_blocks
from shs_rssm.moves import MoveBuffer, aggregate_bound

OUT = pathlib.Path(os.environ.get("SHS_OUT", _HERE / "fhn_out"))
DT = torch.float64


def fhn_rhs(v, w, a=0.7, b=0.8, eps=0.08, I=0.8):
    return v - v ** 3 / 3.0 - w + I, eps * (v + a - b * w)


def simulate(n_seq=6, T=40000, dt=0.01, stride=20, burn=8000, obs_sd=0.01,
             seed=0, fast_pct=80.0, **kw):
    rng = np.random.default_rng(seed)
    Xs, Ls, Cs = [], [], []
    for _ in range(n_seq):
        v, w = rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0)
        traj = np.zeros((T, 2))
        dv = np.zeros(T)
        for t in range(T):
            traj[t] = (v, w)
            k1v, k1w = fhn_rhs(v, w, **kw)
            k2v, k2w = fhn_rhs(v + .5 * dt * k1v, w + .5 * dt * k1w, **kw)
            k3v, k3w = fhn_rhs(v + .5 * dt * k2v, w + .5 * dt * k2w, **kw)
            k4v, k4w = fhn_rhs(v + dt * k3v, w + dt * k3w, **kw)
            dv[t] = k1v
            v += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        traj, dv = traj[burn::stride], dv[burn::stride]
        Cs.append(traj.copy())
        thr = np.percentile(np.abs(dv), fast_pct)
        fast = np.abs(dv) > thr
        Ls.append(np.where(fast, np.where(dv > 0, 1, 2),
                           np.where(traj[:, 0] > 0, 0, 3)).astype(np.int64))
        Xs.append(traj + obs_sd * rng.normal(size=traj.shape))
    lens = [len(x) for x in Xs]
    doc_range = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    X = np.concatenate(Xs, 0)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    return ((X - mu) / sd, doc_range, np.concatenate(Ls, 0),
            np.concatenate(Cs, 0))


def innovation_variance(X, doc_range):
    d = [X[a + 1:b] - X[a:b - 1]
         for a, b in zip(doc_range[:-1], doc_range[1:])]
    return np.concatenate(d, 0).var(0)


def _contingency(t, p):
    tu, pu = np.unique(t), np.unique(p)
    C = np.zeros((tu.size, pu.size), dtype=np.int64)
    ti = {v: i for i, v in enumerate(tu)}
    pi = {v: i for i, v in enumerate(pu)}
    for a, b in zip(t, p):
        C[ti[a], pi[b]] += 1
    return C, tu, pu


def metrics(t, p):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    C, _, _ = _contingency(t, p)
    n = max(C.shape)
    sq = np.zeros((n, n), dtype=np.int64)
    sq[:C.shape[0], :C.shape[1]] = C
    r, c = linear_sum_assignment(-sq)
    return dict(hamming=1.0 - sq[r, c].sum() / t.size,
                m2o=C.max(axis=0).sum() / t.size,
                nmi=normalized_mutual_info_score(t, p),
                ari=adjusted_rand_score(t, p),
                K_used=int(np.unique(p).size))


def build_corpus(X, doc_range):
    corpus = []
    for s in range(len(doc_range) - 1):
        a, b = int(doc_range[s]), int(doc_range[s + 1])
        z = torch.tensor(X[a:b], dtype=DT).unsqueeze(0)
        T = z.shape[1]
        isf = torch.zeros(1, T, dtype=DT)
        isf[0, 0] = 1.0
        corpus.append((f"seq{s}", z, torch.zeros(1, T, 1, dtype=DT), isf))
    return corpus


def run(X, doc_range, labels, b0, K0=12, laps=6, seed=0, sF=1.0, verbose=True):
    if b0 is None:
        b0 = float(sF * 2.0 * innovation_variance(X, doc_range).mean())
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = RegimeHead(stoch=X.shape[1], deter=1, K=K0, proj_dim=None, a0=3.0, b0=b0,
                      ard=False, identity_init=True, q_rank=0, shared_carry=False,
                      gamma=5.0, alpha=0.5, kappa=50.0, start_alpha=1.0,
                      recurrent=False, online_mode="memoized",
                      expected_batches=len(doc_range) - 1,
                      dtype=DT, device=torch.device("cpu"))
    corpus = build_corpus(X, doc_range)
    init_from_random_blocks(head, corpus, K=K0, seed=seed)
    out = fit_offline_corpus(head, corpus, laps=laps, sweep_every=3,
                             sweep_kwargs=dict(threshold=0.0, create_bonus=0.0,
                                               refine_iters=2, do_birth=True,
                                               do_split=True),
                             verbose=verbose)
    with torch.no_grad():
        z = np.concatenate([head.regime_inference(c[1], c[2], is_first=c[3])[0][0]
                            .argmax(-1).cpu().numpy() for c in corpus])
    buf = MoveBuffer(max_batches=len(corpus))
    for (bid, s_, d_, f_) in corpus:
        buf.add(s_, d_, f_, batch_id=bid, repr_version=int(head.repr_version))
    return dict(b0=b0, EQ=b0 / 2.0, K=int(head.K), K_trace=out["K_trace"],
                bound=float(aggregate_bound(head, buf)), **metrics(labels, z)), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=float, default=None,
                    help="Normal-Gamma rate; omit to calibrate from the data")
    ap.add_argument("--K0", type=int, default=12)
    ap.add_argument("--laps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep E[Q] and report K at each setting")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    X, dr, lab, clean = simulate(seed=args.seed)
    iv = innovation_variance(X, dr)
    print("FitzHugh-Nagumo: %d sequences, %d steps, %d dims, true K = %d"
          % (len(dr) - 1, X.shape[0], X.shape[1], len(np.unique(lab))))
    print("occupancy %s" % np.round(np.bincount(lab) / len(lab), 3))
    print("measured innovation variance %s   (static default E[Q]=1.0 is %.0fx looser)"
          % (np.round(iv, 6), 1.0 / iv.mean()))
    np.savez(OUT / "dataset.npz", X=X, doc_range=dr, TrueZ=lab + 1, clean=clean,
             true_state_names=np.array([b"SLOW-RIGHT", b"FAST-UP",
                                        b"FAST-DOWN", b"SLOW-LEFT"]))

    grid = [2.0, 0.25, 0.02, None] if args.sweep else [args.b0]
    rows = []
    for b0 in grid:
        r, _ = run(X, dr, lab, b0, K0=args.K0, laps=args.laps, seed=args.seed,
                   verbose=not args.sweep)
        rows.append(r)
        print("  E[Q]=%-9.5f K=%-3d hamming %.3f  m2o %.3f  NMI %.3f  ARI %.3f%s"
              % (r["EQ"], r["K"], r["hamming"], r["m2o"], r["nmi"], r["ari"],
                 "   <- calibrated" if b0 is None else ""))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    print("true K = %d;  results -> %s" % (len(np.unique(lab)), OUT / "results.json"))


if __name__ == "__main__":
    main()

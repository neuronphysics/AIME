"""FitzHugh-Nagumo multi-step prediction: SHS-RSSM vs TrSLDS vs rSLDS.

Protocol follows Becker-Ehmck, Peters & van der Smagt, "Switching Linear
Dynamics for Variational Bayes Filtering", ICML 2019, Sec. 5.4 / Fig. 7b:

    tau * dw/dt = v + a - b*w,     dv/dt = v - v^3/3 - w + I_ext
    a = 0.7, b = 0.8, tau = 12.5,  I_ext ~ N(0.7, 0.04)  (per trajectory)
    100 trajectories of length 430; the last 30 steps are withheld during
    training and used for evaluation; starting states drawn from U[-3, 3]^2.

Reported metric is normalised multi-step prediction error as a function of
horizon h = 1..30:

    NRMSE(h) = sqrt( mean_n || x_n(T0+h) - xhat_n(T0+h) ||^2 / var(x) )

so 1.0 is the variance of the data (predicting the mean) and 0.0 is exact.
The paper plots a normalised error of the same shape, so the curves are
directly comparable in level, not only in ordering.

Baselines
---------
  shs     SHS-RSSM regime head, fit offline with birth/merge/delete moves.
  trslds  Nassar et al. 2019, vendored in shs_demo/trslds (Gibbs).
  rslds   Linderman et al. 2017, vendored in shs_demo/rslds.  Requires the
          2017 stack (pyhsmm/pybasicbayes), which does not build on
          python>=3.10; run it from the environment in
          shs_demo/compare/environment-baselines.yml.  Skipped with a clear
          message when unavailable, so the other two still produce a figure.

Usage
-----
    python fhn_benchmark.py --models shs trslds --seeds 0 1 2
    python fhn_benchmark.py --models rslds --seeds 0 1 2   # baselines env
    python fhn_benchmark.py --plot-only                    # re-draw figure
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "compare"))

OUT = _HERE / "fhn_bench"
OUT.mkdir(exist_ok=True)

A_FHN, B_FHN, TAU = 0.7, 0.8, 12.5
N_TRAJ, T_TOTAL, T_HOLDOUT = 100, 430, 30
T_TRAIN = T_TOTAL - T_HOLDOUT


# --------------------------------------------------------------------------
# data: the paper's FitzHugh-Nagumo setup
# --------------------------------------------------------------------------
def simulate_fhn(n_traj=N_TRAJ, T=T_TOTAL, dt=0.1, seed=0, obs_sd=0.0):
    """RK4 integration of FHN with a per-trajectory stochastic drive."""
    rng = np.random.default_rng(seed)

    def rhs(v, w, I):
        return v - v ** 3 / 3.0 - w + I, (v + A_FHN - B_FHN * w) / TAU

    X = np.zeros((n_traj, T, 2))
    for n in range(n_traj):
        I = rng.normal(0.7, np.sqrt(0.04))          # N(0.7, 0.04) -> sd 0.2
        v, w = rng.uniform(-3.0, 3.0, size=2)
        for t in range(T):
            X[n, t] = (v, w)
            k1v, k1w = rhs(v, w, I)
            k2v, k2w = rhs(v + .5 * dt * k1v, w + .5 * dt * k1w, I)
            k3v, k3w = rhs(v + .5 * dt * k2v, w + .5 * dt * k2w, I)
            k4v, k4w = rhs(v + dt * k3v, w + dt * k3w, I)
            v += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
    if obs_sd > 0:
        X = X + obs_sd * rng.normal(size=X.shape)
    return X


def standardise(X):
    tr = X[:, :T_TRAIN].reshape(-1, X.shape[-1])
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return (X - mu) / sd, mu, sd


def nrmse_by_horizon(pred, true):
    """pred, true: (n_traj, T_HOLDOUT, D) in standardised units."""
    se = ((pred - true) ** 2).sum(-1)                # (n, h)
    denom = (true ** 2).sum(-1).mean()               # data variance, std units
    return np.sqrt(se.mean(0) / max(denom, 1e-12))


# --------------------------------------------------------------------------
# SHS-RSSM
# --------------------------------------------------------------------------
def run_shs(Xs, K0=12, laps=25, sweep_every=3, seed=0, sF=0.1, verbose=True,
            recurrent=False, gate_input="carry"):
    import torch
    from shs_rssm.regime_head import RegimeHead
    from shs_rssm.offline_trainer import fit_offline_corpus

    torch.manual_seed(seed)
    n, T, D = Xs.shape
    z = torch.tensor(Xs[:, :T_TRAIN], dtype=torch.float32)
    d = torch.zeros(n, T_TRAIN, 2)
    isf = torch.zeros(n, T_TRAIN); isf[:, 0] = 1.0
    zv = torch.full((n, T_TRAIN, D), 1e-4)

    head = RegimeHead(stoch=D, deter=2, K=K0, proj_dim=None, action_dim=0,
                      a0=3.0, b0=sF * float(z.var()),
                      recurrent=recurrent, gate_input=gate_input,
                      online_mode="memoized", expected_batches=1,
                      device=torch.device("cpu"))
    t0 = time.time()
    out = fit_offline_corpus(
        head, encode_fn=lambda: [("corpus", z, d, isf, zv, None)],
        laps=laps, sweep_every=sweep_every, verbose=False,
        sweep_kwargs=dict(threshold=0.0, refine_iters=3))
    fit_s = time.time() - t0

    # roll the fitted switching dynamics forward from the last training state
    with torch.no_grad():
        gam, _, _, _ = head.regime_inference(z, d, is_first=isf, z_var=zv)
        resp = gam[:, -1]                                    # (n, K)
        x = z[:, -1]                                         # (n, D)
        d1 = torch.zeros(n, 2)
        preds = []
        for _ in range(T_HOLDOUT):
            x, resp, mean, _ = head.imagine_prior(x, d1, resp, sample=False)
            preds.append(mean.clone())
            x = mean
        P = torch.stack(preds, 1).numpy()
    if verbose:
        print(f"    shs[{gate_input if recurrent else 'no-gate'}]: K {K0} -> {head.K}, "
              f"bound {out['bounds'][0]:.1f} -> {out['bounds'][-1]:.1f}, {fit_s:.0f}s")
    return P, dict(K=int(head.K), K_trace=out["K_trace"],
                   bounds=out["bounds"], fit_s=fit_s)


# --------------------------------------------------------------------------
# TrSLDS  (Nassar et al. 2019, vendored)
# --------------------------------------------------------------------------
def run_trslds(Xs, K=8, n_samples=150, burnin=100, seed=0, verbose=True):
    from io_utils import ensure_legacy_scipy, ensure_pypolyagamma
    ensure_pypolyagamma(); ensure_legacy_scipy()
    from trslds import conditionals as _cond
    from trslds import initialize as init
    from trslds import models as _models
    from trslds.models import TroSLDS
    # Upstream sets n_cpu = cpu_count()//2 at import time: 0 on a single-core
    # box (joblib raises), and the WHOLE node under Slurm (cpu_count ignores
    # cgroups).  Clamp to [1, allowed] at runtime, as compare/run_trslds.py does.
    import os as _os
    _avail = (len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity")
              else (_os.cpu_count() or 1))
    _models.n_cpu = max(1, min(int(_models.n_cpu), _avail))
    _cond.n_cpu = max(1, min(int(_cond.n_cpu), _avail))

    np.random.seed(seed)
    n, T, D = Xs.shape
    Y = [Xs[i, :T_TRAIN].T.copy() for i in range(n)]         # (D, T) per seq
    na = np.newaxis

    t0 = time.time()
    (A, C, R, X, Z, Path, possible_paths,
     leaf_path, leaf_nodes) = init.initialize(Y, D, K, 100, 32, 1e-3)
    model = TroSLDS(D_in=D, D_out=D, K=K, dynamics=A,
                    dynamics_noise=np.repeat(np.eye(D)[:, :, na], K, axis=2),
                    emission=C, emission_noise=np.eye(D), hyper_planes=R,
                    possible_paths=possible_paths, leaf_path=leaf_path,
                    leaf_nodes=leaf_nodes, scale=0.5)
    for i in range(len(Y)):
        model._add_data(X[i], Y[i], Z[i], Path[i])
    model._initialize_polya_gamma()
    for m in range(n_samples):
        model._sample_emission(); model._sample_hyperplanes()
        model._sample_dynamics(); model._sample_discrete_latent()
        model._sample_continuous_latent()
        if verbose and (m + 1) % 50 == 0:
            print(f"    trslds gibbs {m+1}/{n_samples} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fit_s = time.time() - t0

    # forward-simulate with the vendored rollout (Bayes-classifier leaf choice,
    # noise off -> posterior-mean prediction, matching the other models)
    # NOTE: trslds.utils.generate_trajectory has an upstream bug in its
    # noise=False branch (`choice` is referenced before assignment at
    # utils.py:448; it should be np.argmax(p)).  The vendored tree is kept
    # byte-identical to upstream, so the deterministic rollout is done here
    # using the same leaf rule, utils.compute_leaf_log_prob.
    from trslds import utils as _u
    depth = int(np.asarray(leaf_path).shape[0])
    Aleaf = np.asarray(model.A[-1]) if isinstance(model.A, list) else np.asarray(model.A)
    Cm = np.asarray(model.C)
    preds = np.zeros((n, T_HOLDOUT, D))
    for i in range(n):
        x = np.asarray(model.x[i])[:, -1].copy()
        for h in range(T_HOLDOUT):
            lp = _u.compute_leaf_log_prob(model.R, x, K, depth, leaf_path)
            k = int(np.argmax(np.asarray(lp).ravel()))
            x = (Aleaf[:, :-1, k] @ x[:, None] + Aleaf[:, -1:, k]).ravel()
            preds[i, h] = (Cm @ np.append(x, 1.0) if Cm.shape[1] == D + 1
                           else Cm @ x)
    if verbose:
        print(f"    trslds: K={K}, {fit_s:.0f}s")
    return preds, dict(K=int(K), fit_s=fit_s)


# --------------------------------------------------------------------------
# rSLDS  (Linderman et al. 2017, vendored; needs the 2017 stack)
# --------------------------------------------------------------------------
def run_rslds(Xs, K=8, n_samples=200, seed=0, verbose=True):
    try:
        from pybasicbayes.distributions import Regression  # noqa: F401
        from rslds.models import PGRecurrentSLDS           # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "rSLDS needs the 2017 Linderman stack (pyhsmm, pybasicbayes), "
            "which does not build on python>=3.10. Create the environment in "
            "shs_demo/compare/environment-baselines.yml and re-run with "
            "--models rslds; results merge into the same fhn_bench/ folder.\n"
            f"  underlying import error: {e}")

    import numpy.random as npr
    from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression
    from rslds.models import PGRecurrentSLDS
    from rslds.util import compute_psi_cmoments  # noqa: F401

    npr.seed(seed)
    n, T, D = Xs.shape
    ys = [Xs[i, :T_TRAIN].copy() for i in range(n)]
    t0 = time.time()
    init_dyn = [Gaussian(mu=np.zeros(D), sigma=np.eye(D), nu_0=D + 2,
                         sigma_0=3. * np.eye(D), mu_0=np.zeros(D), kappa_0=1.0)
                for _ in range(K)]
    dyn = [Regression(nu_0=D + 2, S_0=1e-4 * np.eye(D),
                      M_0=np.hstack((np.eye(D), np.zeros((D, 1)))),
                      K_0=np.eye(D + 1)) for _ in range(K)]
    emis = DiagonalRegression(D, D + 1, A=np.hstack((np.eye(D), np.zeros((D, 1)))),
                              sigmasq=np.ones(D), alpha_0=2.0, beta_0=2.0)
    model = PGRecurrentSLDS(
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.),
        init_state_distn='uniform', init_dynamics_distns=init_dyn,
        dynamics_distns=dyn, emission_distns=emis, fixed_emission=False)
    for y in ys:
        model.add_data(y)
    for m in range(n_samples):
        model.resample_model()
        if verbose and (m + 1) % 50 == 0:
            print(f"    rslds gibbs {m+1}/{n_samples} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fit_s = time.time() - t0

    preds = np.zeros((n, T_HOLDOUT, D))
    for i, sd in enumerate(model.states_list):
        x = sd.gaussian_states[-1].copy()
        z = int(sd.stateseq[-1])
        for h in range(T_HOLDOUT):
            Ab = model.dynamics_distns[z].A
            x = Ab[:, :D] @ x + Ab[:, D]
            preds[i, h] = model.emission_distns.A[:, :D] @ x \
                + model.emission_distns.A[:, D]
            z = int(np.argmax(model.trans_distn.get_trans_matrices(
                x[None])[0][z])) if hasattr(model.trans_distn,
                                            "get_trans_matrices") else z
    if verbose:
        print(f"    rslds: K={K}, {fit_s:.0f}s")
    return preds, dict(K=int(K), fit_s=fit_s)


# --------------------------------------------------------------------------
# baselines for context
# --------------------------------------------------------------------------
def run_persistence(Xs):
    last = Xs[:, T_TRAIN - 1][:, None, :]
    return np.repeat(last, T_HOLDOUT, axis=1), dict()


def run_linear(Xs):
    """single global LDS fit by least squares -- the no-switching control"""
    tr = Xs[:, :T_TRAIN]
    Xp = tr[:, :-1].reshape(-1, tr.shape[-1])
    Xn = tr[:, 1:].reshape(-1, tr.shape[-1])
    G = np.hstack([Xp, np.ones((len(Xp), 1))])
    W = np.linalg.lstsq(G, Xn, rcond=None)[0]
    x = Xs[:, T_TRAIN - 1].copy()
    preds = []
    for _ in range(T_HOLDOUT):
        x = np.hstack([x, np.ones((len(x), 1))]) @ W
        preds.append(x.copy())
    return np.stack(preds, 1), dict()


def run_shs_carry(Xs, **kw):
    return run_shs(Xs, recurrent=True, gate_input="carry", **kw)


def run_shs_latent(Xs, **kw):
    return run_shs(Xs, recurrent=True, gate_input="latent", **kw)


RUNNERS = dict(shs=run_shs, shs_carry=run_shs_carry, shs_latent=run_shs_latent,
               trslds=run_trslds, rslds=run_rslds,
               persistence=run_persistence, linear=run_linear)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["persistence", "linear", "shs", "trslds"],
                    choices=sorted(RUNNERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-traj", type=int, default=N_TRAJ)
    ap.add_argument("--shs-K", type=int, default=12)
    ap.add_argument("--trslds-K", type=int, default=8)
    ap.add_argument("--rslds-K", type=int, default=8)
    ap.add_argument("--laps", type=int, default=25)
    ap.add_argument("--gibbs", type=int, default=150)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        for seed in args.seeds:
            X = simulate_fhn(n_traj=args.n_traj, seed=seed)
            Xs, mu, sd = standardise(X)
            true = Xs[:, T_TRAIN:]
            print(f"seed {seed}: {Xs.shape[0]} trajectories, "
                  f"train {T_TRAIN}, holdout {T_HOLDOUT}")
            for name in args.models:
                try:
                    kw = dict(seed=seed)
                    if name.startswith("shs"):
                        kw.update(K0=args.shs_K, laps=args.laps)
                    elif name == "trslds":
                        kw.update(K=args.trslds_K, n_samples=args.gibbs,
                                  burnin=args.gibbs // 3)
                    elif name == "rslds":
                        kw.update(K=args.rslds_K, n_samples=args.gibbs)
                    elif name in ("persistence", "linear"):
                        kw = {}
                    pred, meta = RUNNERS[name](Xs, **kw)
                    curve = nrmse_by_horizon(pred, true)
                    np.savez(OUT / f"{name}_seed{seed}.npz",
                             curve=curve, pred=pred[:8], meta=json.dumps(
                                 {k: v for k, v in meta.items()}))
                    print(f"  {name:12s} NRMSE h=1 {curve[0]:.3f} | "
                          f"h=10 {curve[9]:.3f} | h=30 {curve[-1]:.3f}")
                except Exception as e:
                    print(f"  {name:12s} SKIPPED: {e}")

    make_figure(args.models, args.seeds)


def make_figure(models, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = dict(persistence=("0.6", ":", "persistence"),
                 shs_carry=("tab:purple", "-", "SHS gate=carry"),
                 shs_latent=("tab:red", "-", "SHS gate=latent (z->s)"),
                 linear=("0.35", "--", "single LDS"),
                 rslds=("tab:orange", "-", "rSLDS (Linderman '17)"),
                 trslds=("tab:green", "-", "TrSLDS (Nassar '19)"),
                 shs=("tab:blue", "-", "SHS-RSSM (ours)"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for name in models:
        cs = []
        for s in seeds:
            f = OUT / f"{name}_seed{s}.npz"
            if f.exists():
                cs.append(np.load(f, allow_pickle=True)["curve"])
        if not cs:
            continue
        Cm = np.stack(cs)
        m, lo, hi = Cm.mean(0), Cm.min(0), Cm.max(0)
        h = np.arange(1, len(m) + 1)
        c, ls, lab = style.get(name, ("k", "-", name))
        ax[0].plot(h, m, ls, color=c, lw=2, label=lab)
        if len(cs) > 1:
            ax[0].fill_between(h, lo, hi, color=c, alpha=0.15, lw=0)
        rows.append((lab, m[0], m[4], m[9], m[-1], len(cs)))
    ax[0].set_xlabel("steps predicted into the future")
    ax[0].set_ylabel("normalised prediction error (NRMSE)")
    ax[0].set_title("FitzHugh-Nagumo multi-step prediction\n"
                    "(Becker-Ehmck et al. 2019 protocol)", fontsize=10)
    ax[0].axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax[0].text(1.2, 1.02, "predicting the mean", fontsize=7, alpha=0.6)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.25)
    ax[0].set_xlim(1, T_HOLDOUT)

    ax[1].axis("off")
    if not rows:
        print("no result files found in", OUT, "-- nothing to plot")
        return
    tbl = [["model", "h=1", "h=5", "h=10", "h=30", "seeds"]]
    tbl += [[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.3f}",
             f"{r[4]:.3f}", str(r[5])] for r in rows]
    t = ax[1].table(cellText=tbl[1:], colLabels=tbl[0], loc="center",
                    cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
    ax[1].set_title("NRMSE by horizon", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fhn_multistep.png", dpi=150)
    print(f"\nwrote {OUT/'fhn_multistep.png'}")
    with open(OUT / "table.md", "w") as fh:
        fh.write("| " + " | ".join(tbl[0]) + " |\n")
        fh.write("|" + "---|" * len(tbl[0]) + "\n")
        for r in tbl[1:]:
            fh.write("| " + " | ".join(r) + " |\n")
    print(f"wrote {OUT/'table.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TrSLDS (Nassar et al., ICLR 2019) baseline on nascar / toyark13 / mocap6.

Runs the FULL reference pipeline from ``examples/tree_nascar.py`` against the
vendored ``shs_demo/trslds`` package: greedy tree initialisation (torch SGD),
then the Polya-Gamma Gibbs sampler over emissions, hyperplanes, dynamics,
discrete and continuous latents.  Nothing about the model is reimplemented
here -- only data loading, metrics and result IO.

On python>=3.10 the original ``pypolyagamma`` extension does not build; this
script transparently substitutes the maintained ``polyagamma`` sampler through
``io_utils.ensure_pypolyagamma`` (identical PG(b, c) parameterisation).

The reported segmentation is the per-timestep posterior mode over post-burn-in
Gibbs draws.  TrSLDS labels are tree leaves tied to hyperplanes, so they are
stable within a chain; still, the mode is a summary, not a single sample.

What is and is not comparable: TrSLDS partitions the *continuous latent
space* with hyperplanes, which matches NASCAR's generative structure and is
the regime where it is expected to be strong.  ToyARK13 and mocap6 switch by a
Markov chain with no spatial structure, the regime the HDP-HMM family targets;
report that context with the numbers, and note K is fixed by the tree (no
model selection), so compare segmentation quality, not inferred K.

Examples
--------
    python run_trslds.py --dataset nascar --samples 200 --burnin 100
    python run_trslds.py --dataset toyark13 --nseq 12
    python run_trslds.py --dataset mocap6 --dlatent 6
"""
import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # compare/ modules
sys.path.insert(0, str(HERE.parents[0]))         # shs_demo/ -> vendored trslds

import datasets                                   # noqa: E402
from io_utils import (Timer, ensure_legacy_scipy, ensure_pypolyagamma,  # noqa: E402
                      save_result)
from metrics import all_metrics                   # noqa: E402

_SCIPY = ensure_legacy_scipy()
_PG = ensure_pypolyagamma()

from trslds import conditionals as _cond          # noqa: E402
from trslds import initialize as init             # noqa: E402
from trslds import models as _models              # noqa: E402
from trslds import utils                          # noqa: E402
from trslds.models import TroSLDS                 # noqa: E402

# Upstream sets n_cpu = cpu_count() // 2 at import time.  That is 0 on a
# single-core box (joblib raises on n_jobs=0) and, on a cluster, counts the
# WHOLE node rather than the allocation: os.cpu_count() ignores cgroup CPU
# sets, so a 4-CPU Slurm allocation on a 64-core node would spawn 32 loky
# workers.  Clamp to [1, allowed-CPU count] at runtime rather than editing
# the vendored source.
import os as _os
_avail = (len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity")
          else (_os.cpu_count() or 1))
_models.n_cpu = max(1, min(int(_models.n_cpu), _avail))
_cond.n_cpu = max(1, min(int(_cond.n_cpu), _avail))

na = np.newaxis

DEFAULTS = dict(
    nascar=dict(K=4, dlatent=2, samples=200, burnin=100),
    toyark13=dict(K=16, dlatent=3, samples=300, burnin=150),
    mocap6=dict(K=16, dlatent=6, samples=300, burnin=150),
)


def run_trslds(Y, D_in, K, n_samples, burnin, max_epochs, batch_size, lr, seed):
    np.random.seed(seed)
    D_out = Y[0].shape[0]

    t0 = time.time()
    (A, C, R, X, Z, Path, possible_paths,
     leaf_path, leaf_nodes) = init.initialize(Y, D_in, K, max_epochs,
                                              batch_size, lr)
    t_init = time.time() - t0

    Qstart = np.repeat(np.eye(D_in)[:, :, na], K, axis=2)
    Sstart = np.eye(D_out)
    model = TroSLDS(D_in=D_in, D_out=D_out, K=K, dynamics=A,
                    dynamics_noise=Qstart, emission=C, emission_noise=Sstart,
                    hyper_planes=R, possible_paths=possible_paths,
                    leaf_path=leaf_path, leaf_nodes=leaf_nodes, scale=0.5)
    for idx in range(len(Y)):
        model._add_data(X[idx], Y[idx], Z[idx], Path[idx])

    # Gibbs: the five conditionals, in the reference order.
    t0 = time.time()
    model._initialize_polya_gamma()
    z_draws = []
    for m in range(n_samples):
        model._sample_emission()
        model._sample_hyperplanes()
        model._sample_dynamics()
        model._sample_discrete_latent()
        model._sample_continuous_latent()
        if m >= burnin:
            # model.z[n] has length T_n + 1: index 0 is the regime of the
            # initial latent state x_0; entries 1..T align with the T
            # observations.  Drop index 0 so metrics compare like with like.
            z_draws.append([np.asarray(z).ravel().astype(np.int64)[1:]
                            for z in model.z])
        if (m + 1) % 25 == 0:
            print(f"    gibbs {m + 1}/{n_samples} ({time.time() - t0:.0f}s)",
                  flush=True)
    t_gibbs = time.time() - t0

    # per-timestep posterior mode of the discrete path
    z_mode = []
    for s in range(len(model.z)):
        stack = np.stack([d[s] for d in z_draws], axis=0)
        z_mode.append(np.array([np.bincount(stack[:, t], minlength=K).argmax()
                                for t in range(stack.shape[1])]))
    x_last = [np.asarray(x).T[1:].copy() for x in model.x]  # (T, D_in) per seq
    return dict(z_mode=z_mode, x_last=x_last, t_init=t_init, t_gibbs=t_gibbs,
                n_kept=len(z_draws))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    ap.add_argument("--nseq", type=int, default=None)
    ap.add_argument("--K", type=int, default=None,
                    help="tree LEAVES; must be a power of two")
    ap.add_argument("--dlatent", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--burnin", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0,
                    help="nascar: pins the synthetic data realisation "
                         "(shared across models and seeds); --seed varies "
                         "initialisation only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="trslds")
    args = ap.parse_args()

    cfg = dict(DEFAULTS[args.dataset])
    for k in ("K", "dlatent", "samples", "burnin"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.quick:
        cfg["samples"], cfg["burnin"] = 40, 20
        args.max_epochs = min(args.max_epochs, 30)
    if cfg["K"] & (cfg["K"] - 1):
        sys.exit(f"--K must be a power of two (balanced binary tree); "
                 f"got {cfg['K']}")
    if cfg["burnin"] >= cfg["samples"]:
        sys.exit("--burnin must be < --samples")

    load_kw = {}
    if args.dataset == "toyark13":
        load_kw["n_seq"] = args.nseq or (4 if args.quick else 12)
    if args.dataset == "nascar":
        load_kw["n_seq"] = args.nseq or 5
        load_kw["seed"] = args.data_seed
    bundle = datasets.load(args.dataset, **load_kw)
    Y = [s.T.copy() for s in bundle["seqs"]]        # trslds expects (D, T)
    z_true = np.concatenate(bundle["z_true"])

    print(f"[trslds] {args.dataset}: {len(Y)} seqs, D={Y[0].shape[0]}, "
          f"K={cfg['K']} leaves, D_lat={cfg['dlatent']}, "
          f"{cfg['samples']} samples ({cfg['burnin']} burn-in), PG={_PG}")

    with Timer() as tm:
        res = run_trslds(Y, cfg["dlatent"], cfg["K"], cfg["samples"],
                         cfg["burnin"], args.max_epochs, args.batch_size,
                         args.lr, args.seed)

    z_pred = np.concatenate(res["z_mode"])
    m, _ = all_metrics(z_true, z_pred)
    print(f"[trslds] {args.dataset}: used={m['K_used']}/{cfg['K']} "
          f"hamming={m['hamming']:.3f} m2o={m['m2o']:.3f} "
          f"nmi={m['nmi']:.3f} (init {res['t_init']:.0f}s, "
          f"gibbs {res['t_gibbs']:.0f}s)")

    params = dict(vars(args), **cfg, pg_backend=_PG, n_kept=res["n_kept"])
    save_result(bundle["name"], args.tag, z_pred, bundle["doc_range"],
                tm.elapsed, params,
                x_latent=(np.concatenate(res["x_last"], 0)
                          if cfg["dlatent"] == 2 else None))


if __name__ == "__main__":
    main()

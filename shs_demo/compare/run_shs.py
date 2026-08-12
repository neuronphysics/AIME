#!/usr/bin/env python3
"""Fit SHS-RSSM's regime head to a benchmark dataset and save a standard result.

Uses the package's offline path -- ``init_from_random_blocks`` +
``fit_offline_corpus`` with structure moves on the frozen corpus -- i.e. the
same code path as ``fhn_demo.py`` and the Hughes-protocol configuration for
ToyARK13.  Per-dataset defaults:

    toyark13   K0=25, kappa=0,  alpha=0.5, gamma=10, start_alpha=10,
               b0 calibrated with sF=0.1 (bnpy convention), seqcreate births
    nascar     K0=8,  kappa=50, alpha=0.5, gamma=5,  start_alpha=1,
               PCA to 2-D first (the head plays the encoder-latent role),
               b0 from innovation variance
    mocap6     K0=20, kappa=50, alpha=0.5, gamma=5,  start_alpha=1,
               standardized channels, package default b0=2.0, and
               init_block=100 (contiguous-window initialisation: the
               auto default of ~10 frames seeds a fast-switching
               basin the AR likelihood cannot escape here)

Examples
--------
    python run_shs.py --dataset toyark13 --nseq 12 --laps 8
    python run_shs.py --dataset nascar --quick
    python run_shs.py --dataset mocap6
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # compare/ modules
sys.path.insert(0, str(HERE.parents[1]))         # AIME root -> shs_rssm

import datasets                                   
from io_utils import Timer, save_result           
from metrics import all_metrics                   

import torch                                      
from shs_rssm.regime_head import RegimeHead      
from shs_rssm.offline_trainer import fit_offline_corpus  
from shs_rssm.init_data import init_from_random_blocks   

DT = torch.float64

DEFAULTS = dict(
    toyark13=dict(K0=25, kappa=0.0, alpha=0.5, gamma=10.0, start_alpha=10.0,
                  laps=8, sweep_every=1, b0_mode="calibrate", sF=0.1, pca=0,
                  sweep=dict(threshold=0.0, create_bonus=0.0, refine_iters=2,
                             do_birth=True, do_split=False,
                             birth_style="seqcreate")),
    nascar=dict(K0=8, kappa=50.0, alpha=0.5, gamma=5.0, start_alpha=1.0,
                laps=6, sweep_every=3, b0_mode="innovation", sF=1.0, pca=2,
                sweep=dict(threshold=0.0, create_bonus=0.0, refine_iters=2,
                           do_birth=True, do_split=True)),
    mocap6=dict(K0=20, kappa=50.0, alpha=0.5, gamma=5.0, start_alpha=1.0,
                laps=8, sweep_every=2, b0_mode="default", sF=1.0, pca=0,
                init_block=100,  
                sweep=dict(threshold=0.0, create_bonus=0.0, refine_iters=2,
                           do_birth=True, do_split=True)),
)


def innovation_variance(seqs):
    d = np.concatenate([s[1:] - s[:-1] for s in seqs], 0)
    return d.var(0)


def build_corpus(seqs):
    corpus = []
    for i, s in enumerate(seqs):
        z = torch.tensor(np.asarray(s), dtype=DT).unsqueeze(0)
        T = z.shape[1]
        isf = torch.zeros(1, T, dtype=DT)
        isf[0, 0] = 1.0
        corpus.append((f"seq{i}", z, torch.zeros(1, T, 1, dtype=DT), isf))
    return corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    ap.add_argument("--nseq", type=int, default=None,
                    help="toyark13/nascar: number of sequences")
    ap.add_argument("--K0", type=int, default=None, help="truncation")
    ap.add_argument("--laps", type=int, default=None)
    ap.add_argument("--no-birth", action="store_true",
                    help="disable birth and split: the sweep then only merges "
                         "and deletes, so extra laps add PRUNE pressure "
                         "without also adding growth pressure")
    ap.add_argument("--move-threshold", type=float, default=None,
                    help="minimum bound improvement (nats) a move must clear "
                         "to be accepted; the default 0.0 accepts any "
                         "improvement, so with a tight noise prior births "
                         "essentially always win and K grows with the move "
                         "budget. A positive value is an Occam margin.")
    ap.add_argument("--create-bonus", type=float, default=None,
                    help="extra bound credit given to a birth (negative = "
                         "penalty on adding states)")
    ap.add_argument("--sweep-every", type=int, default=None,
                    help="structure-move sweep interval in laps "
                         "(1 = merges/deletes considered every lap)")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--sF", type=float, default=None)
    ap.add_argument("--b0-mode", choices=["default", "calibrate", "innovation"],
                    default=None, help="override the dataset's noise-rate mode")
    ap.add_argument("--b0", type=float, default=None,
                    help="explicit Normal-Gamma rate (implies --b0-mode default)")
    ap.add_argument("--learn-b0", action="store_true",
                    help="conjugate Gamma hierarchy on the noise rate "
                         "(manuscript Table mocap rows E/E'), initialised at "
                         "the resolved b0")
    ap.add_argument("--b0-strength", type=float, default=2.0,
                    help="prior shape c0 of b0 ~ Gamma(c0, c0/b0)")
    ap.add_argument("--q-rank", type=int, default=0,
                    help="rank of correlated emission noise (row C uses 3)")
    ap.add_argument("--init-block", type=int, default=None,
                    help="contiguous init block length in frames; overrides the "
                         "dataset default (mocap6: 100; others: auto ~ "
                         "max(10, T//2K))")
    ap.add_argument("--pca", type=int, default=None,
                    help="project observations to this many PCs first (0=off)")
    ap.add_argument("--prior-persist", type=float, default=0.9,
                    help="recurrent gate: prior mean self-persistence. 0.9 is "
                         "the package default; 0.98 = the exact carry-over of "
                         "kappa=50, alpha=1")
    ap.add_argument("--bias-prior-var", type=float, default=4.0,
                    help="recurrent gate: prior variance of the logit bias. "
                         "4.0 (default) spans persistence 0.55-0.985 and is "
                         "easily overruled by the likelihood; ~0.25 holds the "
                         "gate near its prior and suppresses flicker")
    ap.add_argument("--recurrent", action="store_true",
                    help="state-dependent persistence (Polya-Gamma gate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0,
                    help="nascar: pins the synthetic data realisation "
                         "(shared across models and seeds); --seed varies "
                         "initialisation only")
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test scale (fewer sequences/laps)")
    ap.add_argument("--tag", default="shs", help="result file stem")
    args = ap.parse_args()

    cfg = dict(DEFAULTS[args.dataset])
    for k in ("K0", "laps", "kappa", "sF", "pca", "sweep_every"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.quick:
        cfg["laps"] = min(cfg["laps"], 3)
    cfg["sweep"] = dict(cfg["sweep"])          # do not mutate DEFAULTS
    if args.no_birth:
        cfg["sweep"]["do_birth"] = False
        cfg["sweep"]["do_split"] = False
    if args.move_threshold is not None:
        cfg["sweep"]["threshold"] = float(args.move_threshold)
    if args.create_bonus is not None:
        cfg["sweep"]["create_bonus"] = float(args.create_bonus)

    load_kw = {}
    if args.dataset == "toyark13":
        load_kw["n_seq"] = args.nseq or (4 if args.quick else 12)
    if args.dataset == "nascar":
        load_kw["n_seq"] = args.nseq or 5
        load_kw["seed"] = args.data_seed
    bundle = datasets.load(args.dataset, **load_kw)
    seqs = bundle["seqs"]

    if cfg["pca"]:
        from sklearn.decomposition import PCA
        X = np.concatenate(seqs, 0)
        p = PCA(n_components=cfg["pca"], whiten=True).fit(X)
        seqs = [p.transform(s) for s in seqs]
        print(f"[shs] PCA {bundle['D']} -> {cfg['pca']} "
              f"(explained {p.explained_variance_ratio_.sum():.3f})")
    D = seqs[0].shape[1]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.b0_mode is not None:
        cfg["b0_mode"] = args.b0_mode
    if args.b0 is not None:
        cfg["b0_mode"] = "default"
    if cfg["b0_mode"] == "innovation":
        b0 = float(cfg["sF"] * 2.0 * innovation_variance(seqs).mean())
    else:
        b0 = 2.0 if args.b0 is None else float(args.b0)
    head = RegimeHead(stoch=D, deter=1, K=cfg["K0"], proj_dim=None, a0=3.0,
                      b0=b0, ard=False, identity_init=True,
                      q_rank=int(args.q_rank),
                      learn_b0=bool(args.learn_b0),
                      b0_strength=float(args.b0_strength),
                      shared_carry=False, gamma=cfg["gamma"],
                      alpha=cfg["alpha"], kappa=cfg["kappa"],
                      start_alpha=cfg["start_alpha"],
                      recurrent=bool(args.recurrent),
                      prior_persist=float(args.prior_persist),
                      rstick_bias_var=float(args.bias_prior_var),
                      online_mode="memoized",
                      expected_batches=len(seqs), dtype=DT,
                      device=torch.device("cpu"))
    corpus = build_corpus(seqs)
    # Bound the q(u) L-BFGS inner loop (sticky_hdp exposes the
    # lbfgs_max_iter hook; default 200).  With --recurrent, birth-candidate
    # rho/omega landscapes can pin the strong-Wolfe line search at the full
    # 200 iterations for every candidate refinement, which multiplies into
    # minutes per sweep.  40 keeps normal runs identical (the gradient /
    # change tolerances trigger far earlier) while bounding the worst case.
    head.hdp.lbfgs_max_iter = 40
    if cfg["b0_mode"] == "calibrate":
        head.regimes.calibrate_b0_from_data(
            torch.tensor(np.concatenate(seqs, 0), dtype=DT).unsqueeze(0),
            sF=cfg["sF"])

    with Timer() as tm:
        blk = args.init_block if args.init_block is not None else cfg.get("init_block")
        print(f"[shs] init: contiguous blocks, block_len="
              f"{blk if blk is not None else 'auto(~T//2K)'}")
        init_from_random_blocks(head, corpus, K=cfg["K0"], seed=args.seed,
                                block_len=blk)
        out = fit_offline_corpus(head, corpus, laps=cfg["laps"],
                                 sweep_every=cfg["sweep_every"],
                                 sweep_kwargs=cfg["sweep"], verbose=True)
        with torch.no_grad():
            z_pred = np.concatenate(
                [head.regime_inference(c[1], c[2], is_first=c[3])[0][0]
                 .argmax(-1).cpu().numpy() for c in corpus])

    z_true = np.concatenate(bundle["z_true"])
    m, _ = all_metrics(z_true, z_pred)
    print(f"[shs] {args.dataset}: K={head.K} used={m['K_used']} "
          f"hamming={m['hamming']:.3f} m2o={m['m2o']:.3f} "
          f"nmi={m['nmi']:.3f} ({tm.elapsed:.0f}s)")

    params = dict(vars(args), **{k: v for k, v in cfg.items() if k != "sweep"},
                  sweep=json.dumps(cfg["sweep"]), b0=b0, D_input=D,
                  K_final=int(head.K), K_trace=out["K_trace"])
    save_result(bundle["name"], args.tag, z_pred, bundle["doc_range"],
                tm.elapsed, params, objective=out["bounds"],
                x_latent=(np.concatenate(seqs, 0) if cfg["pca"] == 2 else None))


if __name__ == "__main__":
    main()
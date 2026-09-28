import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets
from io_utils import Timer, save_result

try:
    import ssm
    import autograd.numpy.random as npr
except ImportError as e:
    sys.exit(
        f"[rslds-ssm] missing dependency: {e}\n"
        "Install it on a LOGIN node, into the same venv as the rest of the "
        "sweep:\n"
        "    pip install --no-index 'cython<3' 'numpy<2' autograd tqdm\n"
        "    pip install --no-build-isolation "
        "git+https://github.com/lindermanlab/ssm.git")

VARIANTS = {
    "rslds": "recurrent",
    "rslds-ro": "recurrent_only",
    "sticky-rslds": "recurrent",
    "sticky-rslds-ro": "recurrent_only",
}
SUBSTITUTED = {"sticky-rslds", "sticky-rslds-ro"}

DEFAULTS = dict(
    fhn=dict(K=4, dlatent=2, iters=100, variant="rslds"),
    nascar=dict(K=4, dlatent=2, iters=100, variant="rslds"),
    toyark13=dict(K=16, dlatent=3, iters=100, variant="rslds"),
    mocap6=dict(K=16, dlatent=6, iters=100, variant="rslds"),
)


def floor_emission_noise(model, datas, rel_floor):
    em = model.emissions
    if not hasattr(em, "inv_etas"):
        return 0, None
    chan_var = np.var(np.concatenate(datas, 0), axis=0)
    lo = np.log(np.maximum(rel_floor * chan_var, 1e-12))
    inv = np.asarray(em.inv_etas, dtype=float)
    bad = int(np.sum(~np.isfinite(inv) | (inv < lo)))
    inv = np.where(np.isfinite(inv), inv, lo)
    em.inv_etas = np.maximum(inv, lo)
    return bad, float(np.exp(lo.min()))


def fit_slds(datas, N, K, D, transitions, emissions, iters, seed, verbose,
             var_floor):
    """Fit and return (z per sequence, x per sequence, elbo trace)."""
    npr.seed(seed)
    model = ssm.SLDS(N, K, D,
                     transitions=transitions,
                     dynamics="gaussian",
                     emissions=emissions,
                     single_subspace=True)
    model.initialize(datas)

    bad, floor = floor_emission_noise(model, datas, var_floor)
    if bad:
        print(f"[rslds-ssm] floored {bad} emission variance(s) to >= "
              f"{floor:.3e} (PCA residual was zero; expected when "
              f"D_latent == D_obs)")

    try:
        elbos, posterior = model.fit(
            datas, method="laplace_em",
            variational_posterior="structured_meanfield",
            initialize=False, num_iters=iters, verbose=verbose)
    except TypeError as exc:
        print(f"[rslds-ssm] fit() rejected a kwarg ({exc}); retrying minimal")
        elbos, posterior = model.fit(
            datas, method="laplace_em",
            variational_posterior="structured_meanfield",
            num_iters=iters)
    except AssertionError:
        raise SystemExit(
            "[rslds-ssm] Laplace-EM hit a non-finite expected log joint.\n"
            f"  Current --var-floor is {var_floor:g}.  Raise it (try 1e-2) "
            "or lower --dlatent below D_obs so the PCA residual is nonzero.")

    elbos = np.asarray(elbos, dtype=np.float64)
    if not np.isfinite(elbos).any():
        raise SystemExit(
            "[rslds-ssm] every ELBO is non-finite; raise --var-floor "
            f"(currently {var_floor:g}) or lower --dlatent.")

    xs = posterior.mean_continuous_states
    zs = [model.most_likely_states(x, y) for x, y in zip(xs, datas)]
    return zs, xs, elbos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(datasets.LOADERS))
    ap.add_argument("--nseq", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--dlatent", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None,
                    help="Laplace-EM iterations")
    ap.add_argument("--samples", type=int, default=None,
                    help="accepted for CLI parity with run_rslds.py; treated "
                         "as --iters")
    ap.add_argument("--emissions", default=None,
                    help="ssm emissions class (default gaussian_orthog, or "
                         "gaussian when D_obs == D_latent)")
    ap.add_argument("--variant", default=None, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0,
                    help="nascar: pins the synthetic data realisation (shared "
                         "across models and seeds); --seed varies "
                         "initialisation only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--var-floor", type=float, default=1e-3,
                    help="emission noise variance floor, relative to each "
                         "channel's variance; needed whenever D_latent == "
                         "D_obs, where the PCA init residual is exactly zero")
    ap.add_argument("--tag", default=None, help="result file stem")
    args = ap.parse_args()

    cfg = dict(DEFAULTS[args.dataset])
    for k in ("K", "dlatent", "variant"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.iters is not None:
        cfg["iters"] = args.iters
    elif args.samples is not None:
        cfg["iters"] = args.samples
    if args.quick:
        cfg["iters"] = min(cfg["iters"], 25)
    tag = args.tag or "rslds"

    load_kw = {}
    if args.dataset == "fhn":
        load_kw["n_seq"] = args.nseq or 6
    if args.dataset == "toyark13":
        load_kw["n_seq"] = args.nseq or 12
    if args.dataset == "nascar":
        load_kw["n_seq"] = args.nseq or 5
        load_kw["seed"] = args.data_seed

    bundle = datasets.load(args.dataset, **load_kw)
    datas = [np.ascontiguousarray(s, dtype=np.float64) for s in bundle["seqs"]]
    N, D, K = datas[0].shape[1], cfg["dlatent"], cfg["K"]
    if N < D:
        sys.exit(f"[rslds-ssm] D_latent={D} exceeds D_obs={N}; lower --dlatent")

    emissions = args.emissions or ("gaussian" if N == D else "gaussian_orthog")
    transitions = VARIANTS[cfg["variant"]]
    if cfg["variant"] in SUBSTITUTED:
        print(f"[rslds-ssm] NOTE {cfg['variant']} has no sticky+recurrent "
              f"class in ssm; using '{transitions}' and recording the "
              f"substitution in params")

    print(f"[rslds-ssm] {args.dataset}: {len(datas)} seqs, D_obs={N}, "
          f"K={K}, D_lat={D}, transitions={transitions}, "
          f"emissions={emissions}, {cfg['iters']} Laplace-EM iters")
    if N == D:
        print(f"[rslds-ssm] D_latent == D_obs, so the emission is invertible "
              f"and the PCA residual is zero; var_floor={args.var_floor:g} "
              f"applies")

    with Timer() as tm:
        zs, xs, elbos = fit_slds(datas, N, K, D, transitions, emissions,
                                 cfg["iters"], args.seed, args.verbose,
                                 args.var_floor)

    z_pred = np.concatenate([np.asarray(z, dtype=np.int64) for z in zs])
    params = dict(model="rslds", implementation="lindermanlab/ssm",
                  inference="laplace_em/structured_meanfield",
                  requested_variant=cfg["variant"], transitions=transitions,
                  substituted=cfg["variant"] in SUBSTITUTED,
                  emissions=emissions, K=K, dlatent=D, iters=cfg["iters"],
                  seed=args.seed, data_seed=args.data_seed,
                  var_floor=args.var_floor, quick=args.quick,
                  ssm_commit="eb6c8aa33e5311d3564075807dec340759dd8081",
                  ssm_version=getattr(ssm, "__version__", "unknown"))

    save_result(bundle["name"], tag, z_pred, bundle["doc_range"],
                tm.elapsed, params, objective=elbos,
                x_latent=(np.concatenate(xs, 0) if D == 2 else None))

    K_used = int(np.unique(z_pred).size)
    print(f"[rslds-ssm] done in {tm.elapsed:.1f}s, K_used={K_used}, "
          f"final ELBO={elbos[-1]:.1f} (first {elbos[0]:.1f}, "
          f"delta over last 10: {elbos[-1] - elbos[max(0, len(elbos) - 11)]:.1f})")


if __name__ == "__main__":
    main()
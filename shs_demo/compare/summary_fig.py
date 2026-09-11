"""One unified summary figure for the offline comparison suite.

Layout: one column per dataset (fhn | nascar | toyark13 | mocap6), and inside
each column, top to bottom: one observation channel of a representative
sequence, then aligned regime ribbons -- ground truth first, then one ribbon
per model (shs, trslds, rslds).  Predicted labels are Hungarian-matched to the
truth so colours agree across ribbons (same convention as make_figures.py);
each model ribbon is annotated with corpus-level Hamming distance and K_used.
A metrics CSV (dataset x model) is written next to the figure.

Two modes:
  real:  python summary_fig.py --datasets fhn nascar toyark13 mocap6 --out offline_summary.pdf
         reads results/<dataset>/<model>.npz written by run_shs / run_rslds /
         run_trslds (io_utils schema) and ground truth from datasets.load().
  mock:  python summary_fig.py --mock --out mock.png
         self-contained synthetic preview of the exact final layout; no repo
         imports, no results needed.  Use it to sign off on the figure before
         burning compute.  run_all.py calls this
         automatically after a sweep.

Metric functions mirror compare/metrics.py (Hungarian Hamming, many-to-one).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

CMAP, VMAX = "tab20", 19          # keep identical to plots.py ribbon colours
MODEL_LABEL = {"shs": "SHS (ours)", "trslds": "TrSLDS", "rslds": "rSLDS"}
DATASET_TITLE = {"fhn": "FitzHugh--Nagumo", "nascar": "NASCAR",
                 "toyark13": "ToyARK13", "mocap6": "MoCap6"}


# ---------------------------------------------------------------- style
def set_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "axes.edgecolor": "0.15",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })


# ---------------------------------------------------------------- metrics
def _contingency(z_true, z_pred):
    z_true = np.asarray(z_true).ravel().astype(int)
    z_pred = np.asarray(z_pred).ravel().astype(int)
    n = int(max(z_true.max(), z_pred.max())) + 1
    C = np.zeros((n, n), dtype=np.int64)
    np.add.at(C, (z_true, z_pred), 1)
    return C


def hamming(z_true, z_pred):
    C = _contingency(z_true, z_pred)
    r, c = linear_sum_assignment(-C)
    dist = 1.0 - C[r, c].sum() / np.asarray(z_true).size
    return float(dist), {int(cc): int(rr) for rr, cc in zip(r, c)}


def many_to_one(z_true, z_pred):
    C = _contingency(z_true, z_pred)
    return float(C.max(axis=0).sum() / np.asarray(z_true).size)


def _remap(z, mapping):
    z = np.asarray(z).astype(int)
    lut = np.array([mapping.get(k, k) for k in range(int(z.max()) + 1)])
    return lut[z]


# ---------------------------------------------------------------- spec
# spec = list of dataset dicts:
#   name, title, obs (T,), z_true (T,), rows = [(model, z_pred_seq (T,),
#   corpus_metrics dict)], with z_pred_seq already colour-remapped.
def build_from_results(names, models, seq_idx=0):
    sys.path.insert(0, str(HERE))
    import datasets as D          # noqa: E402  (repo-local)
    import io_utils as IO         # noqa: E402

    spec = []
    for name in names:
        try:
            bundle = D.load(name)
        except Exception as exc:           # missing data artifact etc.
            print(f"  [warn] {name}: dataset unavailable "
                  f"({type(exc).__name__}); column skipped")
            continue
        z_true_all = np.concatenate(bundle["z_true"], 0).astype(int)
        z_true_all -= z_true_all.min()
        dr = bundle["doc_range"]
        s0, s1 = int(dr[seq_idx]), int(dr[seq_idx + 1])
        obs = np.concatenate(bundle["seqs"], 0)[s0:s1, 0]

        found = IO.load_results(name)
        rows = []
        for model in models:
            cands = [(k, v) for k, v in found.items()
                     if k == model or k.startswith(model)]
            if not cands:
                print(f"  [warn] {name}: no result for '{model}', skipping")
                continue
            scored = []
            for k, v in cands:
                zp = np.asarray(v["z_pred"]).astype(int)
                h, mp = hamming(z_true_all, zp)
                scored.append((h, k, v, zp, mp))
            scored.sort(key=lambda t: t[0])           # median-Hamming seed is
            ham, tag, hit, z_pred_all, mapping = scored[len(scored) // 2]
            met = dict(hamming=ham, m2o=many_to_one(z_true_all, z_pred_all),
                       K_used=int(np.unique(z_pred_all).size),
                       wall_time=float(hit.get("wall_time", np.nan)),
                       n_seeds=len(scored), tag=tag)  # ...the drawn ribbon
            rows.append((model, _remap(z_pred_all, mapping)[s0:s1], met))
        spec.append(dict(name=name, title=DATASET_TITLE.get(name, name),
                         obs=obs, z_true=z_true_all[s0:s1], rows=rows,
                         K_true=int(bundle["K_true"])))
    return spec


# ---------------------------------------------------------------- mock
def _sticky_chain(T, K, p_stay, rng):
    z = np.empty(T, dtype=int)
    z[0] = rng.integers(K)
    for t in range(1, T):
        z[t] = z[t - 1] if rng.random() < p_stay else rng.integers(K)
    return z


def _mock_obs(z, rng):
    means = rng.normal(0, 1.6, size=int(z.max()) + 1)
    x = np.zeros(len(z))
    for t in range(1, len(z)):
        x[t] = 0.9 * x[t - 1] + 0.1 * means[z[t]] + 0.05 * rng.normal()
    return x + means[z] * 0.4


def _corrupt(z, rng, jitter, err_rate, oversplit=False):
    K = int(z.max()) + 1
    zp = z.copy()
    cps = np.flatnonzero(np.diff(z)) + 1
    for c in cps:                                   # boundary jitter
        j = int(rng.integers(-jitter, jitter + 1))
        lo, hi = sorted((c, max(1, min(len(z) - 1, c + j))))
        zp[lo:hi] = zp[max(lo - 1, 0)]
    n_err = int(err_rate * len(z) / 12)             # short wrong segments
    for _ in range(n_err):
        a = int(rng.integers(0, len(z) - 12))
        zp[a:a + int(rng.integers(4, 12))] = rng.integers(K)
    if oversplit:                                    # K_used > K_true
        tgt = int(rng.integers(K))
        m = zp == tgt
        zp[m & (np.arange(len(z)) % 2 == 0)] = K    # new spurious label
    perm = rng.permutation(int(zp.max()) + 1)       # scramble label ids
    return perm[zp]


def mock_spec(names=("fhn", "nascar", "toyark13", "mocap6")):
    cfg = dict(fhn=(1200, 4, 0.988), nascar=(1600, 4, 0.992),
               toyark13=(1000, 13, 0.982), mocap6=(900, 12, 0.985))
    noise = dict(shs=(2, 0.02, False), trslds=(4, 0.06, False),
                 rslds=(5, 0.10, True))
    spec = []
    for i, name in enumerate(names):
        T, K, p = cfg.get(name, (1000, 6, 0.985))
        rng = np.random.default_rng(7 + i)
        z = _sticky_chain(T, K, p, rng)
        rows = []
        for model in ("shs", "trslds", "rslds"):
            j, e, o = noise[model]
            zp = _corrupt(z, rng, j, e, o)
            ham, mapping = hamming(z, zp)
            met = dict(hamming=ham, m2o=many_to_one(z, zp),
                       K_used=int(np.unique(zp).size), wall_time=np.nan)
            rows.append((model, _remap(zp, mapping), met))
        spec.append(dict(name=name, title=DATASET_TITLE.get(name, name),
                         obs=_mock_obs(z, rng), z_true=z, rows=rows,
                         K_true=K))
    return spec


# ---------------------------------------------------------------- figure
def _ribbon(ax, z):
    ax.imshow(np.asarray(z)[None, :], aspect="auto", cmap=CMAP,
              vmin=0, vmax=VMAX, interpolation="nearest")
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def render(spec, out, title=None):
    if not spec:
        raise SystemExit("summary_fig: no dataset columns available to draw")
    set_style()
    n_ds = len(spec)
    n_models = max(len(d["rows"]) for d in spec)
    units = 2.4 + 1.0 * (1 + n_models)               # obs + truth + models
    fig = plt.figure(figsize=(3.0 * n_ds, 0.30 * units + 0.55))
    outer = gridspec.GridSpec(1, n_ds, wspace=0.10, figure=fig)

    for col, d in enumerate(spec):
        heights = [2.4] + [1.0] * (1 + len(d["rows"]))
        inner = gridspec.GridSpecFromSubplotSpec(
            len(heights), 1, subplot_spec=outer[col],
            height_ratios=heights, hspace=0.14)
        T = len(d["z_true"])

        ax = fig.add_subplot(inner[0])
        ax.plot(np.arange(T), d["obs"], color="0.2", lw=0.55)
        ax.set_xlim(0, T - 1)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if col == 0:
            ax.set_ylabel("obs", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{d['title']}  ($K_{{\\mathrm{{true}}}}={d['K_true']}$)",
                     fontsize=9.5, pad=3)

        ax = fig.add_subplot(inner[1])
        _ribbon(ax, d["z_true"])
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("True", fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=14)

        for r, (model, z_seq, met) in enumerate(d["rows"]):
            ax = fig.add_subplot(inner[2 + r])
            _ribbon(ax, z_seq)
            last = (r == len(d["rows"]) - 1)
            if last:
                ax.set_xlabel("$t$", fontsize=8, labelpad=1)
                ax.tick_params(labelsize=6.2, length=2)
                ax.xaxis.set_major_locator(MaxNLocator(4))
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(MODEL_LABEL.get(model, model), fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=14)
            ax.text(0.994, 0.5,
                    f"H={met['hamming']:.2f}  K={met['K_used']}",
                    transform=ax.transAxes, fontsize=6.0, va="center",
                    ha="right", color="0.1",
                    bbox=dict(fc="white", ec="none", alpha=0.78,
                              boxstyle="round,pad=0.15"))

    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def write_csv(spec, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "hamming", "m2o", "K_used",
                    "wall_time_s", "n_seeds", "tag"])
        for d in spec:
            for model, _z, met in d["rows"]:
                w.writerow([d["name"], model, f"{met['hamming']:.4f}",
                            f"{met['m2o']:.4f}", met["K_used"],
                            f"{met.get('wall_time', float('nan')):.1f}",
                            met.get("n_seeds", 1), met.get("tag", model)])
    print(f"wrote {path}")


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--datasets", nargs="*",
                    default=["fhn", "nascar", "toyark13", "mocap6"])
    ap.add_argument("--models", nargs="*", default=["shs", "trslds", "rslds"])
    ap.add_argument("--seq", type=int, default=0,
                    help="which sequence to draw in the ribbons")
    ap.add_argument("--out", default="offline_summary.pdf")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    spec = (mock_spec(tuple(args.datasets)) if args.mock else
            build_from_results(args.datasets, args.models, args.seq))
    render(spec, args.out)
    write_csv(spec, args.csv or
              str(pathlib.Path(args.out).with_suffix("")) + "_metrics.csv")


if __name__ == "__main__":
    main()
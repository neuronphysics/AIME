#!/usr/bin/env python3
"""Aggregate results/<dataset>/*.npz into comparison tables and figures.

Runs in either environment (results written by the legacy rSLDS env aggregate
identically), touches no model code, and never re-fits anything.

Per dataset it produces

    results/<dataset>/table.csv     one row per run (seed-level)
    results/<dataset>/table.md      one row per model, mean +/- std over seeds
    results/<dataset>/table.tex     same, booktabs body (with --latex)
    figures/<dataset>_ribbons.png   the regime-change figure: truth ribbon,
                                    one Hungarian-matched ribbon per model,
                                    and an observation channel per sequence
    figures/<dataset>_latents.png   nascar only: inferred 2-D latents coloured
                                    by regime next to the true track
    figures/<dataset>_objectives.png  per-model training objectives, own axes

and figures/summary.csv across datasets.

Multi-seed convention: runs tagged ``<model>_seed<k>`` are grouped under
``<model>``; the table reports mean +/- std, the ribbon shows the seed with
median Hamming.  Headline metrics are label-based (Hamming after Hungarian
matching, many-to-one purity, NMI, ARI) because raw objectives are not
comparable across model classes -- see metrics.py.

Examples
--------
    python make_figures.py --dataset nascar
    python make_figures.py --dataset all --latex
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets                                    # noqa: E402
from io_utils import RESULTS, load_results         # noqa: E402
from metrics import all_metrics                    # noqa: E402
import plots                                       # noqa: E402

FIGURES = HERE / "figures"
_SEED_RE = re.compile(r"^(?P<model>.+?)(?:_seed(?P<seed>\d+))?$")

# Preferred row order in tables and ribbon stacking (unknown models follow,
# alphabetically).
MODEL_ORDER = ["shs", "rslds", "rslds_ro", "rslds_sticky", "trslds"]


def _params(run):
    try:
        return json.loads(str(run["params"]))
    except Exception:
        return {}


def _split_by_doc_range(z, dr):
    return [np.asarray(z[dr[i]:dr[i + 1]]) for i in range(len(dr) - 1)]


def _load_bundle(dataset, runs):
    """Load ground truth once, big enough for the largest run."""
    S_max = max(len(r["doc_range"]) - 1 for r in runs.values())
    kw = {}
    if dataset in ("toyark13", "nascar"):
        kw["n_seq"] = S_max
    if dataset == "nascar":
        seeds = {int(_params(r).get("data_seed",
                              _params(r).get("seed", 0)))
                 for r in runs.values()}
        if len(seeds) > 1:
            print(f"[warn] nascar runs disagree on data seed {sorted(seeds)}; "
                  f"using {min(seeds)} for ground truth")
        kw["seed"] = min(seeds)
    return datasets.load(dataset, **kw)


def _score_runs(dataset, runs, bundle):
    """Per-run metrics against each run's own slice of the ground truth."""
    rows = []
    for tag, run in sorted(runs.items()):
        m = _SEED_RE.match(tag)
        dr = np.asarray(run["doc_range"], dtype=int)
        S = len(dr) - 1
        z_true = np.concatenate(bundle["z_true"][:S])
        z_pred = np.asarray(run["z_pred"], dtype=int)
        if z_true.shape != z_pred.shape:
            print(f"[warn] {dataset}/{tag}: truth/pred length mismatch "
                  f"{z_true.shape} vs {z_pred.shape}; skipping")
            continue
        met, mapping = all_metrics(z_true, z_pred)
        rows.append(dict(tag=tag, model=m.group("model"),
                         seed=(int(m.group("seed")) if m.group("seed")
                               else _params(run).get("seed", 0)),
                         n_seq=S, wall_time=float(run.get("wall_time", np.nan)),
                         mapping=mapping, run=run, **met))
    return rows


def _order_key(model):
    return (MODEL_ORDER.index(model) if model in MODEL_ORDER
            else len(MODEL_ORDER), model)


def _aggregate(rows):
    """Group seed-level rows by model; mean +/- std for the shared metrics."""
    groups = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    agg = []
    for model in sorted(groups, key=_order_key):
        g = groups[model]
        entry = dict(model=model, n_runs=len(g),
                     n_seq="/".join(sorted({str(r["n_seq"]) for r in g})))
        for key in ("hamming", "m2o", "nmi", "ari", "wall_time"):
            vals = np.array([r[key] for r in g], dtype=float)
            entry[key] = float(np.nanmean(vals))
            entry[key + "_std"] = (float(np.nanstd(vals, ddof=1))
                                   if len(vals) > 1 else float("nan"))
        entry["K_used"] = "/".join(str(r["K_used"]) for r in
                                   sorted(g, key=lambda r: r["seed"]))
        # representative run: median hamming (ties -> first)
        entry["_rep"] = sorted(g, key=lambda r: r["hamming"])[len(g) // 2]
        agg.append(entry)
    return agg


def _fmt(mean, std, prec=3):
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} +/- {std:.{prec}f}"


def _write_tables(dataset, rows, agg, latex=False):
    out = RESULTS / dataset
    # seed-level CSV
    csv = out / "table.csv"
    cols = ["tag", "model", "seed", "n_seq", "K_used", "hamming", "m2o",
            "nmi", "ari", "wall_time"]
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"table  -> {csv}")

    # model-level markdown
    md = out / "table.md"
    with open(md, "w") as f:
        f.write(f"# {dataset}: segmentation vs ground truth\n\n")
        f.write("Labels Hungarian-matched per run; mean +/- std over seeds. "
                "Hamming lower is better; the rest higher is better. "
                "Model objectives are per-model quantities and are not "
                "listed here because they are not comparable across model "
                "classes (see metrics.py).\n\n")
        f.write("| model | runs | K used | Hamming | many-to-one | NMI | ARI "
                "| wall (s) |\n|---|---|---|---|---|---|---|---|\n")
        for a in agg:
            f.write(f"| {a['model']} | {a['n_runs']} | {a['K_used']} | "
                    f"{_fmt(a['hamming'], a['hamming_std'])} | "
                    f"{_fmt(a['m2o'], a['m2o_std'])} | "
                    f"{_fmt(a['nmi'], a['nmi_std'])} | "
                    f"{_fmt(a['ari'], a['ari_std'])} | "
                    f"{a['wall_time']:.0f} |\n")
    print(f"table  -> {md}")

    if latex:
        tex = out / "table.tex"
        with open(tex, "w") as f:
            f.write("% generated by make_figures.py -- booktabs body\n"
                    "\\begin{tabular}{lccccc}\n\\toprule\n"
                    "model & $K$ used & Hamming $\\downarrow$ & "
                    "many-to-one $\\uparrow$ & NMI $\\uparrow$ & "
                    "ARI $\\uparrow$ \\\\\n\\midrule\n")
            for a in agg:
                f.write(f"{a['model'].replace('_', '-')} & {a['K_used']} & "
                        f"{_fmt(a['hamming'], a['hamming_std'])} & "
                        f"{_fmt(a['m2o'], a['m2o_std'])} & "
                        f"{_fmt(a['nmi'], a['nmi_std'])} & "
                        f"{_fmt(a['ari'], a['ari_std'])} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"table  -> {tex}")


def _figures(dataset, bundle, agg, n_show, obs_dim):
    FIGURES.mkdir(exist_ok=True)

    # 1. regime ribbons (the regime-change figure)
    model_rows, S_common = [], min(3 if n_show is None else n_show,
                                   len(bundle["seqs"]))
    for a in agg:
        rep = a["_rep"]
        dr = np.asarray(rep["run"]["doc_range"], dtype=int)
        z_seqs = _split_by_doc_range(np.asarray(rep["run"]["z_pred"], int), dr)
        S_common = min(S_common, len(z_seqs))
        label = a["model"] if a["n_runs"] == 1 else \
            f"{a['model']}\n(seed {rep['seed']})"
        model_rows.append((label, z_seqs, rep["mapping"]))
    if model_rows and S_common:
        plots.regime_ribbons(
            bundle, model_rows, FIGURES / f"{dataset}_ribbons.png",
            n_show=S_common, obs_dim=obs_dim,
            title=f"{dataset}: regime changes -- truth vs models "
                  "(labels Hungarian-matched; representative seed)")

    # 1b. state usage over the whole corpus
    if model_rows:
        plots.state_usage(bundle, model_rows,
                          FIGURES / f"{dataset}_usage.png")

    # 2. nascar latent partition
    if bundle.get("x_true") is not None:
        pts = []
        for a in agg:
            rep = a["_rep"]
            run = rep["run"]
            if "x_latent" not in run:
                continue
            x = np.asarray(run["x_latent"], dtype=float)
            z = np.asarray(run["z_pred"], dtype=int)
            if x.ndim == 2 and x.shape[1] == 2 and x.shape[0] == z.shape[0]:
                pts.append((a["model"], x, z, rep["mapping"]))
        if pts:
            plots.latent_partition(bundle, pts,
                                   FIGURES / f"{dataset}_latents.png")

    # 3. objective traces (own scales)
    traces = {a["model"]: (np.asarray(a["_rep"]["run"]["objective"])
                           if "objective" in a["_rep"]["run"] else None)
              for a in agg}
    plots.objective_traces(traces, FIGURES / f"{dataset}_objectives.png",
                           dataset)


def process(dataset, n_show=None, obs_dim=0, latex=False):
    runs = load_results(dataset)
    if not runs:
        print(f"[{dataset}] no results in {RESULTS / dataset} -- run the "
              "runners first")
        return None
    bundle = _load_bundle(dataset, runs)
    rows = _score_runs(dataset, runs, bundle)
    if not rows:
        return None
    agg = _aggregate(rows)
    _write_tables(dataset, rows, agg, latex=latex)
    _figures(dataset, bundle, agg, n_show, obs_dim)
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=sorted(datasets.LOADERS) + ["all"])
    ap.add_argument("--n-show", type=int, default=None,
                    help="sequences per ribbons figure (default 3; mocap6 has "
                         "6, and each sequence visits only part of the 12-state "
                         "vocabulary, so use 6 to see them all)")
    ap.add_argument("--obs-dim", type=int, default=0,
                    help="observation channel drawn under the ribbons")
    ap.add_argument("--latex", action="store_true",
                    help="also write results/<dataset>/table.tex")
    args = ap.parse_args()

    names = sorted(datasets.LOADERS) if args.dataset == "all" else [args.dataset]
    summary = []
    for d in names:
        agg = process(d, args.n_show, args.obs_dim, args.latex)
        if agg:
            for a in agg:
                summary.append((d, a["model"], a["n_runs"], a["K_used"],
                                a["hamming"], a["hamming_std"], a["m2o"],
                                a["nmi"], a["ari"]))
    if summary:
        FIGURES.mkdir(exist_ok=True)
        p = FIGURES / "summary.csv"
        with open(p, "w") as f:
            f.write("dataset,model,n_runs,K_used,hamming,hamming_std,"
                    "m2o,nmi,ari\n")
            for row in summary:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"summary -> {p}")


if __name__ == "__main__":
    main()
import argparse
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets as DS
import io_utils as IO
from metrics import all_metrics

MODEL_TAGS = {
    "shs": ["shs", "shsPersist"],
    "trslds": ["trslds"],
    "rslds": ["rslds"],
    "rslds_ro": ["rslds_ro"],
    "rslds_sticky": ["rslds_sticky"],
}
LABEL = {"shs": "SHS (ours)", "trslds": "TrSLDS", "rslds": "rSLDS",
         "rslds_ro": "rSLDS (ro)", "rslds_sticky": "rSLDS (sticky)"}
TITLE = {"fhn": "FitzHugh--Nagumo", "nascar": "NASCAR",
         "toyark13": "ToyARK13", "mocap6": "MoCap6"}
LOAD_KW = {"fhn": dict(n_seq=6), "nascar": dict(n_seq=5, seed=0),
           "toyark13": dict(n_seq=12), "mocap6": {}}
_SEED_RE = re.compile(r"_seed(\d+)$")


def resolve(found, model):
    """(tag, {seed: run}) for one model row; ({}, ) if nothing on disk."""
    for name in MODEL_TAGS.get(model, [model]):
        hits = {}
        for k, v in found.items():
            m = _SEED_RE.search(k)
            stem = _SEED_RE.sub("", k)
            if stem == name:
                hits[int(m.group(1)) if m else 0] = v
        if hits:
            return name, hits
    return model, {}


def collect(dataset, models):
    try:
        bundle = DS.load(dataset, **LOAD_KW.get(dataset, {}))
    except Exception as exc:
        print(f"[warn] {dataset}: cannot load ground truth ({exc}), skipping")
        return None, []
    z_true = np.concatenate(bundle["z_true"]).astype(int)
    z_true -= z_true.min()
    found = IO.load_results(dataset)

    out = []
    for model in models:
        tag, runs = resolve(found, model)
        if not runs:
            print(f"[warn] {dataset}: no runs for '{model}' "
                  f"(looked for {MODEL_TAGS.get(model, [model])})")
            continue
        per_seed = []
        for seed in sorted(runs):
            z = np.asarray(runs[seed]["z_pred"], int)
            if z.shape[0] != z_true.shape[0]:
                print(f"[warn] {dataset}/{tag}_seed{seed}: length "
                      f"{z.shape[0]} != {z_true.shape[0]}, skipped")
                continue
            m, _ = all_metrics(z_true, z)
            m["wall"] = float(runs[seed].get("wall_time", np.nan))
            per_seed.append(m)
        if not per_seed:
            continue
        agg = {k: (float(np.mean([r[k] for r in per_seed])),
                   float(np.std([r[k] for r in per_seed])))
               for k in ("hamming", "m2o", "nmi", "ari", "K_used", "wall")}
        out.append(dict(model=model, tag=tag, n_seeds=len(per_seed), **agg))
    return bundle, out


def fmt(mean, std, places=3, show_std=True):
    if np.isnan(mean):
        return "--"
    if not show_std or std == 0:
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*",
                    default=["fhn", "nascar", "toyark13", "mocap6"])
    ap.add_argument("--models", nargs="*", default=["shs", "trslds", "rslds"])
    ap.add_argument("--out", default="results/comparison",
                    help="path stem; writes .csv, .md and .tex")
    args = ap.parse_args()

    rows = []
    for d in args.datasets:
        bundle, entries = collect(d, args.models)
        if bundle is None:
            continue
        for e in entries:
            e["dataset"] = d
            e["K_true"] = int(bundle["K_true"])
            rows.append(e)

    if not rows:
        sys.exit("no results found -- run the fits first")

    stem = pathlib.Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{stem}.csv", "w") as f:
        f.write("dataset,model,tag,n_seeds,K_true,K_used,K_used_std,"
                "one_to_one,one_to_one_std,m2o,m2o_std,nmi,nmi_std,"
                "ari,ari_std,wall_s\n")
        for r in rows:
            f.write(",".join(str(x) for x in [
                r["dataset"], r["model"], r["tag"], r["n_seeds"], r["K_true"],
                f"{r['K_used'][0]:.2f}", f"{r['K_used'][1]:.2f}",
                f"{1 - r['hamming'][0]:.4f}", f"{r['hamming'][1]:.4f}",
                f"{r['m2o'][0]:.4f}", f"{r['m2o'][1]:.4f}",
                f"{r['nmi'][0]:.4f}", f"{r['nmi'][1]:.4f}",
                f"{r['ari'][0]:.4f}", f"{r['ari'][1]:.4f}",
                f"{r['wall'][0]:.1f}"]) + "\n")
    print(f"csv   -> {stem}.csv")

    with open(f"{stem}.md", "w") as f:
        f.write("# Offline switching-system benchmarks\n\n")
        f.write("Mean ± std over seeds. Labels Hungarian-matched per run. "
                "1-to-1, m2o, NMI and ARI: higher is better.\n\n")
        for d in args.datasets:
            sub = [r for r in rows if r["dataset"] == d]
            if not sub:
                continue
            f.write(f"## {TITLE.get(d, d)}  (K_true = {sub[0]['K_true']})\n\n")
            f.write("| model | seeds | K used | 1-to-1 | m2o | NMI | ARI | wall (s) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in sub:
                f.write(f"| {LABEL.get(r['model'], r['model'])} | {r['n_seeds']} "
                        f"| {fmt(*r['K_used'], places=1)} "
                        f"| {fmt(1 - r['hamming'][0], r['hamming'][1])} "
                        f"| {fmt(*r['m2o'])} | {fmt(*r['nmi'])} "
                        f"| {fmt(*r['ari'])} "
                        f"| {r['wall'][0]:.0f} |\n")
            f.write("\n")
    print(f"md    -> {stem}.md")

    with open(f"{stem}.tex", "w") as f:
        f.write("% requires \\usepackage{booktabs}\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Dataset & Model & $K$ & 1-to-1 & NMI & ARI \\\\\n\\midrule\n")
        for d in args.datasets:
            sub = [r for r in rows if r["dataset"] == d]
            for i, r in enumerate(sub):
                name = TITLE.get(d, d) if i == 0 else ""
                f.write(f"{name} & {LABEL.get(r['model'], r['model'])} "
                        f"& {r['K_used'][0]:.1f} "
                        f"& ${1 - r['hamming'][0]:.3f} \\pm {r['hamming'][1]:.3f}$ "
                        f"& ${r['nmi'][0]:.3f} \\pm {r['nmi'][1]:.3f}$ "
                        f"& ${r['ari'][0]:.3f} \\pm {r['ari'][1]:.3f}$ \\\\\n")
            if sub:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"latex -> {stem}.tex")

    print()
    print(f"{'dataset':10s} {'model':14s} {'tag':16s} {'seeds':>5s} "
          f"{'K':>9s} {'1-to-1':>15s} {'NMI':>15s}")
    for r in rows:
        print(f"{r['dataset']:10s} {LABEL.get(r['model'], r['model']):14s} "
              f"{r['tag']:16s} {r['n_seeds']:5d} "
              f"{r['K_used'][0]:4.1f}/{r['K_true']:<4d} "
              f"{fmt(1 - r['hamming'][0], r['hamming'][1]):>15s} "
              f"{fmt(*r['nmi']):>15s}")

    missing = [(d, m) for d in args.datasets for m in args.models
               if not any(r["dataset"] == d and r["model"] == m for r in rows)]
    if missing:
        print("\nmissing cells (run these before quoting the table):")
        for d, m in missing:
            print(f"  {d} / {m}")


if __name__ == "__main__":
    main()

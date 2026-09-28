from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

KNOWN_ENVS = [
    "acrobot", "cartpole", "cheetah", "cup", "finger", "hopper", "humanoid",
    "pendulum", "quadruped", "reacher", "walker", "dog", "fish", "manipulator",
    "swimmer", "point_mass", "ball_in_cup",
]
SEED_RE = re.compile(r"(?:seed|s)[_-]?(\d+)", re.I)

KNOWN_TASKS = [
    "walk", "run", "hop", "stand", "swingup", "swingup_sparse", "balance",
    "balance_sparse", "catch", "spin", "turn_easy", "turn_hard", "easy",
    "hard", "swim", "reach", "bring_ball", "insert_ball", "fetch", "upright",
    "walk_backwards", "flip", "run_backwards",
]
DEFAULT_TASK = {"walker": "walk", "cheetah": "run", "hopper": "hop",
                "acrobot": "swingup", "pendulum": "swingup"}
METHOD_TOKENS = {"dreamer", "dreamerv3", "shs", "shsrssm", "rssm", "vanilla",
                 "baseline", "compare", "ours", "final", "dmc"}

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#B22222", "#000000"]

METRIC_LABELS = {
    "eval_return": "Evaluation return",
    "train_return": "Training return",
    "eval_episode/score": "Evaluation return",
    "episode/score": "Training return",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman",
                       "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8, "axes.edgecolor": "0.15", "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "grid.color": "0.87", "grid.linewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8", "legend.fancybox": False,
    })


def style_axis(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
    ax.grid(True, which="major")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def pretty_env(env: str) -> str:
    if env.lower() in KNOWN_ENVS:
        return env.replace("_", " ").title()
    head, _, arm = env.rpartition("_")
    if head and head.lower() in KNOWN_ENVS:
        return f"{head.replace('_', ' ').title()} ({arm})"
    return env.replace("_", " ").title()


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def glob_root(pattern: str) -> str:
    """Directory part of a glob pattern up to its first wildcard (for relpaths)."""
    if not any(c in pattern for c in "*?["):
        return pattern
    head = pattern
    while any(c in os.path.basename(head) for c in "*?[") and head not in ("", os.sep):
        head = os.path.dirname(head)
    return head or "."


def find_metric_files(logdir: str, filename: str = "metrics.jsonl") -> list[str]:
    """metrics.jsonl files under a directory, a file, or a glob pattern of run dirs."""
    if any(c in logdir for c in "*?["):
        roots = sorted(glob.glob(logdir))
    else:
        roots = [logdir]
    hits = []
    for r in roots:
        if os.path.isfile(r):
            hits.append(r); continue
        for root, _dirs, files in os.walk(r):
            for f in files:
                if f == filename or f.endswith("_" + filename):
                    hits.append(os.path.join(root, f))
    return sorted(set(hits))


def infer_env(path: str, logdir: str, strip: set[str] = frozenset()) -> str:
    rel = os.path.relpath(path, logdir if os.path.isdir(logdir) else os.path.dirname(logdir))
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    hay = "_".join(parts).lower()
    for env in KNOWN_ENVS:
        if env in hay:
            for task in sorted(KNOWN_TASKS, key=len, reverse=True):
                if re.search(rf"{env}[_-]{task}(?![a-z])", hay):
                    return f"{env}_{task}"
            return f"{env}_{DEFAULT_TASK[env]}" if env in DEFAULT_TASK else env
    tokens = METHOD_TOKENS | {t.lower() for t in strip}
    for p in reversed(parts[:-1]):
        if p in (".", "logs", "logdir"):
            continue
        name = "_".join(t for t in re.split(r"[_-]", p.lower())
                        if t and t not in tokens and not SEED_RE.fullmatch(t))
        if name:
            return name
    return os.path.splitext(parts[-1])[0].replace("_metrics", "")


def variant_label(path: str, strip: set[str] = frozenset()) -> str:
    name = os.path.basename(os.path.dirname(path)).lower()
    toks = [tk for tk in re.split(r"[_-]", name) if tk]
    drop = {"dmc", "seed"} | {tk.lower() for tk in strip} | set(KNOWN_ENVS) | set(KNOWN_TASKS)
    for multi in list(KNOWN_ENVS) + list(KNOWN_TASKS):
        drop |= set(multi.split("_"))
    keep = [tk for tk in toks if tk not in drop and not tk.isdigit() and not SEED_RE.fullmatch(tk)]
    joined = "_".join(keep)
    for m in re.findall(r"[a-z0-9]+-[a-z0-9-]+", name):
        parts = m.split("-")
        if all(pp in keep for pp in parts):
            joined = joined.replace("_".join(parts), m)
    return joined or "base"


def infer_seed(path: str) -> str | None:
    m = SEED_RE.search(path)
    return m.group(1) if m else None


def load_series(path: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    steps, vals, bad = [], [], 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if metric in d and d[metric] is not None and "step" in d:
                try:
                    steps.append(float(d["step"])); vals.append(float(d[metric]))
                except (TypeError, ValueError):
                    bad += 1
    if bad:
        print(f"    ! {bad} unparseable lines skipped in {os.path.basename(path)}",
              file=sys.stderr)
    o = np.argsort(steps, kind="stable")
    s, v = np.asarray(steps)[o], np.asarray(vals)[o]
    if len(s):
        keep = np.concatenate([s[1:] != s[:-1], [True]])
        s, v = s[keep], v[keep]
    return s, v


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def aggregate(series: list, smooth_w: int):
    """Median and IQR of several (steps, values) runs on their common grid."""
    if len(series) == 1:
        _, s, v, _ = series[0]
        y = smooth(v, smooth_w)
        x = s[len(s) - len(y):]
        return x, y, None, None
    grid = np.unique(np.concatenate([s for _, s, _, _ in series]))
    lo = max(s[0] for _, s, _, _ in series)
    hi = min(s[-1] for _, s, _, _ in series)
    grid = grid[(grid >= lo) & (grid <= hi)]
    M = np.vstack([np.interp(grid, s, v) for _, s, v, _ in series])
    med = np.nanmedian(M, 0)
    q1, q3 = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
    mp, q1p, q3p = (smooth(a, smooth_w) for a in (med, q1, q3))
    x = grid[len(grid) - len(mp):]
    return x, mp, q1p, q3p


def final_score(v: np.ndarray, frac: float = 0.25) -> tuple[float, float]:
    k = max(2, int(len(v) * frac))
    return float(np.nanmean(v[-k:])), float(np.nanstd(v[-k:]))



def seed_notes(runs, envs, labels, dropped):
    counts = {env: {lab: len(runs[env][lab]) for lab in labels if runs[env].get(lab)} for env in envs}
    all_n = {n for c in counts.values() for n in c.values()}
    legend_label_of = {lab: lab for lab in labels}
    legend_title, panel_note_of = None, {env: "" for env in envs}
    if len(all_n) == 1:
        legend_title = f"$n = {all_n.pop()}$ seeds per method"
    else:
        per_method = {}
        for lab in labels:
            ns = {counts[env][lab] for env in envs if lab in counts[env]}
            per_method[lab] = ns.pop() if len(ns) == 1 else None
        if all(v is not None for v in per_method.values()):
            legend_label_of = {lab: f"{lab} ($n={per_method[lab]}$)" for lab in labels}
        else:
            for lab in labels:
                ns = [counts[env][lab] for env in envs if lab in counts[env]]
                mode = max(set(ns), key=ns.count)
                legend_label_of[lab] = f"{lab} ($n={mode}$)"
                for env in envs:
                    if lab in counts[env] and counts[env][lab] != mode:
                        panel_note_of[env] += (", " if panel_note_of[env] else "") + f"{lab} $n={counts[env][lab]}$"
    footnote = ""
    if dropped:
        where = defaultdict(set)
        for env in envs:
            for lab, sd in dropped.get(env, []):
                where[(lab, sd)].add(env)
        groups = defaultdict(list)
        for (lab, sd), es in sorted(where.items()):
            scope = "all panels" if es >= set(envs) else ", ".join(pretty_env(e) for e in sorted(es))
            groups[scope].append(f"{lab} seed {sd}")
        parts = []
        for scope, items in groups.items():
            by_lab = defaultdict(list)
            for it in items:
                lab, sd = it.rsplit(" seed ", 1); by_lab[lab].append(sd)
            txt = "; ".join(f"{lab} seed {', '.join(sorted(s))}" for lab, s in by_lab.items())
            parts.append(f"{txt} ({scope})")
        import textwrap
        footnote = "\n".join(textwrap.wrap("Excluded seeds: " + "; ".join(parts), width=120))
    return legend_label_of, legend_title, panel_note_of, footnote


def parse_drops(items) -> list[tuple[str, str, set[str]]]:
    """'LABEL:ENV:SEEDS' -> (label, env substring or '*', {seeds})."""
    out = []
    for item in items or []:
        try:
            label, env, seeds = item.split(":", 2)
        except ValueError:
            sys.exit(f"--drop entries must be LABEL:ENV:SEED[,SEED...], got '{item}'")
        out.append((label.strip(), env.strip().lower(),
                    {x.strip() for x in seeds.split(",") if x.strip()}))
    return out


def is_dropped(label: str, env: str, seed: str | None, drops) -> bool:
    for lab, env_sub, seeds in drops:
        if lab != label:
            continue
        if env_sub != "*" and env_sub not in env.lower():
            continue
        if seed is not None and seed in seeds:
            return True
    return False


def parse_runs(args) -> list[tuple[str, str]]:
    if args.runs:
        out = []
        for item in args.runs:
            if "=" not in item:
                sys.exit(f"--runs entries must be LABEL=PATH, got '{item}'")
            label, path = item.split("=", 1)
            out.append((label.strip(), os.path.expanduser(path.strip())))
        return out
    if args.logdir:
        return [(args.label or "run", os.path.expanduser(args.logdir))]
    sys.exit("give --runs LABEL=PATH [LABEL=PATH ...] or --logdir PATH")


def parse_metrics(items) -> dict[str, str]:
    """'LABEL=METRIC' -> {label: metric}; methods not listed use --metric."""
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--metrics entries must be LABEL=METRIC, got '{item}'")
        label, metric = item.split("=", 1)
        out[label.strip()] = metric.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", metavar="LABEL=PATH",
                    help="one entry per method, e.g. DreamerV3=./logdir SHS-RSSM=./logdir_shs")
    ap.add_argument("--logdir", help="one root directory of runs, plotted as a single method (see --label, --variants)")
    ap.add_argument("--label", default=None, help="label for --logdir")
    ap.add_argument("--filename", default="metrics.jsonl")
    ap.add_argument("--metric", default="eval_return",
                    help="eval_return (default) | train_return | any logged key")
    ap.add_argument("--metrics", nargs="*", metavar="LABEL=METRIC", default=None,
                    help="per-method metric key when it differs from --metric, "
                         "e.g. Recall2Imagine=eval_episode/score")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="substring filter, e.g. --envs cheetah hopper")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window")
    ap.add_argument("--max-steps", type=float, default=None,
                    help="truncate every run at this many environment steps")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="draw every seed instead of median +- IQR")
    ap.add_argument("--drop", nargs="*", metavar="LABEL:ENV:SEEDS", default=None,
                    help="exclude seeds, e.g. SHS-RSSM:walker:2  DreamerV3:*:0,3")
    ap.add_argument("--strip", nargs="*", default=None,
                    help="extra tokens to remove from environment names (method names "
                         "such as dreamer/shs are removed by default)")
    ap.add_argument("--variants", action="store_true",
                    help="with --logdir: treat each run-directory variant (what remains of the "
                         "name after env/task/seed) as its own method with its own colour")
    ap.add_argument("--rename", nargs="*", metavar="OLD=NEW", default=None,
                    help="legend names for variant/method labels, e.g. shs=base shs_p256='proj 256'")
    ap.add_argument("--ncol", type=int, default=3, help="panels per row")
    ap.add_argument("--out", default="returns.pdf")
    ap.add_argument("--csv", default=None, help="also write a summary CSV here")
    args = ap.parse_args()

    methods = parse_runs(args)
    drops = parse_drops(args.drop)
    metric_of = parse_metrics(args.metrics)

    runs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    dropped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    rename = {}
    for item in args.rename or []:
        if "=" not in item:
            sys.exit(f"--rename entries must be OLD=NEW, got '{item}'")
        k, v = item.split("=", 1); rename[k.strip()] = v.strip()
    variant_order: list[str] = []
    for label0, root in methods:
        metric = metric_of.get(label0, args.metric)
        files = find_metric_files(root, args.filename)
        if not files:
            print(f"! no '{args.filename}' under {root} ({label0})", file=sys.stderr)
            continue
        base_root = glob_root(root)
        for f in files:
            label = label0
            if args.variants:
                label = rename.get(variant_label(f, set(args.strip or [])), None) \
                    or variant_label(f, set(args.strip or []))
                if label not in variant_order:
                    variant_order.append(label)
            else:
                label = rename.get(label0, label0)
            env = infer_env(f, base_root, set(args.strip or []))
            if args.envs and not any(e.lower() in env.lower() or e.lower() in f.lower()
                                     for e in args.envs):
                continue
            seed = infer_seed(f)
            if is_dropped(label, env, seed, drops):
                print(f"  {label:<12} {env:<18} seed={seed:<4} dropped (--drop)")
                dropped[env].append((label, seed))
                continue
            s, v = load_series(f, metric)
            if len(s) == 0:
                print(f"    ! no '{metric}' in {f}", file=sys.stderr)
                continue
            if args.max_steps:
                m = s <= args.max_steps
                s, v = s[m], v[m]
            runs[env][label].append((infer_seed(f), s, v, f))
            print(f"  {label:<12} {env:<18} seed={infer_seed(f) or '-':<4} "
                  f"n={len(s):<6} steps<= {s[-1]:>12,.0f}  [{metric}]")

    if not runs:
        print("Nothing matched.", file=sys.stderr)
        return 1

    set_style()
    envs = sorted(runs)
    if args.variants:
        labels = sorted(variant_order, key=lambda s: (len(s), s))
    else:
        labels = [rename.get(lab, lab) for lab, _ in methods if any(rename.get(lab, lab) in runs[e] for e in envs)]
    colors = {lab: OKABE_ITO[i % len(OKABE_ITO)] for i, lab in enumerate(labels)}
    ylab = metric_label(args.metric)

    n = len(envs)
    ncol = min(args.ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 2.9 * nrow),
                             squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    legend_label_of, legend_title, panel_note_of, footnote = seed_notes(runs, envs, labels, dropped)

    summary = []
    for i, env in enumerate(envs):
        ax = axes[i]
        for lab in labels:
            series = runs[env].get(lab)
            if not series:
                continue
            c = colors[lab]
            if args.no_aggregate:
                for sd, s, v, _ in series:
                    y = smooth(v, args.smooth); x = s[len(s) - len(y):]
                    ax.plot(x / 1e6, y, color=c, lw=1.4, alpha=.8, zorder=3,
                            label=lab if sd == series[0][0] else None)
                v_ref = series[0][2]
            else:
                x, med, q1, q3 = aggregate(series, args.smooth)
                if q1 is not None:
                    ax.fill_between(x / 1e6, q1, q3, color=c, alpha=.16, lw=0, zorder=2)
                ax.plot(x / 1e6, med, color=c, lw=2.0, zorder=3, label=lab)
                v_ref = med
            fin, sd_ = final_score(v_ref)
            summary.append(dict(env=env, method=lab, n_runs=len(series),
                                steps=float(max(s[-1] for _, s, _, _ in series)),
                                final=fin, std=sd_, best=float(np.nanmax(v_ref))))
        style_axis(ax)
        if panel_note_of[env]:
            ax.text(0.98, 0.03, panel_note_of[env], transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color="0.35",
                    bbox=dict(fc="white", ec="none", alpha=0.8, boxstyle="round,pad=0.15"))
        ax.set_title(pretty_env(env), loc="left", fontweight="bold")
        ax.set_xlabel(r"Environment steps ($\times 10^6$)" if i // ncol == nrow - 1 or i + ncol >= n else "")
        ax.set_ylabel(ylab if i % ncol == 0 else "")
        ax.margins(x=0.01)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, seen = [], set()
    for ax in axes[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append((labels.index(l) if l in labels else 99, h, l))
    handles.sort(key=lambda x: x[0])
    title_lines = [s for s in (legend_title, footnote) if s]
    leg = fig.legend([h for _, h, _ in handles], [legend_label_of.get(l, l) for _, _, l in handles],
                     loc="outside lower center", ncol=min(len(handles), 6),
                     title="\n".join(title_lines) if title_lines else None,
                     frameon=False, handlelength=2.2, columnspacing=1.6)
    if title_lines:
        leg.get_title().set_fontsize(8.5); leg.get_title().set_color("0.3")
        leg._legend_box.align = "center"; leg.get_title().set_multialignment("center")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    print(f"\n{'env':<18}{'method':<14}{'runs':>5}{'steps':>13}{'final':>12}{'best':>9}")
    for r in summary:
        print(f"{r['env']:<18}{r['method']:<14}{r['n_runs']:>5}{r['steps']:>13,.0f}"
              f"{r['final']:>8.1f} ±{r['std']:<4.0f}{r['best']:>9.1f}")
    if len(labels) == 2:
        a, b = labels
        print(f"\n{'env':<18}{'final ' + b + ' - ' + a:>28}")
        for env in envs:
            fa = next((r["final"] for r in summary if r["env"] == env and r["method"] == a), None)
            fb = next((r["final"] for r in summary if r["env"] == env and r["method"] == b), None)
            if fa is not None and fb is not None:
                print(f"{env:<18}{fb - fa:>28.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
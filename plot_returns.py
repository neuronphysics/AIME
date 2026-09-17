from __future__ import annotations

import argparse
import csv
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

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#B22222", "#000000"]

METRIC_LABELS = {
    "eval_return": "Evaluation return",
    "train_return": "Training return",
}

# run-directory suffix -> method label. SHS is the proposed model (solid),
# vanilla DreamerV3 is the baseline (dashed). Unknown -> single-method mode.
METHOD_TAGS = [("_shs", "SHS-RSSM"), ("_vanilla", "DreamerV3"), ("_dreamer", "DreamerV3")]
METHOD_STYLE = {"SHS-RSSM": ("#0072B2", "-"), "DreamerV3": ("#D55E00", "--")}


def infer_run(path: str, logdir: str) -> tuple[str, str]:
    """(env/grid label, method label) from the run directory name.
    carl_walker_gravity_friction_shs/seed_2/metrics.jsonl -> ('walker_gravity_friction', 'SHS-RSSM')
    pendulum_swingup_dreamer/seed_1/metrics.jsonl         -> ('pendulum_swingup', 'DreamerV3')
    walker_walk/seed_1/metrics.jsonl                       -> ('walker_walk', '')"""
    rel = os.path.relpath(path, logdir if os.path.isdir(logdir) else os.path.dirname(logdir))
    parts = [p for p in rel.replace("\\", "/").split("/") if p][:-1]  # drop filename
    comp = next((p for p in parts if any(e in p.lower() for e in KNOWN_ENVS)), None)
    if comp is None:
        comp = next((p for p in reversed(parts) if not SEED_RE.fullmatch(p) and p not in (".", "logs", "logdir")), parts[-1] if parts else "run")
    name, method = comp.lower(), ""
    visible = name.endswith("_visible")
    if visible:
        name = name[: -len("_visible")]
    for tag, label in METHOD_TAGS:
        if name.endswith(tag):
            method, name = label, name[: -len(tag)]
            break
    if visible:
        method = (method or "run") + " (ctx visible)"
    name = re.sub(r"^(carl_|dmc_)", "", name)
    return name, method


def method_style(method: str, i: int) -> tuple[str, str]:
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    base = method.split(" (")[0]
    if base in METHOD_STYLE:  # "(ctx visible)" variants: same hue, dotted
        return METHOD_STYLE[base][0], ":"
    return OKABE_ITO[(i + 2) % len(OKABE_ITO)], "-"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.15",
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "grid.color": "0.87",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
    })


def style_axis(ax) -> None:
    for s in ax.spines.values():           # full frame on all four sides
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


def find_metric_files(logdir: str, filename: str = "metrics.jsonl") -> list[str]:
    hits = []
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if f == filename or f.endswith("_" + filename):
                hits.append(os.path.join(root, f))
    if os.path.isfile(logdir):
        hits.append(logdir)
    return sorted(set(hits))


def infer_env(path: str, logdir: str) -> str:
    rel = os.path.relpath(path, logdir if os.path.isdir(logdir) else os.path.dirname(logdir))
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    hay = "_".join(parts).lower()
    for env in KNOWN_ENVS:
        if env in hay:
            m = re.search(rf"{env}[_-]([a-z]+)", hay)
            return f"{env}_{m.group(1)}" if m and m.group(1) not in ("metrics", "seed") else env
    for p in reversed(parts[:-1]):
        if p not in (".", "logs", "logdir"):
            return p
    return os.path.splitext(parts[-1])[0].replace("_metrics", "")


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
        print(f"    ! {bad} unparseable/!numeric lines skipped in {os.path.basename(path)}",
              file=sys.stderr)
    o = np.argsort(steps, kind="stable")
    s, v = np.asarray(steps)[o], np.asarray(vals)[o]
    if len(s):
        keep = np.concatenate([s[1:] != s[:-1], [True]])   # resume-safe: last wins
        s, v = s[keep], v[keep]
    return s, v


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def slope_per_1m(steps: np.ndarray, vals: np.ndarray, frac: float = 0.25) -> float:
    n = len(vals)
    if n < 4:
        return float("nan")
    k = max(2, int(n * frac))
    return float(np.polyfit(steps[-k:], vals[-k:], 1)[0] * 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True, help="root to search (or a single file)")
    ap.add_argument("--filename", default="metrics.jsonl")
    ap.add_argument("--metric", default="eval_return",
                    help="eval_return (default) | train_return | any logged key")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="substring filter, e.g. --envs cheetah hopper")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="keep only these methods, e.g. --methods SHS-RSSM DreamerV3")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window")
    ap.add_argument("--ref", default=None,
                    help='JSON of reference scores, e.g. \'{"cheetah": 880}\'')
    ap.add_argument("--separate", action="store_true", help="one panel per environment")
    ap.add_argument("--no-aggregate", action="store_true", help="draw every seed")
    ap.add_argument("--max-steps", type=float, default=None)
    ap.add_argument("--out", default="returns.png")
    ap.add_argument("--csv", default=None, help="also write a summary CSV here")
    args = ap.parse_args()

    files = find_metric_files(args.logdir, args.filename)
    if not files:
        print(f"No '{args.filename}' found under {args.logdir}", file=sys.stderr)
        return 1

    runs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        env, method = infer_run(f, args.logdir)
        if args.envs and not any(e.lower() in env.lower() or e.lower() in f.lower()
                                 for e in args.envs):
            continue
        if args.methods and method not in args.methods:
            continue
        s, v = load_series(f, args.metric)
        if len(s) == 0:
            print(f"    ! no '{args.metric}' in {f}", file=sys.stderr)
            continue
        if args.max_steps:
            m = s <= args.max_steps
            s, v = s[m], v[m]
        runs[env][method].append((infer_seed(f), s, v, f))
        print(f"  {env:<26} {method or '-':<12} seed={infer_seed(f) or '-':<4} n={len(s):<6} "
              f"steps<= {s[-1]:>12,.0f}  {os.path.relpath(f, args.logdir)}")

    if not runs:
        print("Nothing matched the filters.", file=sys.stderr)
        return 1

    set_style()
    ref = json.loads(args.ref) if args.ref else {}
    envs = sorted(runs)
    colors = {e: OKABE_ITO[i % len(OKABE_ITO)] for i, e in enumerate(envs)}
    ylab = metric_label(args.metric)

    if args.separate:
        n = len(envs)
        ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.8 * nrow),
                                 squeeze=False, constrained_layout=True)
        axes = axes.ravel()
    else:
        fig, ax0 = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        axes = [ax0] * len(envs)

    summary = []
    for i, env in enumerate(envs):
      ax = axes[i]
      for mi, (method, series) in enumerate(sorted(runs[env].items(), key=lambda kv: kv[0] != "SHS-RSSM")):
        # separate panels: colour = method; combined panel: colour = env, linestyle = method
        mc, ls = method_style(method, mi)
        c = mc if (args.separate or len(envs) == 1) else colors[env]
        if not method:
            c, ls = colors[env], "-"
        label = pretty_env(env) if not method else (method if args.separate else f"{pretty_env(env)} — {method}")
        if len(series) > 1 and not args.no_aggregate:
            grid = np.unique(np.concatenate([s for _, s, _, _ in series]))
            lo = max(s[0] for _, s, _, _ in series)
            hi = min(s[-1] for _, s, _, _ in series)
            grid = grid[(grid >= lo) & (grid <= hi)]
            M = np.vstack([np.interp(grid, s, v, left=np.nan, right=np.nan)
                           for _, s, v, _ in series])
            med = np.nanmedian(M, 0)
            q1, q3 = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
            mp, q1p, q3p = (smooth(a, args.smooth) for a in (med, q1, q3))
            gp = grid[len(grid) - len(mp):]
            ax.fill_between(gp / 1e6, q1p, q3p, color=c, alpha=.16, lw=0, zorder=2)
            ax.plot(gp / 1e6, mp, color=c, ls=ls, lw=2.0, zorder=3,
                    label=f"{label}, $n={len(series)}$")
            s_ref, v_ref = grid, med
        else:
            for sd, s, v, _ in series:
                y = smooth(v, args.smooth); x = s[len(s) - len(y):]
                lab = label if sd is None else f"{label} s{sd}"
                ax.plot(x / 1e6, y, color=c, ls=ls, lw=1.5, alpha=.85, zorder=3, label=lab)
            s_ref, v_ref = series[0][1], series[0][2]

        if env in ref or any(k in env for k in ref):
            key = env if env in ref else next(k for k in ref if k in env)
            ax.axhline(ref[key], color="0.25", ls=(0, (5, 3)), lw=1.1, zorder=1)
            ax.text(.995, ref[key], f"ref {ref[key]:g} ",
                    transform=ax.get_yaxis_transform(),
                    va="top", ha="right", fontsize=7.5, color="0.25")

        sl = slope_per_1m(s_ref, v_ref)
        k = max(2, int(len(v_ref) * .25))
        summary.append(dict(env=env, method=method or "-", n_runs=len(series), steps=float(s_ref[-1]),
                            final=float(np.nanmean(v_ref[-k:])),
                            std=float(np.nanstd(v_ref[-k:])), best=float(np.nanmax(v_ref)),
                            slope_per_1M=sl,
                            status="converged" if abs(sl) < 100 else "still changing"))
      if args.separate:
        style_axis(ax)
        ax.set_title(pretty_env(env), loc="left", fontweight="bold")
        finals = [r for r in summary if r["env"] == env]
        ax.text(0.98, 0.04, "\n".join(f"{r['method']}: {r['final']:.0f}$\\pm${r['std']:.0f}" for r in finals),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.35")
        ax.set_xlabel(r"Environment steps ($\times 10^6$)")
        ax.set_ylabel(ylab)
        if len(runs[env]) > 1 or any(runs[env]):
            ax.legend(borderaxespad=0.6)

    if not args.separate:
        style_axis(ax0)
        ax0.set_xlabel(r"Environment steps ($\times 10^6$)")
        ax0.set_ylabel(ylab)
        ax0.set_title(ylab + ("  (median $\\pm$ IQR over seeds)"
                              if not args.no_aggregate else ""),
                      loc="left", fontweight="bold")
        ax0.legend(ncols=1 if len(envs) <= 4 else 2, borderaxespad=0.6)
        ax0.margins(x=0.01)
    else:
        for j in range(len(envs), len(axes)):
            axes[j].axis("off")

    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    print(f"\n{'env':<26}{'method':<12}{'runs':>5}{'steps':>13}{'final':>10}{'best':>9}"
          f"{'slope/1M':>11}  status")
    for r in summary:
        print(f"{r['env']:<26}{r['method']:<12}{r['n_runs']:>5}{r['steps']:>13,.0f}{r['final']:>10.1f}"
              f"{r['best']:>9.1f}{r['slope_per_1M']:>11.0f}  {r['status']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

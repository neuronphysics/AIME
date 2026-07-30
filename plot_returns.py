#!/usr/bin/env python3
"""plot_returns.py — find metrics.jsonl under a logdir and plot return vs env steps.

Discovers runs by walking a logdir tree, infers the environment (and seed) from the
directory name, groups repeats, and writes a figure plus a summary CSV.

Typical Dreamer layouts all work:

    logdir/dmc_cheetah_run/metrics.jsonl
    logdir/dmc_hopper_hop/metrics.jsonl
    logdir/dmc_humanoid_walk/metrics.jsonl

Examples
--------
    python plot_returns.py --logdir ~/logdir
    python plot_returns.py --logdir ~/logdir --envs cheetah hopper humanoid
    python plot_returns.py --logdir ~/logdir --metric train_return --smooth 25
    python plot_returns.py --logdir ~/logdir --ref '{"cheetah": 880, "hopper": 370}'
    python plot_returns.py --logdir ~/logdir --separate --out returns.png

Notes
-----
* `--metric eval_return` (default) is what you should report. `train_return` is logged
  far more often and is noisier because it includes exploration.
* Multiple seeds for the same env are aggregated as median with an IQR band. Pass
  `--no-aggregate` to draw every seed separately.
* `--ref` draws horizontal reference lines (e.g. published DreamerV3 scores). Supply the
  score for the SAME task variant and observation modality as your run: DMC Vision and
  DMC Proprio differ substantially, and e.g. hopper_hop vs hopper_stand differ ~3x.
"""
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
import numpy as np

KNOWN_ENVS = [
    "acrobot", "cartpole", "cheetah", "cup", "finger", "hopper", "humanoid",
    "pendulum", "quadruped", "reacher", "walker", "dog", "fish", "manipulator",
    "swimmer", "point_mass", "ball_in_cup",
]
SEED_RE = re.compile(r"(?:seed|s)[_-]?(\d+)", re.I)


# discovery
def find_metric_files(logdir: str, filename: str = "metrics.jsonl") -> list[str]:
    """Every `filename` under `logdir`, plus flat `*_<filename>` siblings."""
    hits = []
    for root, _dirs, files in os.walk(logdir):
        for f in files:
            if f == filename or f.endswith("_" + filename):
                hits.append(os.path.join(root, f))
    if os.path.isfile(logdir):
        hits.append(logdir)
    return sorted(set(hits))


def infer_env(path: str, logdir: str) -> str:
    """Best-effort environment name from the path (basename first, then parents)."""
    rel = os.path.relpath(path, logdir if os.path.isdir(logdir) else os.path.dirname(logdir))
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    hay = "_".join(parts).lower()
    for env in KNOWN_ENVS:
        if env in hay:
            m = re.search(rf"{env}[_-]([a-z]+)", hay)
            return f"{env}_{m.group(1)}" if m and m.group(1) not in ("metrics", "seed") else env
    for p in reversed(parts[:-1]):          # fall back to nearest meaningful dir
        if p not in (".", "logs", "logdir"):
            return p
    return os.path.splitext(parts[-1])[0].replace("_metrics", "")


def infer_seed(path: str) -> str | None:
    m = SEED_RE.search(path)
    return m.group(1) if m else None


# loading
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
    o = np.argsort(steps)
    return np.asarray(steps)[o], np.asarray(vals)[o]


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def slope_per_1m(steps: np.ndarray, vals: np.ndarray, frac: float = 0.25) -> float:
    """Least-squares slope over the final `frac` of the run, in return per 1M steps."""
    n = len(vals)
    if n < 4:
        return float("nan")
    k = max(2, int(n * frac))
    return float(np.polyfit(steps[-k:], vals[-k:], 1)[0] * 1e6)


# main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True, help="root to search (or a single file)")
    ap.add_argument("--filename", default="metrics.jsonl")
    ap.add_argument("--metric", default="eval_return",
                    help="eval_return (default) | train_return | any logged key")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="substring filter, e.g. --envs cheetah hopper")
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

    runs: dict[str, list] = defaultdict(list)
    for f in files:
        env = infer_env(f, args.logdir)
        if args.envs and not any(e.lower() in env.lower() or e.lower() in f.lower()
                                 for e in args.envs):
            continue
        s, v = load_series(f, args.metric)
        if len(s) == 0:
            print(f"    ! no '{args.metric}' in {f}", file=sys.stderr)
            continue
        if args.max_steps:
            m = s <= args.max_steps
            s, v = s[m], v[m]
        runs[env].append((infer_seed(f), s, v, f))
        print(f"  {env:<18} seed={infer_seed(f) or '-':<4} n={len(s):<6} "
              f"steps<= {s[-1]:>12,.0f}  {os.path.relpath(f, args.logdir)}")

    if not runs:
        print("Nothing matched the filters.", file=sys.stderr)
        return 1



    ref = json.loads(args.ref) if args.ref else {}
    envs = sorted(runs)
    cmap = plt.get_cmap("tab10")
    colors = {e: cmap(i % 10) for i, e in enumerate(envs)}

    if args.separate:
        n = len(envs)
        ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
        axes = axes.ravel()
    else:
        fig, ax0 = plt.subplots(figsize=(9.2, 5.4))
        axes = [ax0] * len(envs)

    summary = []
    for i, env in enumerate(envs):
        ax = axes[i]; c = colors[env]; series = runs[env]
        if len(series) > 1 and not args.no_aggregate:
            grid = np.unique(np.concatenate([s for _, s, _, _ in series]))
            M = np.vstack([np.interp(grid, s, v, left=np.nan, right=np.nan)
                           for _, s, v, _ in series])
            med = np.nanmedian(M, 0)
            q1, q3 = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
            ax.fill_between(grid / 1e3, q1, q3, color=c, alpha=.20, lw=0)
            ax.plot(grid / 1e3, med, color=c, lw=2, label=f"{env} (n={len(series)})")
            s_ref, v_ref = grid, med
        else:
            for sd, s, v, _ in series:
                y = smooth(v, args.smooth); x = s[len(s) - len(y):]
                lab = env if sd is None else f"{env} s{sd}"
                ax.plot(x / 1e3, y, color=c, lw=1.8, alpha=.9, label=lab)
            s_ref, v_ref = series[0][1], series[0][2]

        if env in ref or any(k in env for k in ref):
            key = env if env in ref else next(k for k in ref if k in env)
            ax.axhline(ref[key], color=c, ls="--", lw=1.4, alpha=.8)
            ax.text(.01, ref[key], f"  ref {ref[key]:g}", transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=7.5, color=c)

        sl = slope_per_1m(s_ref, v_ref)
        k = max(2, int(len(v_ref) * .25))
        summary.append(dict(env=env, n_runs=len(series), steps=float(s_ref[-1]),
                            final=float(np.nanmean(v_ref[-k:])),
                            std=float(np.nanstd(v_ref[-k:])), best=float(np.nanmax(v_ref)),
                            slope_per_1M=sl,
                            status="converged" if abs(sl) < 100 else "still changing"))
        if args.separate:
            ax.set_title(f"{env}   final {summary[-1]['final']:.0f}   slope {sl:+.0f}/1M",
                         loc="left", fontweight="bold", fontsize=10)
            ax.set_xlabel("env steps (k)"); ax.set_ylabel(args.metric)
            ax.spines[["top", "right"]].set_visible(False)

    if not args.separate:
        ax0.set_xlabel("env steps (k)"); ax0.set_ylabel(args.metric)
        ax0.legend(frameon=False, fontsize=9)
        ax0.spines[["top", "right"]].set_visible(False)
        ax0.set_title(f"{args.metric} vs env steps", loc="left", fontweight="bold")
    else:
        for j in range(len(envs), len(axes)):
            axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    print(f"\n{'env':<18}{'runs':>5}{'steps':>13}{'final':>10}{'best':>9}"
          f"{'slope/1M':>11}  status")
    for r in summary:
        print(f"{r['env']:<18}{r['n_runs']:>5}{r['steps']:>13,.0f}{r['final']:>10.1f}"
              f"{r['best']:>9.1f}{r['slope_per_1M']:>11.0f}  {r['status']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

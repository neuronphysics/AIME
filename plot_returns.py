#!/usr/bin/env python3
"""Learning curves of SHS-RSSM against its baselines.

  python3 plot_returns.py door-open --input vision --csv metaworld_curves.csv
  python3 plot_returns.py --suite carl|procgen|atari|dmc --logdir ./logdir --format png
  python3 plot_returns.py walker_walk --suite dmc --arm "SHS-RSSM, old=dmc_{task}_shs_seed_*"
"""

import argparse
import csv
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ARMS = {
    "state":  {"DreamerV3": "baseline",        "SHS-RSSM": "shs_latent"},
    "vision": {"DreamerV3": "baseline_vision", "SHS-RSSM": "shs_latent_vision"},
}
COLORS = {"SHS-RSSM": "#d62728", "DreamerV3": "#1f77b4",
          "R2I": "#2ca02c", "PPO": "#9467bd"}
ORDER = ["SHS-RSSM", "DreamerV3", "R2I", "PPO"]

REFERENCE = {
    "door-open": (268, 4492),
    "reach": (716, 4841),
    "drawer-open": (613, 4057),
    "button-press-topdown": (193, 3652),
}

# Run folders written by the launchers, relative to --logdir, one pattern per method.
# Colour marks the forgetting rule; dashed = structure moves, dotted = Gaussian-latent DreamerV3.
SHS = r"_shs(?P<moves>_moves)?_(?P<forget>gated|retain|fixed)"
SUITES = {
    "carl": dict(at=1e6, rules=[
        rf"carl_(?P<task>.+?){SHS}/seed_?\d+",
        r"carl_(?P<task>.+?)_(?P<dreamer>vanilla)/seed_?\d+"]),
    "procgen": dict(at=1e6, rules=[
        rf"procgen_(?P<task>[a-z]+){SHS}/seed_?\d+",
        r"procgen_(?P<task>[a-z]+)_(?P<dreamer>vanilla)/seed_?\d+"]),
    "atari": dict(at=4e5, rules=[
        r"atari_(?P<forget>gated|retain|fixed)(?P<fixedk>_fixedk)?/(?P<task>[a-z0-9_]+)/seed_?\d+",
        r"atari_(?P<dreamer>vanilla)/(?P<task>[a-z0-9_]+)/seed_?\d+"]),
    "dmc": dict(at=1e6, rules=[
        r"dmc_(?P<task>[a-z0-9_]+?)_shs_(?P<forget>gated|retain|fixed)(?P<fixedk>_fixedk)?_seed_\d+",
        r"dreamer_(?P<dreamer>cat|gauss)/dmc_(?P<task>[a-z0-9_]+)/seed_?\d+"]),
}
FORGET_COLORS = {"gated": "#d62728", "retain": "#ff7f0e", "fixed": "#17becf"}
DREAMER_COLOR = "#1f77b4"
EXTRA_COLORS = ["#e377c2", "#8c564b", "#7f7f7f", "#bcbd22"]


def canon(task):
    t = task.strip()
    return t[:-3] if t.endswith("-v3") else t


def read_seeds(paths, metric):
    out = []
    for p in sorted(paths):
        latest = {}
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric in r:
                latest[r["step"]] = r[metric]
        if len(latest) > 1:
            s = np.array(sorted(latest), float)
            out.append((s, np.array([latest[k] for k in s], float)))
    return out


def from_logdir(logdir, arm, task, metric):
    return read_seeds(glob.glob(os.path.join(logdir, arm, task, "seed*", "metrics.jsonl")), metric)


def from_csv(path, algo, inp, task, metric):
    per_seed = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["algorithm"] != algo or r["input"] != inp:
                continue
            if canon(r["task"]) != task:
                continue
            try:
                per_seed[r["seed"]][float(r["step"])] = float(r[metric])
            except (TypeError, ValueError):
                continue
    out = []
    for d in per_seed.values():
        if len(d) > 1:
            s = np.array(sorted(d), float)
            out.append((s, np.array([d[k] for k in s], float)))
    return out


def band(curves, at, points=120):
    curves = [(s, v) for s, v in curves if s[0] <= at]
    if not curves:
        return None
    # every seed is cut at the shortest one, so the mean is over the same seeds everywhere
    cut = []
    for s, v in curves:
        m = s <= at
        cut.append((s[m], v[m]))
    hi_end = min(s[-1] for s, _ in cut)
    lo_end = max(s[0] for s, _ in cut)
    if hi_end <= lo_end:
        return None
    grid = np.linspace(lo_end, hi_end, points)
    stack = np.stack([np.interp(grid, s, v) for s, v in cut])
    return grid, stack.mean(0), stack.min(0), stack.max(0), len(cut)


def draw(ax, curves, at, color, label, ls="-"):
    b = band(curves, at)
    if b is None:
        return None
    grid, mean, lo, hi, n = b
    line, = ax.plot(grid, mean, lw=1.9, color=color, ls=ls, label=label, zorder=3)
    ax.fill_between(grid, lo, hi, alpha=0.15, color=color, linewidth=0, zorder=2)
    return line, n, grid[-1]


def style(ax, ylabel, success=False, xlabel="environment steps"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if success:
        ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25, lw=0.5, zorder=0)
    ax.margins(x=0.01)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.get_offset_text().set_fontsize(8)


def legend_below(owner, handles, anchor, handlelength=1.6, ncol=None):
    ncol = ncol or (2 if len(handles) > 3 else len(handles))
    owner.legend(handles=handles, fontsize=7.5, frameon=False, ncol=ncol,
                 loc="upper center", bbox_to_anchor=anchor,
                 handlelength=handlelength, columnspacing=1.4, borderaxespad=0.0)


def save(fig, args, stem):
    name = os.path.join(args.out, f"{stem}.{args.format}")
    fig.savefig(name, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print("wrote", name)


def plot_metaworld(args):
    task = canon(args.task)
    at = 5e5 if args.at is None else args.at
    metric = args.metric or "eval_return"
    series = {}
    for name, arm in ARMS[args.input].items():
        series[name] = from_logdir(args.logdir, arm, task, metric)
    if metric == "eval_success":
        for name, arm in ARMS[args.input].items():
            if not series[name]:
                series[name] = from_logdir(args.logdir, arm, task, "eval_log_success")
    for algo in ("R2I", "PPO"):
        series[algo] = from_csv(args.csv, algo, args.input, task, metric)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    handles = []
    for name in ORDER:
        drawn = draw(ax, series.get(name, []), at, COLORS[name], name)
        if drawn is None:
            print(f"  no data: {name}")
            continue
        handles.append(drawn[0])

    if not args.no_refs and metric == "eval_return" and task in REFERENCE:
        rnd, exp = REFERENCE[task]
        h1 = ax.axhline(exp, ls="--", lw=1.0, color="0.35", zorder=1)
        h2 = ax.axhline(rnd, ls=":", lw=1.0, color="0.55", zorder=1)
        h1.set_label(f"scripted expert ({exp})")
        h2.set_label(f"random policy ({rnd})")
        handles += [h1, h2]

    style(ax, "success rate" if metric == "eval_success" else "evaluation return",
          success=metric == "eval_success")
    legend_below(ax, handles, (0.5, -0.22))
    fig.tight_layout()
    save(fig, args, f"fig_{task}_{args.input}_{metric}")


def task_key(suite, task):
    t = task.strip()
    for prefix in (f"{suite}_", "dmc_"):
        t = t[len(prefix):] if t.startswith(prefix) else t
    return t[:-len("_shs")] if t.endswith("_shs") else t


def run_dirs(logdir, depth=3):
    skip = {"train_eps", "eval_eps", "shs_diagnostics"}
    for d, subdirs, files in os.walk(logdir):
        rel = os.path.relpath(d, logdir).replace(os.sep, "/")
        level = 0 if rel == "." else rel.count("/") + 1
        subdirs[:] = [] if level >= depth else [s for s in subdirs if s not in skip and not s.startswith(".")]
        if "metrics.jsonl" in files and level:
            yield d, rel


def discover(logdir, suite):
    # task -> label -> arm; an arm collects the seed folders of one method
    found = defaultdict(dict)
    rules = [re.compile(r) for r in SUITES[suite]["rules"]]
    for d, rel in run_dirs(logdir):
        m = next((m for r in rules if (m := r.fullmatch(rel))), None)
        if m is None:
            continue
        g = m.groupdict()
        if g.get("dreamer"):
            gauss = g["dreamer"] == "gauss"
            arm = dict(label="DreamerV3, Gaussian latents" if gauss else "DreamerV3",
                       color=DREAMER_COLOR, ls=":" if gauss else "-", rank=(3, gauss))
        else:
            moves = bool(g.get("moves")) or suite == "procgen" or (suite in ("atari", "dmc") and not g.get("fixedk"))
            arm = dict(label=f"SHS-RSSM, {g['forget']}" + (" + moves" if moves else ""),
                       color=FORGET_COLORS[g["forget"]], ls="--" if moves else "-",
                       rank=(list(FORGET_COLORS).index(g["forget"]), moves))
        found[g["task"]].setdefault(arm["label"], dict(arm, dirs=[]))["dirs"].append(d)
    return found


def extra_arms(logdir, specs, task):
    arms = {}
    for i, spec in enumerate(specs):
        label, _, pattern = spec.partition("=")
        dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(logdir, pattern.format(task=task), "metrics.jsonl")))
        if dirs:
            arms[label] = dict(label=label, color=EXTRA_COLORS[i % len(EXTRA_COLORS)], ls="-", rank=(4, i), dirs=dirs)
    return arms


def eval_contexts(arms):
    # CARL eval context i is the i-th entry of the product of the eval_contexts lists, as in envs/carl.py
    for arm in arms:
        for d in arm["dirs"]:
            path = os.path.join(d, "run_config.json")
            if not os.path.exists(path):
                continue
            spec = json.load(open(path)).get("carl", {}).get("eval_contexts") or {}
            if spec:
                keys = list(spec)
                return [", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in zip(keys, values))
                        for values in itertools.product(*(spec[k] for k in keys))]
    return []


def panel(ax, arms, metric, at, seeds, task=None):
    for arm in arms:
        drawn = draw(ax, read_seeds(arm["metrics"], metric), at, arm["color"], arm["label"], ls=arm["ls"])
        if drawn is None:
            continue
        line, n, end = drawn
        seeds.setdefault(arm["label"], [line, [], arm["rank"]])[1].append(n)
        if task:
            print(f"  {task}: {arm['label']}: {n} seeds up to {end:.0f} steps")


def seed_legend(seeds):
    # the seed count goes in the legend; a range means it differs between tasks or contexts
    handles = []
    for label, (line, ns, _) in sorted(seeds.items(), key=lambda kv: kv[1][2]):
        n = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}"
        line.set_label(f"{label} (n={n})")
        handles.append(line)
    return handles


def grid(n):
    cols = 2 if n == 4 else min(n, 3)
    return -(-n // cols), cols


def label_outer(axes, used, ylabel):
    rows, cols = axes.shape
    for i, ax in enumerate(axes.flat):
        if i >= used:
            ax.set_visible(False)
            continue
        below = i + cols < used
        style(ax, ylabel if i % cols == 0 else "", xlabel="" if below else "environment steps")


def plot_suite(args):
    suite, metric = args.suite, args.metric or "eval_return"
    at = SUITES[suite]["at"] if args.at is None else args.at
    found = discover(args.logdir, suite)
    tasks = sorted(found) if args.task == "all" else [task_key(suite, args.task)]
    table = {}
    for task in tasks:
        arms = dict(found.get(task, {}))
        arms.update(extra_arms(args.logdir, args.arm, task))
        if arms:
            arms = sorted(arms.values(), key=lambda a: a["rank"])
            for arm in arms:
                arm["metrics"] = [os.path.join(d, "metrics.jsonl") for d in arm["dirs"]]
            table[task] = arms
        else:
            print(f"  no {suite} runs for {task}")
    if not table:
        print(f"no {suite} runs under {args.logdir}")
        return
    ylabel = "evaluation return" if metric == "eval_return" else metric.replace("_", " ")
    stem = f"fig_{suite}_{'all' if args.task == 'all' else tasks[0]}_{metric}"

    # one panel per task, so every task of the suite is in the same figure
    rows, cols = grid(len(table))
    single = len(table) == 1
    size = (4.6, 3.4) if single else (3.4 * cols, 2.8 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    seeds = {}
    for ax, (task, arms) in zip(axes.flat, table.items()):
        panel(ax, arms, metric, at, seeds, task)
        if not single:
            ax.set_title(task.replace("_", " "), fontsize=9)
    label_outer(axes, len(table), ylabel)
    fig.tight_layout()
    handles = seed_legend(seeds)
    if single:
        legend_below(axes[0, 0], handles, (0.5, -0.22), handlelength=2.2)
    else:
        legend_below(fig, handles, (0.5, 0.0), handlelength=2.2, ncol=min(len(handles), max(2, cols)))
    save(fig, args, stem)

    # CARL: one panel per (task, held-out context); the contexts of a task share their y range
    if suite != "carl" or metric != "eval_return":
        return
    ctx = {task: labels for task, arms in table.items() if len(labels := eval_contexts(arms)) >= 2}
    panels = [(task, c, label) for task, labels in ctx.items() for c, label in enumerate(labels)]
    if not panels:
        return
    cols = min(4, len(panels))
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.8 * rows), squeeze=False)
    seeds, by_task = {}, defaultdict(list)
    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.set_visible(False)
            continue
        task, c, label = panels[i]
        panel(ax, table[task], f"eval_return_ctx{c}", at, seeds)
        style(ax, ylabel if i % cols == 0 else "",
              xlabel="" if i + cols < len(panels) else "environment steps")
        ax.set_title(f"{task.replace('_', ' ')}\n{label}", fontsize=8)
        by_task[task].append(ax)
    for task_axes in by_task.values():
        lo = min(ax.get_ylim()[0] for ax in task_axes)
        hi = max(ax.get_ylim()[1] for ax in task_axes)
        for ax in task_axes:
            ax.set_ylim(lo, hi)
    fig.tight_layout()
    handles = seed_legend(seeds)
    legend_below(fig, handles, (0.5, 0.0), handlelength=2.2, ncol=min(len(handles), max(2, cols)))
    save(fig, args, f"{stem}_contexts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all",
                    help="task name (door-open, walker, coinrun, pong, walker_walk, ...) or 'all' (default)")
    ap.add_argument("--suite", default="metaworld", choices=["metaworld", *SUITES])
    ap.add_argument("--input", default="vision", choices=["state", "vision"])
    ap.add_argument("--csv", default="metaworld_curves.csv")
    ap.add_argument("--logdir", default=None,
                    help="default ./logdir/latent_gate_v1 (Meta-World) or ./logdir")
    ap.add_argument("--metric", default=None,
                    choices=["eval_return", "eval_success", "train_return"])
    ap.add_argument("--at", type=float, default=None,
                    help="truncate every curve here (default 5e5 Meta-World and 4e5 Atari, else 1e6)")
    ap.add_argument("--arm", action="append", default=[],
                    help="extra method as 'LABEL=GLOB', GLOB relative to --logdir, {task} is filled in")
    ap.add_argument("--out", default=None, help="output folder (default: --logdir)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=200, help="resolution of png figures")
    ap.add_argument("--no-refs", action="store_true",
                    help="omit the expert / random reference lines")
    args = ap.parse_args()
    args.logdir = args.logdir or ("./logdir/latent_gate_v1" if args.suite == "metaworld" else "./logdir")
    args.out = args.out or args.logdir
    os.makedirs(args.out, exist_ok=True)
    if args.suite == "metaworld":
        if args.task == "all":
            ap.error("Meta-World needs a task name, e.g. door-open")
        plot_metaworld(args)
    else:
        plot_suite(args)


if __name__ == "__main__":
    main()
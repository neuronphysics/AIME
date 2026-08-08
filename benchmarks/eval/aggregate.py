"""Aggregate metrics.jsonl across tasks and seeds into a comparable score.

The point of this script is to make "did we beat DreamerV3" a question with a
number attached instead of a vibe from eyeballing TensorBoard. It computes the
Agarwal et al. (2021) aggregates -- median, IQM, mean, optimality gap -- with
stratified bootstrap CIs over the (task, seed) matrix, which is the standard
these benchmarks are reported under.

Expected layout:

    logdir/
      <arm>/<task>/seed<k>/metrics.jsonl

Usage:

    python benchmarks/eval/aggregate.py ./logdir \\
        --metric eval_log_success --arms shs baseline \\
        --at 1000000

`--metric eval_log_success` gives Meta-World success rate averaged over the
eval episodes; `--metric eval_return`
with `--normalize dmc` gives DMC normalised return.

Caveat worth stating in any writeup: a bootstrap CI over 3 seeds is wide and
mostly reflects seed noise. If the SHS and baseline CIs overlap, the honest
sentence is "comparable", not "beats".
"""

import argparse
import json
import pathlib
import re

import numpy as np


def read_curve(path, metric):
    """Return (steps, values) arrays for one run's metrics.jsonl.

    metrics.jsonl is APPENDED to, not truncated, so a logdir that was resumed
    after a timeout -- or rerun after a crash -- contains several runs' rows
    interleaved, with repeated step values. Left alone, a stale row from an
    abandoned run can win the `last N points` selection and silently corrupt a
    score. Later rows are from the more recent run, so keep the last value seen
    for each step.
    """
    latest = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric in rec:
                latest[rec["step"]] = rec[metric]
    steps = np.array(sorted(latest))
    values = np.array([latest[s] for s in steps], np.float64)
    return steps, values


def value_at(steps, values, at, window=5):
    """Mean of the last `window` logged points at or before `at`."""
    if len(steps) == 0:
        return np.nan
    mask = steps <= at if at is not None else np.ones_like(steps, bool)
    if not mask.any():
        return np.nan
    return float(np.mean(values[mask][-window:]))


def collect(logdir, arm, metric, at, window=5):
    """-> dict[task] -> list of per-seed scalars."""
    root = pathlib.Path(logdir) / arm
    out = {}
    for jsonl in sorted(root.glob("*/*/metrics.jsonl")):
        task = jsonl.parent.parent.name
        steps, values = read_curve(jsonl, metric)
        score = value_at(steps, values, at, window)
        if not np.isnan(score):
            out.setdefault(task, []).append(score)
    return out


def to_matrix(per_task):
    """dict[task]->seeds  =>  (n_seeds, n_tasks) matrix, truncated to min seeds."""
    tasks = sorted(per_task)
    if not tasks:
        return np.zeros((0, 0)), tasks
    n = min(len(per_task[t]) for t in tasks)
    return np.array([[per_task[t][s] for t in tasks] for s in range(n)]), tasks


def iqm(x):
    flat = np.sort(x.reshape(-1))
    lo, hi = int(0.25 * len(flat)), int(np.ceil(0.75 * len(flat)))
    return float(np.mean(flat[lo:hi])) if hi > lo else float(np.mean(flat))


AGGREGATES = {
    "median": lambda x: float(np.median(np.mean(x, axis=0))),
    "iqm": iqm,
    "mean": lambda x: float(np.mean(x)),
}


def stratified_bootstrap(matrix, fn, reps=2000, seed=0):
    """Resample seeds within each task independently (Agarwal et al. 2021)."""
    rng = np.random.default_rng(seed)
    n_seeds, n_tasks = matrix.shape
    if n_seeds == 0:
        return np.nan, np.nan
    draws = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n_seeds, size=(n_seeds, n_tasks))
        draws[r] = fn(np.take_along_axis(matrix, idx, axis=0))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--metric", default="eval_return")
    ap.add_argument("--at", type=float, default=None,
                    help="env-step budget to score at (default: end of run)")
    ap.add_argument("--window", type=int, default=5,
                    help="average the last N logged points at/below --at "
                         "(smoothing; 5 x eval_every=1e4 -> last 50k steps)")
    ap.add_argument("--divide-by", type=float, default=1.0,
                    help="normaliser, e.g. 1000 for DMC return -> [0,1]")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    results = {}
    for arm in args.arms:
        per_task = collect(args.logdir, arm, args.metric, args.at, args.window)
        matrix, tasks = to_matrix(per_task)
        matrix = matrix / args.divide_by
        results[arm] = (matrix, tasks, per_task)
        print(f"\n=== {arm} ===")
        if matrix.size == 0:
            print("  no runs found -- check the logdir layout")
            continue
        n_seeds, n_tasks = matrix.shape
        print(f"  {n_tasks} tasks x {n_seeds} seeds "
              f"(seeds truncated to the min across tasks)")
        dropped = {t: len(v) for t, v in per_task.items() if len(v) > n_seeds}
        if dropped:
            print(f"  NOTE: dropped extra seeds from {dropped}")
        for name, fn in AGGREGATES.items():
            point = fn(matrix)
            lo, hi = stratified_bootstrap(matrix, fn, args.reps)
            print(f"  {name:>7}: {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    if len(args.arms) == 2:
        a, b = args.arms
        ma, ta, _ = results[a]
        mb, tb, _ = results[b]
        shared = sorted(set(ta) & set(tb))
        if shared and ma.size and mb.size:
            ia = [ta.index(t) for t in shared]
            ib = [tb.index(t) for t in shared]
            da, db = ma[:, ia], mb[:, ib]
            diff = iqm(da) - iqm(db)
            # Bootstrap the difference directly: comparing two separate CIs is
            # not a test, and overlapping CIs do not imply no difference.
            rng = np.random.default_rng(0)
            draws = np.empty(args.reps)
            for r in range(args.reps):
                idxa = rng.integers(0, da.shape[0], size=da.shape)
                idxb = rng.integers(0, db.shape[0], size=db.shape)
                draws[r] = (iqm(np.take_along_axis(da, idxa, axis=0))
                            - iqm(np.take_along_axis(db, idxb, axis=0)))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"\n=== {a} - {b} on {len(shared)} shared tasks ===")
            print(f"  IQM difference: {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            print("  crosses zero -> not a win" if lo <= 0 <= hi
                  else "  CI excludes zero")


if __name__ == "__main__":
    main()

"""Aggregate metrics.jsonl across tasks and seeds into a comparable score.

Computes the Agarwal et al. (2021) aggregates -- median, IQM, mean -- with
stratified bootstrap CIs over the (task, seed) matrix.

Expected layout:  logdir/<arm>/<task>/seed<k>/metrics.jsonl

    python benchmarks/eval/aggregate.py ./logdir --arms shs baseline \
        --metric eval_log_success --at 500000

`--metric eval_log_success` gives Meta-World success rate averaged over the
eval episodes; `--metric eval_return --divide-by 4492` gives normalised return.

Caveat worth stating in any writeup: a bootstrap CI over 3 seeds is wide and
mostly reflects seed noise. If the CIs overlap, the honest word is
"comparable", not "beats".
"""
import argparse, json, pathlib
import numpy as np


def read_curve(path, metric):
    """Return (steps, values) for one run.

    metrics.jsonl is APPENDED to, so a logdir resumed after a timeout -- or
    rerun after a crash -- contains several runs' rows with repeated steps. Keep
    the LAST value per step so a stale row from an abandoned run cannot win the
    `last N points` selection.
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
    return steps, np.array([latest[s] for s in steps], np.float64)


def value_at(steps, values, at, window=5):
    if len(steps) == 0:
        return np.nan
    mask = steps <= at if at is not None else np.ones_like(steps, bool)
    if not mask.any():
        return np.nan
    return float(np.mean(values[mask][-window:]))


def collect(logdir, arm, metric, at, window=5):
    out = {}
    for jsonl in sorted((pathlib.Path(logdir) / arm).glob("*/*/metrics.jsonl")):
        task = jsonl.parent.parent.name
        steps, values = read_curve(jsonl, metric)
        score = value_at(steps, values, at, window)
        if not np.isnan(score):
            out.setdefault(task, []).append(score)
    return out


def to_matrix(per_task):
    tasks = sorted(per_task)
    if not tasks:
        return np.zeros((0, 0)), tasks
    n = min(len(per_task[t]) for t in tasks)
    return np.array([[per_task[t][s] for t in tasks] for s in range(n)]), tasks


def iqm(x):
    flat = np.sort(x.reshape(-1))
    lo, hi = int(0.25 * len(flat)), int(np.ceil(0.75 * len(flat)))
    return float(np.mean(flat[lo:hi])) if hi > lo else float(np.mean(flat))


AGGREGATES = {"median": lambda x: float(np.median(np.mean(x, axis=0))),
              "iqm": iqm,
              "mean": lambda x: float(np.mean(x))}


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
    ap.add_argument("--metric", default="eval_log_success")
    ap.add_argument("--at", type=float, default=None)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--divide-by", type=float, default=1.0)
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
        print(f"  {n_tasks} tasks x {n_seeds} seeds (truncated to the min across tasks)")
        dropped = {t: len(v) for t, v in per_task.items() if len(v) > n_seeds}
        if dropped:
            print(f"  NOTE: dropped extra seeds from {dropped}")
        for name, fn in AGGREGATES.items():
            lo, hi = stratified_bootstrap(matrix, fn, args.reps)
            print(f"  {name:>7}: {fn(matrix):.4f}  [{lo:.4f}, {hi:.4f}]")

    if len(args.arms) == 2:
        a, b = args.arms
        ma, ta, _ = results[a]
        mb, tb, _ = results[b]
        shared = sorted(set(ta) & set(tb))
        if shared and ma.size and mb.size:
            da = ma[:, [ta.index(t) for t in shared]]
            db = mb[:, [tb.index(t) for t in shared]]
            # Bootstrap the DIFFERENCE directly: comparing two separate CIs is
            # not a test, and overlapping CIs do not imply no difference.
            rng = np.random.default_rng(0)
            draws = np.empty(args.reps)
            for r in range(args.reps):
                ia = rng.integers(0, da.shape[0], size=da.shape)
                ib = rng.integers(0, db.shape[0], size=db.shape)
                draws[r] = (iqm(np.take_along_axis(da, ia, axis=0))
                            - iqm(np.take_along_axis(db, ib, axis=0)))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"\n=== {a} - {b} on {len(shared)} shared tasks ===")
            print(f"  IQM difference: {iqm(da)-iqm(db):+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            print("  crosses zero -> not a win" if lo <= 0 <= hi else "  CI excludes zero")


if __name__ == "__main__":
    main()

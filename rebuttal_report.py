#!/usr/bin/env python3
"""Tables for the rebuttal runs, read from the metrics.jsonl files of dreamer.py.

    python rebuttal_report.py logdir/rebuttal                      # everything that is there
    python rebuttal_report.py logdir/rebuttal --at 500000 --md     # at 500k frames, Markdown
    python rebuttal_report.py logdir/rebuttal --at 500000 --arms dv3_gauss,rb_k1,rb_gate,rb_full
    python rebuttal_report.py logdir/rebuttal --extra dv3:carl_walker=logdir/carl_walker_vanilla
    python rebuttal_report.py logdir/shs_moves --phases            # which curriculum phases old runs reached

Runs are expected as <root>/<task>/<arm>/seed_<n>/metrics.jsonl (the layout of the rebuttal
arms of the launchers).  --extra ARM:TASK=DIR adds runs kept elsewhere (DIR holds seed_<n>
directories, or is one run).  Needs numpy only.

With --at N only runs that have reached N frames are used; the others are listed and left
out (--partial keeps them, marked).  A final return is the mean of the evaluations inside the
last --window frames; a run without an evaluation there shows "-" and is not carried forward.
The aggregate is taken over the tasks that every listed arm covers (choose the arms to compare
with --arms).

Tables: returns per task and arm (final and averaged over training), the interquartile mean
over tasks and seeds with a stratified bootstrap interval, per-context returns (CARL), regime
counts, structure-move statistics, prediction errors, regime/context agreement and throughput.
"""
import argparse
import csv
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

ARM_ORDER = ["dv3", "dv3_gauss", "dv3_ctx", "rb_k1", "rb_nostick", "rb_sticky", "rb_gate", "rb_full",
             "rb_ema", "rb_power", "rb_retain", "rb_noforget", "rb_frozen", "rb_handover"]
ARM_NOTE = {
    "dv3": "DreamerV3 (categorical)", "dv3_gauss": "DreamerV3, Gaussian latents",
    "dv3_ctx": "DreamerV3, context observed", "rb_k1": "SHS, K=1",
    "rb_nostick": "SHS, K=16, no persistence", "rb_sticky": "SHS, K=16, constant stickiness",
    "rb_gate": "SHS, K=16, state-dependent gate", "rb_full": "SHS, gate + structure moves",
    "rb_ema": "as rb_gate, plain EMA", "rb_power": "as rb_gate, power prior",
    "rb_retain": "as rb_gate, inferred retention",
    "rb_noforget": "as rb_gate, uniform average", "rb_frozen": "as rb_gate, factors frozen at 200k",
    "rb_handover": "as rb_full, memory hand-over",
}
MOVES = ["empty", "merge", "delete", "birth", "split"]


def arm_key(arm):
    for i, a in enumerate(ARM_ORDER):
        if arm == a:
            return (i, 0, arm)
    m = re.fullmatch(r"rb_K(\d+)(.*)", arm)
    if m:
        return (len(ARM_ORDER), int(m.group(1)), arm)
    for i, a in enumerate(ARM_ORDER):
        if arm.startswith(a + "_"):
            return (i, 1, arm)
    return (len(ARM_ORDER) + 1, 0, arm)


class Run:
    def __init__(self, path, task, arm, seed):
        self.path, self.task, self.arm, self.seed = path, task, arm, seed
        self.series = {}
        self.last_step = 0.0
        bad = 0
        with open(path / "metrics.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                step = row.get("step")
                if step is None:
                    continue
                step = float(step)
                self.last_step = max(self.last_step, step)
                for k, v in row.items():
                    if k == "step" or not isinstance(v, (int, float)):
                        continue
                    self.series.setdefault(k, []).append((step, float(v)))
        self.bad_lines = bad

    def values(self, key, lo=-math.inf, hi=math.inf):
        return [v for s, v in self.series.get(key, ()) if lo < s <= hi and math.isfinite(v)]

    def window_mean(self, key, at, window):
        """Mean of the values logged in (at - window, at]; NaN if there is none.  Nothing older
        is carried forward, in any table or in the CSV."""
        v = self.values(key, at - window, at)
        return float(np.mean(v)) if v else math.nan

    def last_step_of(self, key, at):
        s = [s for s, v in self.series.get(key, ()) if s <= at and math.isfinite(v)]
        return s[-1] if s else math.nan

    def last(self, key, at):
        v = self.values(key, -math.inf, at)
        return v[-1] if v else math.nan

    def maximum(self, key, at):
        v = self.values(key, -math.inf, at)
        return max(v) if v else math.nan

    def first_step_at_least(self, key, value):
        for s, v in self.series.get(key, ()):
            if v >= value:
                return s
        return math.nan


def find_metrics(root):
    """Every metrics.jsonl under root, following symbolic links to directories."""
    found = []
    for d, _, files in os.walk(root, followlinks=True):
        if "metrics.jsonl" in files:
            found.append(pathlib.Path(d) / "metrics.jsonl")
    return sorted(found)


def discover(roots, extras):
    runs = []
    for root in roots:
        root = pathlib.Path(root)
        for m in find_metrics(root):
            d = m.parent
            s = re.fullmatch(r"seed_?(\d+)", d.name)
            if not s or len(d.relative_to(root).parts) < 3:
                continue
            runs.append((d, d.parent.parent.name, d.parent.name, int(s.group(1))))
    for spec in extras:
        try:
            label, d = spec.split("=", 1)
            arm, task = label.split(":", 1)
        except ValueError:
            sys.exit(f"--extra expects ARM:TASK=DIR, got {spec!r}")
        d = pathlib.Path(d)
        found = [d] if (d / "metrics.jsonl").exists() else sorted(p.parent for p in d.glob("seed*/metrics.jsonl"))
        if not found:
            sys.exit(f"--extra {spec}: no metrics.jsonl in {d} or its seed_* directories")
        for i, r in enumerate(found):
            s = re.search(r"seed_?(\d+)", r.name)
            runs.append((r, task, arm, int(s.group(1)) if s else i + 1))
    out, paths, ids = [], set(), {}
    for d, task, arm, seed in runs:
        real = os.path.realpath(d)
        if real in paths:                       # the same directory reached twice
            continue
        paths.add(real)
        if (task, arm, seed) in ids:
            print(f"note: two directories for {task} {arm} seed {seed}: using {ids[(task, arm, seed)]}, ignoring {d}")
            continue
        r = Run(d, task, arm, seed)
        if r.series:
            ids[(task, arm, seed)] = d
            out.append(r)
    return out


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    cut = int(0.25 * len(x))
    return float(x[cut:len(x) - cut].mean()) if len(x) else math.nan


def iqm_interval(by_task, reps=5000, seed=0):
    """Interquartile mean over all (task, seed) scores; 95% interval from a bootstrap that
    resamples the seeds of each task separately."""
    rng = np.random.default_rng(seed)
    tasks = [np.asarray(v, dtype=float) for v in by_task.values() if len(v)]
    if not tasks:
        return math.nan, math.nan, math.nan
    point = iqm(np.concatenate(tasks))
    boot = [iqm(np.concatenate([t[rng.integers(0, len(t), len(t))] for t in tasks])) for _ in range(reps)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt(x, digits=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def pm(values, digits=1):
    v = [x for x in values if math.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"


def table(title, head, rows, md):
    if not rows:
        return
    print()
    if md:
        print(f"**{title}**\n")
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for r in rows:
            print("| " + " | ".join(str(c) for c in r) + " |")
    else:
        print(title)
        w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
        line = "  ".join(str(c).ljust(w[i]) for i, c in enumerate(head))
        print(line)
        print("-" * len(line))
        for r in rows:
            print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="directories holding <task>/<arm>/seed_<n>/metrics.jsonl")
    ap.add_argument("--extra", action="append", default=[], metavar="ARM:TASK=DIR")
    ap.add_argument("--at", type=float, default=None,
                    help="evaluate at this number of frames; runs that have not reached it are left out "
                         "(default: the end of each run)")
    ap.add_argument("--partial", action="store_true", help="with --at: keep runs that have not reached it, marked *")
    ap.add_argument("--arms", default=None, help="comma-separated arms to report (default: all found)")
    ap.add_argument("--window", type=float, default=50000, help="frames averaged for a final value (default 50000)")
    ap.add_argument("--scale", type=float, default=1000.0, help="return that counts as 1 in the aggregate (default 1000)")
    ap.add_argument("--md", action="store_true", help="Markdown tables")
    ap.add_argument("--csv", default=None, help="write one row per run with all final values")
    ap.add_argument("--curves", default=None, help="write the evaluation curves (task, arm, seed, step, eval_return)")
    ap.add_argument("--phases", action="store_true", help="only list which curriculum phases each run reached")
    args = ap.parse_args()
    if not args.roots and not args.extra:
        ap.error("give at least one directory")

    if args.phases:
        rows = []
        for root in args.roots:
            root = pathlib.Path(root)
            for m in find_metrics(root):
                r = Run(m.parent, "", "", 0)
                ph = r.series.get("shs_curriculum_phase")
                if not ph:
                    rows.append([str(m.parent.relative_to(root)), fmt(r.last_step, 0), "not logged", "-", "-", "-", "-"])
                    continue
                top = int(max(v for _, v in ph))
                starts = " ".join(f"{p}@{fmt(r.first_step_at_least('shs_curriculum_phase', p), 0)}" for p in range(1, top + 1))
                rows.append([str(m.parent.relative_to(root)), fmt(r.last_step, 0), str(top), starts or "-",
                             fmt(r.maximum("shs_recurrent_live", math.inf), 0), fmt(r.maximum("shs_move_birth_live", math.inf), 0),
                             fmt(r.maximum("shs_move_split_live", math.inf), 0)])
        table("Curriculum phases reached (0 = the run never left its first phase)",
              ["run", "frames", "last phase", "phase@frame", "gate on", "births on", "splits on"], rows, args.md)
        return

    runs = discover(args.roots, args.extra)
    if not runs:
        sys.exit("no runs found: expected <root>/<task>/<arm>/seed_<n>/metrics.jsonl")
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        missing = [a for a in want if a not in {r.arm for r in runs}]
        if missing:
            print(f"note: no runs found for {', '.join(missing)}")
        runs = [r for r in runs if r.arm in want]
        if not runs:
            sys.exit("no runs left after --arms")
    left_out = []
    if args.at is not None and not args.partial:
        left_out = [r for r in runs if r.last_step < args.at - 1e-6]
        runs = [r for r in runs if r.last_step >= args.at - 1e-6]
        if left_out:
            print(f"{len(left_out)} runs have not reached {args.at:.0f} frames and are left out (--partial keeps them):")
            for r in sorted(left_out, key=lambda r: (r.task, arm_key(r.arm), r.seed)):
                print(f"  {r.task} {r.arm} seed {r.seed}: {r.last_step:.0f} frames")
        if not runs:
            sys.exit(f"no run has reached {args.at:.0f} frames yet")
    tasks = sorted({r.task for r in runs})
    cell = {}
    for r in runs:
        cell.setdefault((r.task, r.arm), []).append(r)
    for v in cell.values():
        v.sort(key=lambda r: r.seed)
    arms = sorted({r.arm for r in runs}, key=arm_key)
    W = args.window

    def at_of(r):
        return r.last_step if args.at is None else min(args.at, r.last_step)

    def short(r):
        return args.at is not None and r.last_step < args.at - 1e-6

    def agg(rs, fn):
        return [fn(r) for r in rs]

    # ---- 1. returns ---------------------------------------------------------------------------
    rows, final = [], {}
    for t in tasks:
        for a in arms:
            rs = cell.get((t, a))
            if not rs:
                continue
            fin = agg(rs, lambda r: r.window_mean("eval_return", at_of(r), W))
            auc = agg(rs, lambda r: float(np.mean(r.values("eval_return", -math.inf, at_of(r)) or [math.nan])))
            trn = agg(rs, lambda r: r.window_mean("train_return", at_of(r), W))
            final[(t, a)] = [x for x in fin if math.isfinite(x)]
            frames = [r.last_step for r in rs]
            mark = " *" if any(short(r) for r in rs) else ""
            stale = sum(1 for x in fin if not math.isfinite(x))
            rows.append([t, a, ARM_NOTE.get(a, ""), " ".join(str(r.seed) for r in rs),
                         pm(fin) + (f" ({len(rs) - stale} of {len(rs)})" if stale else ""), pm(auc), pm(trn),
                         f"{min(frames) / 1e3:.0f}k" + ("" if min(frames) == max(frames) else f"-{max(frames) / 1e3:.0f}k") + mark])
    where = "the end of each run" if args.at is None else f"{args.at:.0f} frames"
    table(f"Returns at {where}: evaluation return averaged over the last {W:.0f} frames, mean ± s.d. over seeds"
          + (" (* = some seeds have not reached that budget)" if args.at is not None and args.partial else ""),
          ["task", "arm", "model", "seeds", "eval return", "mean over training", "train return", "frames"], rows, args.md)
    for t in tasks:
        sets = {a: tuple(r.seed for r in cell[(t, a)]) for a in arms if (t, a) in cell}
        if len(set(sets.values())) > 1:
            print(f"note: {t}: the arms do not have the same seeds: " + "; ".join(f"{a}: {' '.join(map(str, s))}" for a, s in sets.items()))

    # ---- 2. aggregate -------------------------------------------------------------------------
    rows = []
    suites = sorted({t.split("_")[0] for t in tasks})
    for s in suites:
        stasks = [t for t in tasks if t.split("_")[0] == s]
        sarms = [a for a in arms if any(final.get((t, a)) for t in stasks)]
        common = [t for t in stasks if all(final.get((t, a)) for a in sarms)]
        if len(sarms) < 2:
            continue
        if not common:
            print(f"note: {s}: no task is covered by all of {', '.join(sarms)}; no aggregate (narrow the comparison with --arms)")
            continue
        for a in sarms:
            by_task = {t: [x / args.scale for x in final[(t, a)]] for t in common}
            p, lo, hi = iqm_interval(by_task)
            mean = float(np.mean([np.mean(v) for v in by_task.values()]))
            rows.append([s, a, " ".join(t[len(s) + 1:] for t in common), str(sum(len(v) for v in by_task.values())),
                         f"{p:.3f}", f"[{lo:.3f}, {hi:.3f}]", f"{mean:.3f}"])
    table(f"Aggregate over the tasks that every listed arm covers (return / {args.scale:.0f}): interquartile mean over "
          "tasks and seeds with a bootstrap interval over seeds",
          ["suite", "arm", "tasks", "runs", "IQM", "95% interval", "mean of task means"], rows, args.md)

    # ---- 3. per-context returns (CARL) --------------------------------------------------------
    ctx_keys = sorted({k for r in runs for k in r.series if re.fullmatch(r"eval_return_ctx\d+", k)},
                      key=lambda k: int(k[len("eval_return_ctx"):]))
    if ctx_keys:
        rows = []
        for t in tasks:
            for a in arms:
                rs = cell.get((t, a))
                if not rs or not any(k in r.series for r in rs for k in ctx_keys):
                    continue
                rows.append([t, a, str(len(rs))] + [pm(agg(rs, lambda r, k=k: r.window_mean(k, at_of(r), W))) for k in ctx_keys])
        table("Evaluation return per evaluation context (held-out contexts in the CARL presets)",
              ["task", "arm", "seeds"] + [k.replace("eval_return_", "") for k in ctx_keys], rows, args.md)

    # ---- 4. regimes ---------------------------------------------------------------------------
    rows = []
    for t in tasks:
        for a in arms:
            rs = [r for r in cell.get((t, a), []) if "shs_current_K" in r.series]
            if not rs:
                continue
            rows.append([t, a, str(len(rs)),
                         pm(agg(rs, lambda r: r.last("shs_current_K", at_of(r)))),
                         pm(agg(rs, lambda r: r.window_mean("shs_active_regimes", at_of(r), W))),
                         pm(agg(rs, lambda r: r.window_mean("shs_regime_entropy", at_of(r), W)), 2),
                         pm(agg(rs, lambda r: r.window_mean("shs_memory_top_share", at_of(r), W)), 2),
                         fmt(np.nanmin(agg(rs, lambda r: r.maximum("shs_curriculum_phase", at_of(r)))), 0),
                         pm(agg(rs, lambda r: r.last("shs_reseeds", at_of(r))), 0)])
    table("Regimes: instantiated K, regimes with more than 1% of the occupancy of a replay batch, entropy of the "
          "occupancy (nats), largest regime's share of the stored statistics",
          ["task", "arm", "seeds", "K", "occupied", "entropy", "top share", "last phase", "re-seeds"], rows, args.md)

    # ---- 5. structure moves -------------------------------------------------------------------
    rows = []
    for t in tasks:
        for a in arms:
            rs = [r for r in cell.get((t, a), []) if r.maximum("shs_sweeps_run", at_of(r)) > 0]
            if not rs:
                continue
            for m in MOVES:
                att = agg(rs, lambda r: r.last(f"shs_moves_{m}_attempts", at_of(r)))
                tried = agg(rs, lambda r: r.last(f"shs_moves_{m}_tried", at_of(r)))
                acc = agg(rs, lambda r: r.last(f"shs_moves_{m}_accepted", at_of(r)))
                if not any(math.isfinite(x) and x > 0 for x in att):
                    continue
                tt, aa = np.nansum(tried), np.nansum(acc)
                rows.append([t, a, m, str(len(rs)), fmt(np.nanmean(att), 1), fmt(np.nanmean(tried), 1), fmt(np.nanmean(acc), 1),
                             f"{100 * aa / tt:.0f}%" if tt > 0 else "-",
                             pm(agg(rs, lambda r: r.last(f"shs_moves_{m}_holdout_gain", at_of(r))), 3),
                             pm(agg(rs, lambda r: r.last(f"shs_moves_{m}_fit_gain", at_of(r))), 1)])
            rows.append([t, a, "all", str(len(rs)), fmt(np.nanmean(agg(rs, lambda r: r.last("shs_sweeps_run", at_of(r)))), 1) + " sweeps",
                         "", "", "", "held-out vetoes " + fmt(np.nanmean(agg(rs, lambda r: r.last("shs_move_holdout_vetoes", at_of(r)))), 1),
                         "fit vetoes " + fmt(np.nanmean(agg(rs, lambda r: r.last("shs_move_fit_vetoes", at_of(r)))), 1)])
    table("Structure moves per run (means over seeds of the cumulative counts): sweeps in which the move was searched, "
          "candidates installed for the acceptance checks, candidates kept, and for the kept ones the mean change of the "
          "held-out forecast score (nats per step) and of the window bound (nats)",
          ["task", "arm", "move", "seeds", "searched", "installed", "kept", "kept/installed", "held-out gain", "bound gain"], rows, args.md)

    # ---- 6. prediction ------------------------------------------------------------------------
    rows = []
    for t in tasks:
        for a in arms:
            rs = cell.get((t, a))
            if not rs:
                continue
            cols = [pm(agg(rs, lambda r, k=k: r.window_mean(k, at_of(r), W)), d)
                    for k, d in (("openl_mse", 5), ("openl_mse_15", 5), ("eval_openl_mse", 5), ("kl", 2), ("image_loss", 1))]
            if all(c == "-" for c in cols):
                continue
            cov = [v for r in rs for v in r.values("eval_openl_contexts", at_of(r) - W, at_of(r))]
            rows.append([t, a, str(len(rs))] + cols + [fmt(min(cov), 0) if cov else "-"])
    table("Prediction: open-loop pixel error of the prior rollout after 5 context frames, counted within episodes "
          "(replay batches: whole rollout and first 15 steps; evaluation episodes: the completed episodes of the "
          "evaluation round, every context with a counted step weighing the same), KL between posterior and prior, "
          "image loss.  'contexts scored' is the smallest number of evaluation contexts behind an evaluation error, "
          "over every evaluation of the window and every seed.  Each run is scored on its own trajectories, so these "
          "columns are diagnostics of a run and not a comparison on common data; score_prediction.py scores the saved "
          "models of all arms on one set of held-out episodes",
          ["task", "arm", "seeds", "open-loop MSE", "MSE 15 steps", "MSE eval episodes", "KL", "image loss", "contexts scored"], rows, args.md)

    # ---- 7. regime / context agreement --------------------------------------------------------
    rows = []
    for t in tasks:
        for a in arms:
            rs = [r for r in cell.get((t, a), []) if "shs_context_nmi" in r.series]
            if not rs:
                continue
            rows.append([t, a, str(len(rs)), pm(agg(rs, lambda r: r.window_mean("shs_context_nmi", at_of(r), W)), 3),
                         pm(agg(rs, lambda r: r.window_mean("shs_context_explained", at_of(r), W)), 3),
                         pm(agg(rs, lambda r: r.maximum("shs_context_nmi", at_of(r))), 3),
                         fmt(np.nanmean(agg(rs, lambda r: r.window_mean("shs_context_count", at_of(r), W))), 1)])
    table("Regimes and the hidden context (training contexts, replay batches): normalised mutual information between "
          "regime and context, share of the context entropy explained by the regime (0 = unrelated, 1 = the regime "
          "identifies the context)",
          ["task", "arm", "seeds", "NMI", "explained", "max NMI in training", "contexts per batch"], rows, args.md)

    # ---- 8. throughput ------------------------------------------------------------------------
    rows = []
    for t in tasks:
        for a in arms:
            rs = cell.get((t, a))
            if not rs:
                continue
            fps = [float(np.median([v for v in r.values("fps", -math.inf, at_of(r)) if v > 0] or [math.nan])) for r in rs]
            if not any(math.isfinite(x) for x in fps):
                continue
            f = float(np.nanmean(fps))
            rows.append([t, a, str(len(rs)), fmt(f, 1), fmt(1000000 / f / 3600, 1) if f > 0 else "-"])
    table("Throughput (median of the logged frames per second between training logs, evaluation episodes "
          "included; depends on the GPU and on what shares it)",
          ["task", "arm", "seeds", "frames/s", "hours per 1M frames"], rows, args.md)

    if args.csv:
        keys = ["eval_return", "train_return", "shs_current_K", "shs_active_regimes", "shs_regime_entropy",
                "shs_memory_mass", "shs_memory_top_share", "shs_curriculum_phase", "shs_sweeps_run",
                "shs_move_holdout_vetoes", "shs_move_fit_vetoes", "openl_mse", "openl_mse_15", "eval_openl_mse",
                "kl", "image_loss", "shs_context_nmi", "shs_context_explained", "fps"] + ctx_keys
        keys += [f"shs_moves_{m}_{s}" for m in MOVES for s in ("attempts", "tried", "accepted", "holdout_gain", "fit_gain")]
        counters = ("shs_current_K", "shs_curriculum_phase", "shs_sweeps_run", "shs_move_holdout_vetoes", "shs_move_fit_vetoes")
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "arm", "seed", "frames", "at", "window", "last_eval_step", "path"] + keys)
            for r in sorted(runs, key=lambda r: (r.task, arm_key(r.arm), r.seed)):
                at = at_of(r)
                vals = [r.last(k, at) if (k in counters or k.startswith("shs_moves_")) else r.window_mean(k, at, W) for k in keys]
                le = r.last_step_of("eval_return", at)
                w.writerow([r.task, r.arm, r.seed, r.last_step, at, W, "" if not math.isfinite(le) else f"{le:.0f}", str(r.path)]
                           + ["" if not math.isfinite(v) else f"{v:.6g}" for v in vals])
        print(f"\nwrote {args.csv}")
    if args.curves:
        with open(args.curves, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", "arm", "seed", "step", "eval_return"])
            for r in sorted(runs, key=lambda r: (r.task, arm_key(r.arm), r.seed)):
                for s, v in r.series.get("eval_return", ()):
                    w.writerow([r.task, r.arm, r.seed, f"{s:.0f}", f"{v:.4f}"])
        print(f"wrote {args.curves}")
    bad = sum(r.bad_lines for r in runs)
    if bad:
        print(f"\nnote: {bad} unreadable lines skipped in metrics.jsonl files (a run may have been cut while writing)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print a compact side-by-side eval curve for two or more arms.

Small enough to paste into a chat or a notes file, unlike the raw .out logs.

    python3 benchmarks/eval/summarize.py door-open
    python3 benchmarks/eval/summarize.py door-open --logdir ./logdir
    python3 benchmarks/eval/summarize.py hammer --arms shs_fixedgoal baseline_fixedgoal

Handles the duplicate-step rows that resumed runs leave in metrics.jsonl by
keeping the last value seen for each step.
"""

import argparse
import glob
import json
import os

DIAG = ("shs_current_K", "shs_active_regimes", "shs_regime_entropy",
        "shs_expected_self_transition", "shs_move_birth_gain",
        "shs_curriculum_phase", "actor_entropy", "imag_reward_std",
        "value_std", "actor_grad_norm", "update_count", "fps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--logdir", default="./logdir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    pattern = os.path.join(args.logdir, "*", args.task, f"seed{args.seed}",
                           "metrics.jsonl")
    paths = sorted(glob.glob(pattern))
    if args.arms:
        paths = [p for p in paths
                 if p.split(os.sep)[-4] in args.arms]
    if not paths:
        print(f"no metrics.jsonl found matching {pattern}")
        return

    evals, diags = {}, {}
    for p in paths:
        arm = p.split(os.sep)[-4]
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # later rows win: resumed runs append duplicate steps
            if "eval_log_success" in r:
                evals.setdefault(r["step"], {})[arm] = (
                    r.get("eval_return"), r["eval_log_success"])
            if any(k in r for k in DIAG):
                diags[arm] = r

    arms = sorted({a for v in evals.values() for a in v})
    print(f"task: {args.task}   seed {args.seed}\n")
    print("step".ljust(9) + "".join(f"{a[:20]:>24s}" for a in arms))
    for s in sorted(evals):
        line = f"{s:<9d}"
        for a in arms:
            v = evals[s].get(a)
            cell = f"ret {v[0]:.0f}  succ {v[1]:.2f}" if v else "-"
            line += f"{cell:>24s}"
        print(line)

    print("\nfinal diagnostics")
    for a in arms:
        r = diags.get(a, {})
        present = {k: r[k] for k in DIAG if k in r}
        print(f"  {a}: {present}")


if __name__ == "__main__":
    main()

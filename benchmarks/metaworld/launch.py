#!/usr/bin/env python3
"""Emit the full Meta-World sweep as one command per line.

Prints to stdout by default so you can pipe it wherever it needs to go:

    python benchmarks/metaworld/launch.py                       # inspect
    python benchmarks/metaworld/launch.py | wc -l               # count jobs
    python benchmarks/metaworld/launch.py --sbatch > jobs.txt   # SLURM array
    python benchmarks/metaworld/launch.py | xargs -P4 -I{} sh -c '{}'

Logdir layout matches what benchmarks/eval/aggregate.py expects:

    <logdir>/<arm>/<task>/seed<k>/

Three arms are emitted by default and this is deliberate:

  baseline  stock DreamerV3 (categorical latent, amortised prior)
  gauss     continuous latent, amortised prior
  shs       continuous latent, switching prior

`use_shs: True` forces `dyn_discrete: 0`, so baseline-vs-shs changes two things
at once. The `gauss` arm is what lets you attribute a difference to the
switching prior rather than to the latent type. Dropping it saves a third of
the compute and costs you the ability to make the claim.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tasks as T  # noqa: E402

ARMS = {
    # arm name -> (--configs value, extra flags)
    "baseline": ("metaworld_proprio", ""),
    "gauss": ("metaworld_proprio_gauss", ""),
    "shs": ("metaworld_proprio_shs", ""),
}

VISION_ARMS = {
    "baseline": ("metaworld_vision", ""),
    "shs": ("metaworld_vision_shs", ""),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite15",
                    choices=["suite15", "suite6", "easy", "medium", "hard",
                             "all"])
    ap.add_argument("--arms", nargs="+", default=["baseline", "gauss", "shs"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--logdir", default="./logdir")
    ap.add_argument("--vision", action="store_true",
                    help="pixel observations instead of proprioceptive state")
    ap.add_argument("--steps", type=float, default=None,
                    help="override the per-tier step budget")
    ap.add_argument("--sbatch", action="store_true",
                    help="wrap each command in an sbatch invocation")
    ap.add_argument("--sbatch-args", default="--gres=gpu:1 --mem=32G --time=24:00:00")
    args = ap.parse_args()

    table = VISION_ARMS if args.vision else ARMS
    unknown = [a for a in args.arms if a not in table]
    if unknown:
        ap.error(f"unknown arm(s) {unknown} for this mode; have {list(table)}")

    if args.suite == "suite15":
        rows = T.flat(T.SUITE_15)
    elif args.suite == "suite6":
        rows = [(t, "hard", T.TIER_STEPS["hard"]) for t in T.SUITE_6]
    elif args.suite == "all":
        rows = T.flat(T.TIERS)
    else:
        rows = [(t, args.suite, T.TIER_STEPS[args.suite])
                for t in T.TIERS[args.suite]]

    n = 0
    for task, _tier, tier_steps in rows:
        steps = args.steps if args.steps is not None else tier_steps
        for arm in args.arms:
            config, extra = table[arm]
            for seed in range(args.seeds):
                logdir = f"{args.logdir}/{arm}/{task}/seed{seed}"
                cmd = (
                    f"python3 dreamer.py --configs {config}"
                    f" --task metaworld_{task}"
                    f" --seed {seed}"
                    f" --steps {int(steps)}"
                    f" --logdir {logdir}"
                )
                if extra:
                    cmd += f" {extra}"
                if args.sbatch:
                    cmd = (f"sbatch {args.sbatch_args} "
                           f"--job-name=mw-{arm}-{task}-s{seed} --wrap='{cmd}'")
                print(cmd)
                n += 1
    print(f"# {n} jobs "
          f"({len(rows)} tasks x {len(args.arms)} arms x {args.seeds} seeds)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

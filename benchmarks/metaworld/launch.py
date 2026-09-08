#!/usr/bin/env python3
"""Emit the full Meta-World sweep as one command per line.

    python3 benchmarks/metaworld/launch.py                       # inspect
    python3 benchmarks/metaworld/launch.py | wc -l               # count jobs
    python3 benchmarks/metaworld/launch.py --sbatch > jobs.txt

Logdir layout matches benchmarks/eval/aggregate.py:  <logdir>/<arm>/<task>/seed<k>/

MEASURED throughput on an H100 3g.40gb MIG slice, proprio: baseline 13.3 env
steps/s, SHS 7.7 (0.57x, torch.compile is disabled when use_shs is True). So
500k steps is ~10.4h baseline / ~18h SHS. Size the sweep off the SHS number.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tasks as T  # noqa: E402

ARMS = {
    "baseline": ("metaworld_proprio", ""),
    "gauss": ("metaworld_proprio_gauss", ""),   # BROKEN, see configs.yaml
    "shs": ("metaworld_proprio_shs", ""),
}
VISION_ARMS = {
    "baseline": ("metaworld_vision", ""),
    "shs": ("metaworld_vision_shs", ""),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite15",
                    choices=["suite15", "suite6", "easy", "medium", "hard", "all"])
    ap.add_argument("--arms", nargs="+", default=["baseline", "shs"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--logdir", default="./logdir")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--steps", type=float, default=None)
    ap.add_argument("--extra", default="", help="flags appended to every command")
    ap.add_argument("--sbatch", action="store_true")
    ap.add_argument("--sbatch-args", default="--gres=gpu:1 --mem=24G --time=24:00:00")
    args = ap.parse_args()

    table = VISION_ARMS if args.vision else ARMS
    unknown = [a for a in args.arms if a not in table]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; have {list(table)}")

    if args.suite == "suite15":
        rows = T.flat(T.SUITE_15)
    elif args.suite == "suite6":
        rows = [(t, "hard", T.TIER_STEPS["hard"]) for t in T.SUITE_6]
    elif args.suite == "all":
        rows = T.flat(T.TIERS)
    else:
        rows = [(t, args.suite, T.TIER_STEPS[args.suite]) for t in T.TIERS[args.suite]]

    n = 0
    for task, _tier, tier_steps in rows:
        steps = args.steps if args.steps is not None else tier_steps
        for arm in args.arms:
            config, extra = table[arm]
            for seed in range(args.seeds):
                logdir = f"{args.logdir}/{arm}/{task}/seed{seed}"
                cmd = (f"python3 -u dreamer.py --configs {config}"
                       f" --task metaworld_{task} --seed {seed}"
                       f" --steps {int(steps)} --logdir {logdir}")
                for x in (extra, args.extra):
                    if x:
                        cmd += f" {x}"
                if args.sbatch:
                    cmd = (f"sbatch {args.sbatch_args} "
                           f"--job-name=mw-{arm}-{task}-s{seed} --wrap='{cmd}'")
                print(cmd)
                n += 1
    print(f"# {n} jobs ({len(rows)} tasks x {len(args.arms)} arms x {args.seeds} seeds)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the full comparison: {shs, trslds, rslds} x {nascar, toyark13, mocap6}
x seeds, then aggregate tables and figures.

Each fit is a subprocess of the matching runner, so per-model environments
stay isolated and a crash in one run does not kill the sweep.  Runs that
already have a result file are skipped unless --force.

rSLDS needs the legacy 2017 stack (python<=3.8; see
environment-baselines.yml).  When that stack is absent here the rslds runs
are skipped with a note -- fit them in the legacy environment with the same
--tag convention and their npz files will join the shared results/ folder;
rerun ``make_figures.py`` afterwards.

Examples
--------
    python run_all.py --quick                       # smoke everything runnable
    python run_all.py --seeds 0 1 2                 # publication sweep
    python run_all.py --datasets nascar --models shs trslds
"""
import argparse
import importlib.util
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

MODELS = ("shs", "trslds", "rslds")
DATASETS = ("nascar", "toyark13", "mocap6")


def _legacy_stack_present():
    return importlib.util.find_spec("pybasicbayes") is not None


def _result_exists(dataset, tag):
    return (HERE / "results" / dataset / f"{tag}.npz").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--quick", action="store_true",
                    help="forwarded to every runner (smoke-test scale)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even when the result file exists")
    ap.add_argument("--latex", action="store_true",
                    help="forwarded to make_figures.py")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after --extra is appended to every "
                         "runner call, e.g. --extra --nseq 12")
    args = ap.parse_args()

    models = list(args.models)
    if "rslds" in models and not _legacy_stack_present():
        print("[run_all] legacy rSLDS stack not importable here -- skipping "
              "rslds runs.\n          Fit them in the environment from "
              "environment-baselines.yml with the same tags;\n          "
              "results drop into the shared results/ folder.")
        models = [m for m in models if m != "rslds"]

    failures = []
    for dataset in args.datasets:
        for model in models:
            for seed in args.seeds:
                tag = model if len(args.seeds) == 1 else f"{model}_seed{seed}"
                if _result_exists(dataset, tag) and not args.force:
                    print(f"[run_all] {dataset}/{tag}: result exists, skipping "
                          "(--force to redo)")
                    continue
                cmd = [sys.executable, str(HERE / f"run_{model}.py"),
                       "--dataset", dataset, "--seed", str(seed),
                       "--tag", tag]
                if args.quick:
                    cmd.append("--quick")
                cmd += args.extra
                print(f"[run_all] {' '.join(cmd[1:])}", flush=True)
                rc = subprocess.call(cmd)
                if rc != 0:
                    failures.append((dataset, tag, rc))
                    print(f"[run_all] {dataset}/{tag} FAILED (rc={rc}); "
                          "continuing", flush=True)

    fig_cmd = [sys.executable, str(HERE / "make_figures.py"),
               "--dataset", "all"]
    if args.latex:
        fig_cmd.append("--latex")
    subprocess.call(fig_cmd)

    if failures:
        print("\n[run_all] failed runs:")
        for d, t, rc in failures:
            print(f"    {d}/{t} (rc={rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()

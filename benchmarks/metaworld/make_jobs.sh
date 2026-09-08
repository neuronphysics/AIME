#!/bin/bash
# Generate a Meta-World job list.
#
#   bash benchmarks/metaworld/make_jobs.sh proprio > jobs_proprio.txt
#   bash benchmarks/metaworld/make_jobs.sh vision  > jobs_vision.txt
#
#   wc -l jobs_proprio.txt        # expect 20 (2 tasks x 5 seeds x 2 arms)
#   sbatch --array=1-20%10 benchmarks/metaworld/sbatch_array.sh jobs_proprio.txt
#
# Both tasks are easy tier, so 5e5 steps is the sanctioned budget
# (Seo et al.). Measured reference returns in our wrapper (random / expert):
#   drawer-open           613 / 4057
#   button-press-topdown  193 / 3652
#
# Randomised goals (the wrapper default) -- no flag needed.
#
# Logdirs: ./logdir/{baseline,shs}[_vision]/<task>/seed<k>/
# The _vision suffix keeps the two observation modes in separate trees so that
# aggregate.py cannot mix them.
#
# WARNING: the vision path has never been run. envs/metaworld.py's
# _setup_renderer() swaps the gymnasium MujocoRenderer to force camera and
# resolution, and it was written without a working GL backend to test against.
# Run the smoke test in the header of sbatch_array.sh before submitting a
# vision sweep, and note that sbatch_array.sh exports MUJOCO_GL=disable, which
# MUST be changed to osmesa (or egl) for vision runs.

set -euo pipefail

MODE=${1:?usage: make_jobs.sh <proprio|vision>}
TASKS="drawer-open button-press-topdown"
SEEDS="0 1 2 3 4"
STEPS=500000

case "$MODE" in
  proprio) BASE=metaworld_proprio;      SHS=metaworld_proprio_shs;      SUF="" ;;
  vision)  BASE=metaworld_vision;       SHS=metaworld_vision_shs;       SUF="_vision" ;;
  *) echo "mode must be proprio or vision" >&2; exit 1 ;;
esac

for task in $TASKS; do
  for seed in $SEEDS; do
    echo "python3 -u dreamer.py --configs $BASE --task metaworld_${task}" \
         "--seed $seed --steps $STEPS --logdir ./logdir/baseline${SUF}/${task}/seed${seed}"
    echo "python3 -u dreamer.py --configs $SHS --task metaworld_${task}" \
         "--seed $seed --steps $STEPS --logdir ./logdir/shs${SUF}/${task}/seed${seed}"
  done
done

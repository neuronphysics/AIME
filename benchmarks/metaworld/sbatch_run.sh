#!/bin/bash
#SBATCH --job-name=mw
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gsubbara/AIME/slurm_logs/%x-%j.out
#SBATCH --error=/home/gsubbara/AIME/slurm_logs/%x-%j.err

# Parameterised version of sbatch_sanity.sh. Same allocation, arbitrary
# config/task/seed, so one file covers the whole sweep.
#
#   sbatch --job-name=mw-shs-hammer-s0 benchmarks/metaworld/sbatch_run.sh \
#          metaworld_proprio_shs hammer 0 1000000
#
# Args: <config> <task> <seed> <steps>

set -euo pipefail

CONFIG=${1:?usage: sbatch_run.sh <config> <task> <seed> <steps>}
TASK=${2:?missing task}
SEED=${3:?missing seed}
STEPS=${4:?missing steps}

REPO=/home/gsubbara/AIME
# Layout matches what benchmarks/eval/aggregate.py expects:
#   <logdir>/<arm>/<task>/seed<k>/
ARM=${CONFIG#metaworld_proprio_}
[ "$ARM" = "metaworld_proprio" ] && ARM=baseline
LOGDIR=./logdir/${ARM}/${TASK}/seed${SEED}

cd "$REPO"

module load opencv mujoco/3.3.0
source aime/bin/activate

export MUJOCO_GL=disable          # proprio only; see sbatch_sanity.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "job ${SLURM_JOB_ID:-interactive} | $CONFIG | $TASK | seed $SEED | $STEPS steps"
echo "logdir $LOGDIR | host $(hostname) | started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Auto-resumes from $LOGDIR/latest.pt, so resubmitting after a timeout continues.
python3 dreamer.py \
    --configs "$CONFIG" \
    --task "metaworld_${TASK}" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --logdir "$LOGDIR"

echo "finished $(date)"

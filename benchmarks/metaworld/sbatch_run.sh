#!/bin/bash
#SBATCH --job-name=mw
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/gsubbara/AIME/slurm_logs/%x-%j.out
#SBATCH --error=/home/gsubbara/AIME/slurm_logs/%x-%j.err

# Meta-World runner. mkdir -p /home/gsubbara/AIME/slurm_logs before the first
# submit -- SLURM does not create the --output directory and the job dies
# silently if it is missing.
#
#   sbatch --job-name=mw-shs-do benchmarks/metaworld/sbatch_run.sh \
#          metaworld_proprio_shs door-open 0 500000 fixedgoal --mw_randomize_goal False
#
# Args: <config> <task> <seed> <steps> [tag] [extra dreamer flags...]
#
# Timing at MEASURED throughput (H100 3g.40gb MIG, proprio):
#   baseline 13.3 env steps/s -> 500k in ~10.4h
#   SHS       7.7 env steps/s -> 500k in ~18h
# 24h covers both; raise --time for 1M-step tasks on the SHS arm.

set -euo pipefail

CONFIG=${1:?usage: sbatch_run.sh <config> <task> <seed> <steps> [tag] [extra flags...]}
TASK=${2:?missing task}
SEED=${3:?missing seed}
STEPS=${4:?missing steps}
# Optional 5th arg: a tag appended to the logdir, so a variant run (e.g. fixed
# goals) gets its OWN replay buffer and checkpoint instead of resuming into the
# previous run's data. Anything after it is passed straight to dreamer.py.
TAG=${5:-}
shift $(( $# < 5 ? $# : 5 ))
EXTRA=("$@")

REPO=/home/gsubbara/AIME
# Layout matches what benchmarks/eval/aggregate.py expects:
#   <logdir>/<arm>/<task>/seed<k>/
ARM=${CONFIG#metaworld_proprio_}
[ "$ARM" = "metaworld_proprio" ] && ARM=baseline
[ -n "$TAG" ] && ARM="${ARM}_${TAG}"
LOGDIR=./logdir/${ARM}/${TASK}/seed${SEED}

cd "$REPO"

module load opencv mujoco/3.3.0
source aime/bin/activate

# Proprio configs set mw_render: false, so nothing renders and no GL backend is
# needed. dreamer.py uses setdefault for MUJOCO_GL, so this is respected. Switch
# to osmesa or egl if you run a *_vision config.
export MUJOCO_GL=disable
# Env sim needs ~3% of one core at realistic throughput; cap torch so it does
# not oversubscribe the 4 allocated CPUs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "job ${SLURM_JOB_ID:-interactive} | $CONFIG | $TASK | seed $SEED | $STEPS steps"
echo "logdir $LOGDIR | host $(hostname) | started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail fast on a bad node instead of dying deeper inside model construction.
python3 -c "
import socket, sys, torch
if not torch.cuda.is_available():
    print('FATAL: CUDA unavailable on', socket.gethostname(), flush=True); sys.exit(1)
print('CUDA OK:', torch.cuda.get_device_name(0), flush=True)
" || { echo "GPU check failed on $(hostname), exiting"; exit 1; }

# Auto-resumes from $LOGDIR/latest.pt (saved every eval_every block), so
# resubmitting after a timeout continues rather than restarting. Do NOT delete
# the logdir between submissions unless you intend a fresh run.
# -u is required: python block-buffers stdout when it is a file, so a job killed
# at the SLURM time limit loses whatever is still in the buffer.
python3 -u dreamer.py \
    --configs "$CONFIG" \
    --task "metaworld_${TASK}" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --logdir "$LOGDIR" \
    "${EXTRA[@]}"

echo "finished $(date)"

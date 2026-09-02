#!/bin/bash
#SBATCH --job-name=CHEETAH
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=120G
#SBATCH --time=01-23:59:59
#SBATCH --account=aip-irina
#SBATCH --array=1-4                        # one submission -> seed 1 and seed 2
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamerv3-shs-rssm-cheetah_%A_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamerv3-shs-rssm-cheetah_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-1}}"

STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
SHS_Q_RANK="${SHS_Q_RANK:-4}"
SHS_ACTION_DIM="${SHS_ACTION_DIM:-6}"
FLAGS="${FLAGS:-}"

LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/dmc_cheetah_shs_seed_${SEED}}"


module --force purge || true
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a
module load opencv/4.10.0
module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow

# A python venv's activate script is `set -u` clean (it guards $PS1), unlike
# conda's, so strictness can stay on across it.
source "${VENV_DIR}/bin/activate"

unset CUDA_LAUNCH_BLOCKING
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Pin EGL to the allocated GPU for the egl attempt; the osmesa fallback
# clears these again.
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
cd "${PROJECT_DIR}"

echo "Date:      $(date)"
echo "Hostname:  $(hostname)"
echo "Array:     job=${SLURM_ARRAY_JOB_ID:-NA} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Cheetah:   seed=${SEED} steps=${STEPS} batch=${BATCH_SIZE}x${BATCH_LENGTH}"
echo "           q_rank=${SHS_Q_RANK} action_dim=${SHS_ACTION_DIM}"
echo "           logdir=${LOGDIR}"
echo "           FLAGS='${FLAGS}'"
echo "python:    $(command -v python)"
echo "EGL:       MUJOCO_GL=${MUJOCO_GL} device=${MUJOCO_EGL_DEVICE_ID:-unset}"

nvidia-smi || echo "nvidia-smi unavailable"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

unset DISPLAY
render_ok=0
if python - <<'PY'
from dm_control import suite
suite.load("cheetah", "run").physics.render(64, 64)
print("[gl] egl render OK")
PY
then render_ok=1; fi
if [ "${render_ok}" != "1" ]; then
  echo "[gl] egl failed on $(hostname); falling back to osmesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  python - <<'PY' || { echo "[gl] osmesa render also failed on $(hostname)" >&2; exit 3; }
from dm_control import suite
suite.load("cheetah", "run").physics.render(64, 64)
print("[gl] osmesa render OK")
PY
fi

python -u dreamer.py --configs dmc_vision dmc_cheetah_shs \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --shs_q_rank "${SHS_Q_RANK}" \
  --shs_action_dim "${SHS_ACTION_DIM}" \
  --logdir "${LOGDIR}" \
  ${FLAGS}
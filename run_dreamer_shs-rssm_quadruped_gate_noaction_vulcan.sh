#!/bin/bash
#SBATCH --job-name=QUAD-GATE
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01-23:59:59
#SBATCH --account=aip-irina
#SBATCH --array=0-15
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamerv3-shs-rssm-quad-gate_%A_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamerv3-shs-rssm-quad-gate_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT


set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX > 15 )); then echo "array index ${IDX} out of range 0-15" >&2; exit 2; fi
TASKS=(walk run)
TASK="${TASKS[$(( IDX / 8 ))]}"
if (( (IDX / 4) % 2 == 0 )); then USE_ACTION=True; ARM=gateq4_act; else USE_ACTION=False; ARM=gateq4_noact; fi
SEED="${SEED:-$(( IDX % 4 + 1 ))}"
STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

SHS_K=24
SHS_ACTION_DIM=12
SHS_Q_RANK=4
SHS_CALIBRATE_SF=0.1
case "${TASK}" in
  walk) SHS_PROJ_DIM=128; SHS_B0=0.2; SHS_EMA_TAU=0.005  ;;
  run)  SHS_PROJ_DIM=128; SHS_B0=0.1; SHS_EMA_TAU=0.005 ;;
esac

BLOCK="dmc_quadruped_${TASK}_shs_gate"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/dmc_quadruped_${TASK}_shs_${ARM}_seed_${SEED}}"


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

_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -c 0

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
cd "${PROJECT_DIR}"

if ! grep -q "^${BLOCK}:" configs.yaml; then
  echo "configs.yaml has no ${BLOCK} block" >&2; exit 2
fi

scontrol update JobId="${SLURM_JOB_ID}" JobName="q${TASK}_${ARM}_s${SEED}" 2>/dev/null || true

echo "Date:      $(date)"
echo "Hostname:  $(hostname)"
echo "Array:     job=${SLURM_ARRAY_JOB_ID:-NA} task=${IDX}"
echo "Quadruped: task=${TASK} arm=${ARM} rstick_use_action=${USE_ACTION} seed=${SEED} steps=${STEPS}"
echo "SHS:       K=${SHS_K} q_rank=${SHS_Q_RANK} proj_dim=${SHS_PROJ_DIM} calibrate_sF=${SHS_CALIBRATE_SF} b0=${SHS_B0} ema_tau=${SHS_EMA_TAU}"
echo "           configs='dmc_vision ${BLOCK}' logdir=${LOGDIR}"
echo "           FLAGS='${FLAGS}'"
echo "python:    $(command -v python)"

nvidia-smi || echo "nvidia-smi unavailable"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

unset DISPLAY
render_ok=0
if python - <<'PY'
from dm_control import suite
suite.load("quadruped", "walk").physics.render(64, 64, camera_id=2)
print("[gl] egl render OK")
PY
then render_ok=1; fi
if [ "${render_ok}" != "1" ]; then
  echo "[gl] egl failed on $(hostname); falling back to osmesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  python - <<'PY' || { echo "[gl] osmesa render also failed on $(hostname)" >&2; exit 3; }
from dm_control import suite
suite.load("quadruped", "walk").physics.render(64, 64, camera_id=2)
print("[gl] osmesa render OK")
PY
fi

python -u dreamer.py --configs dmc_vision "${BLOCK}" \
  --seed "${SEED}" \
  --shs_K "${SHS_K}" \
  --shs_action_dim "${SHS_ACTION_DIM}" \
  --shs_q_rank "${SHS_Q_RANK}" \
  --shs_proj_dim "${SHS_PROJ_DIM}" \
  --shs_calibrate_sF "${SHS_CALIBRATE_SF}" \
  --shs_b0 "${SHS_B0}" \
  --shs_ema_tau "${SHS_EMA_TAU}" \
  --shs_rstick_use_action "${USE_ACTION}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --logdir "${LOGDIR}" \
  ${FLAGS}

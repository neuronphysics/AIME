#!/bin/bash
#SBATCH --job-name=shs-dmc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=48gb
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=1-36%8
#SBATCH --output=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_dmc_%A_task%a.out
#SBATCH --error=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_dmc_%A_task%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"

TASKS=(walker cheetah hopper pendulum acrobot finger_turn_hard reacher_hard quadruped_run quadruped_walk)
ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( ID < 1 || ID > 36 )); then
  echo "Array index ${ID} is outside the supported range 1-36" >&2
  exit 2
fi
NAME="${NAME:-${TASKS[$(( (ID - 1) / 4 ))]}}"
SEED="${SEED:-$(( (ID - 1) % 4 + 1 ))}"

ARM="${ARM:-fixedk}"
case "${ARM}" in
  moves)         EXTRA_CONFIGS="";            ARM_FLAGS="--shs_structure_moves True" ;;
  fixedk)        EXTRA_CONFIGS="";            ARM_FLAGS="--shs_structure_moves False" ;;
  fixedk_st)     EXTRA_CONFIGS="shs_imag_st"; ARM_FLAGS="--shs_structure_moves False" ;;
  moves_st)      EXTRA_CONFIGS="shs_imag_st"; ARM_FLAGS="--shs_structure_moves True" ;;
  fixedk_nogate) EXTRA_CONFIGS="shs_fixed_k"; ARM_FLAGS="" ;;
  *) echo "ARM must be moves, fixedk, fixedk_st, moves_st or fixedk_nogate, got ${ARM}" >&2; exit 2 ;;
esac
PRESET="dmc_${NAME}_shs_moves"
CONFIGS="dmc_vision ${PRESET} ${EXTRA_CONFIGS}"

STEPS="${STEPS:-}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/shs_${ARM}/${NAME}/seed_${SEED}}"
RUNLOG="${PROJECT_DIR}/logs/shs_${ARM}_${NAME}_seed${SEED}"

mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "${RUNLOG}.out") 2> >(tee -a "${RUNLOG}.err" >&2)

set +u
module unload python 2>/dev/null || true
module load anaconda/3
conda activate "${CONDA_ENV}"
module load gcc/9.3.0
module unload anaconda
module load python/3.10
set -u
unset CUDA_LAUNCH_BLOCKING

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"

cd "${PROJECT_DIR}"

scontrol update JobId="${SLURM_JOB_ID}" JobName="shs_${ARM}_${NAME}_s${SEED}" 2>/dev/null || true

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Slurm:      job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Run:        arm=${ARM} task=${NAME} seed=${SEED}"
echo "Configs:    ${CONFIGS} ${ARM_FLAGS}"
echo "Logdir:     ${LOGDIR}"
echo "Python:     $(command -v python)"
nvidia-smi
if [[ -f "${LOGDIR}/latest.pt" ]]; then
  echo "Existing checkpoint found: dreamer.py will resume from ${LOGDIR}/latest.pt"
fi

TASK_KEY="$(python - "${PRESET}" <<'PY'
import sys
from ruamel.yaml import YAML
print(YAML(typ="safe", pure=True).load(open("configs.yaml"))[sys.argv[1]]["task"])
PY
)"
_DOMAIN_TASK="${TASK_KEY#dmc_}"
export PRE_DOMAIN="${_DOMAIN_TASK%%_*}"
export PRE_TASK="${_DOMAIN_TASK#*_}"
gpu_and_render_preflight() {
python - <<'PY'
import os
import torch
from dm_control import suite

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
suite.load(os.environ["PRE_DOMAIN"], os.environ["PRE_TASK"]).physics.render(64, 64)
print("[gl] render OK via", os.environ.get("MUJOCO_GL"), "for", os.environ["PRE_DOMAIN"], os.environ["PRE_TASK"])
PY
}

if ! gpu_and_render_preflight; then
  echo "[gl] EGL failed on $(hostname); trying OSMesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=2
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  gpu_and_render_preflight
fi

read -r -a CONFIG_ARGS <<< "${CONFIGS}"
EXTRA_ARGS=()
if [[ -n "${ARM_FLAGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${ARM_FLAGS}"
fi
if [[ -n "${STEPS}" ]]; then
  EXTRA_ARGS+=(--steps "${STEPS}")
fi
if [[ -n "${FLAGS}" ]]; then
  read -r -a FLAG_ARGS <<< "${FLAGS}"
  EXTRA_ARGS+=("${FLAG_ARGS[@]}")
fi

python -u dreamer.py --configs "${CONFIG_ARGS[@]}" \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" \
  --batch_length "${BATCH_LENGTH}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODES}" \
  --logdir "${LOGDIR}" \
  "${EXTRA_ARGS[@]}"

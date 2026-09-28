#!/bin/bash
#SBATCH --job-name=walker-transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=48gb
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --array=1-16%4
#SBATCH --output=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/walker_transfer_%A_task%a.out
#SBATCH --error=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/walker_transfer_%A_task%a.err



set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"

MODES=(full world_model scratch scratch_moves)
ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( ID < 1 || ID > 16 )); then
  echo "Array index ${ID} is outside the supported range 1-16" >&2
  exit 2
fi
SEED="${SEED:-$(( (ID - 1) % 4 + 1 ))}"
MODE="${MODE:-${MODES[$(( (ID - 1) / 4 ))]}}"

STEPS="${STEPS:-200000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"
MOVES="${MOVES:-0}"
TAG="${TAG:-}"
if [[ -z "${TAG}" && "${MOVES}" == 1 && "${MODE}" != scratch* ]]; then
  TAG=moves
fi

case "${MODE}" in
  full|world_model)      CONFIGS="${TRANSFER_CONFIGS:-dmc_vision dmc_walker_run_shs_transfer_moves}" ;;
  scratch|scratch_moves) CONFIGS="${SCRATCH_CONFIGS:-dmc_vision dmc_walker_run_shs_scratch_moves}" ;;
  *) echo "Unknown MODE ${MODE}" >&2; exit 2 ;;
esac

FORGET="${FORGET:-fixed}"
case "${FORGET}" in
  fixed)  FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed) ;;
  gated)  FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
  retain) FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
  slow)   FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed --shs_ema_tau 0.005) ;;
  fast)   FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed --shs_ema_tau 0.08) ;;
  *) echo "FORGET must be fixed, gated, retain, slow or fast, got ${FORGET}" >&2; exit 2 ;;
esac
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}/logdir}"
SOURCE_ARM="${SOURCE_ARM:-${FORGET}}"
SOURCE_CKPT="${SOURCE_CKPT:-${SOURCE_DIR}/dmc_walker_walk_shs_${SOURCE_ARM}_seed_${SEED}/latest.pt}"
SRC_RUN="$(dirname "${SOURCE_CKPT}")"
case "$(basename "${SRC_RUN}")" in
  seed_*) RECIPE="${RECIPE:-$(basename "$(dirname "$(dirname "${SRC_RUN}")")")_$(basename "$(dirname "${SRC_RUN}")")}" ;;
  *)      RECIPE="${RECIPE:-$(basename "${SRC_RUN}" | sed 's/_seed_[0-9]*$//')}" ;;
esac
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/transfer_walker_run/dreamerv3_shs-rssm_${MODE}_${FORGET}${TAG:+_${TAG}}_from_${RECIPE}_seed_${SEED}}"
RUNLOG="${PROJECT_DIR}/logs/walker_transfer_${MODE}_${FORGET}${TAG:+_${TAG}}_from_${RECIPE}_seed${SEED}"

mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "${RUNLOG}.out") 2> >(tee -a "${RUNLOG}.err" >&2)
echo "Logs: ${RUNLOG}.out / ${RUNLOG}.err  (Slurm task file: walker_transfer_${SLURM_ARRAY_JOB_ID:-NA}_task${ID}.{out,err})"

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
export TORCH_LINALG_PREFER_CUSOLVER=1
export TORCH_DISABLE_ADDR2LINE=1

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"

cd "${PROJECT_DIR}"

scontrol update JobId="${SLURM_JOB_ID}" JobName="walk2run_${MODE}_${FORGET}${TAG:+_${TAG}}_s${SEED}" 2>/dev/null || true

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Slurm:      job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Resources:  cpus=${SLURM_CPUS_PER_TASK:-NA} gpu=${CUDA_VISIBLE_DEVICES:-NA}"
echo "Transfer:   mode=${MODE} forgetting=${FORGET} seed=${SEED} steps=${STEPS} moves=${MOVES} tag=${TAG:-none} flags=${FLAGS:-none}"
echo "Configs:    ${CONFIGS}"
echo "Batch:      ${BATCH_SIZE}x${BATCH_LENGTH}"
echo "Source:     ${SOURCE_CKPT}"
echo "Target:     ${LOGDIR}"
echo "Python:     $(command -v python)"
nvidia-smi

if [[ ! -f "${SOURCE_CKPT}" && ! -f "${LOGDIR}/latest.pt" ]]; then
  echo "Missing walker-walk checkpoint: ${SOURCE_CKPT}" >&2
  echo "Set SOURCE_DIR or SOURCE_CKPT when submitting if it is stored elsewhere." >&2
  exit 2
fi

gpu_and_render_preflight() {
python - <<'PY'
import os
import torch
from dm_control import suite

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
suite.load("walker", "run").physics.render(64, 64)
print("[gl] render OK via", os.environ.get("MUJOCO_GL"))
PY
}

if ! gpu_and_render_preflight; then
  echo "[gl] EGL failed on $(hostname); trying OSMesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=2
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  gpu_and_render_preflight
fi

if [[ -f "${LOGDIR}/latest.pt" ]]; then
  echo "Resuming ${LOGDIR}/latest.pt"
  ARM_ARGS=(--resume)
elif [[ "${MODE}" == scratch* ]]; then
  ARM_ARGS=(--transfer-mode scratch --setup-from "${SOURCE_CKPT}")
else
  ARM_ARGS=(--transfer-mode "${MODE}" --checkpoint "${SOURCE_CKPT}")
fi
if [[ ! -f "${LOGDIR}/latest.pt" ]]; then
  if [[ "${MODE}" == scratch_moves || ( "${MOVES}" == 1 && "${MODE}" != scratch ) ]]; then
    ARM_ARGS+=(--shs_structure_moves True)
  else
    ARM_ARGS+=(--shs_structure_moves False)
  fi
  ARM_ARGS+=("${FORGET_FLAGS[@]}")
fi

read -r -a CONFIG_ARGS <<< "${CONFIGS}"
EXTRA_ARGS=()
if [[ -n "${FLAGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${FLAGS}"
fi

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
CHAINED=0
chain_next_slice() {
  local name reply
  if [[ "${CHAINED}" -ne 0 || -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  name="walk2run_${MODE}_${FORGET}${TAG:+_${TAG}}_s${SEED}"
  if reply="$(sbatch --job-name="${name}" --dependency="afterany:${SLURM_JOB_ID},singleton" \
      --array="${ID}" "${SCRIPT}" 2>&1)"; then
    CHAINED=1
    echo "[slice] next slice of task ${ID} queued after job ${SLURM_JOB_ID} (singleton ${name}): ${reply}"
    return 0
  fi
  echo "[slice] sbatch of the next slice failed: ${reply}" >&2
  return 1
}

python -u transfer_walker.py --configs "${CONFIG_ARGS[@]}" \
  --task dmc_walker_run \
  --source-task dmc_walker_walk \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" \
  --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODES}" \
  --logdir "${LOGDIR}" \
  "${ARM_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &
py=$!

on_usr1() {
  echo "[slice] time limit in 10 min: asking the runner to checkpoint and exit"
  kill -USR1 "${py}" 2>/dev/null || true
  chain_next_slice || true
}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  trap on_usr1 USR1
  trap 'kill -TERM "${py}" 2>/dev/null || true' TERM
fi

set +e
wait "${py}"; status=$?
while (( status > 128 )); do
  wait "${py}"; s=$?
  (( s == 127 )) && break
  status=${s}
done
set -e
trap - USR1 TERM

case "${status}" in
  0)  echo "[slice] run complete" ;;
  75) chain_next_slice || true
      if (( CHAINED )); then
        echo "[slice] checkpointed; the next slice resumes from ${LOGDIR}/latest.pt"
        status=0
      else
        echo "[slice] latest.pt is saved but no successor job is queued; resubmit: sbatch --array=${ID} run_dreamer_shs-rssm_walker_transfer_mila.sh" >&2
      fi ;;
  *)  echo "[slice] transfer_walker.py failed with status ${status}; not chaining." >&2
      echo "[slice] Fix the error, then resubmit: sbatch --array=${ID} run_dreamer_shs-rssm_walker_transfer_mila.sh" >&2 ;;
esac
exit "${status}"

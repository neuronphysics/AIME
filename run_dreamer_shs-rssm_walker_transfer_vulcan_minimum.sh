#!/bin/bash
#SBATCH --job-name=WALKER-SHS-TRANSFER
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=14:00:00
#SBATCH --account=aip-irina
#SBATCH --array=1-12
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamer_shs_walker_transfer_%A_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamer_shs_walker_transfer_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

MODES=(full world_model scratch)
ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED="${SEED:-$(( (ID - 1) % 4 + 1 ))}"
MODE="${MODE:-${MODES[$(( (ID - 1) / 4 ))]}}"

STEPS="${STEPS:-200000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

TRANSFER_CONFIGS="${TRANSFER_CONFIGS:-dmc_vision dmc_walker_run_shs_transfer_moves}"
SCRATCH_CONFIGS="${SCRATCH_CONFIGS:-dmc_vision dmc_walker_run_shs_scratch_moves}"
SOURCE_RECIPES="${SOURCE_RECIPES:-dmc_vision,dmc_walker_shs,shs_walker_recipe dmc_vision,dmc_walker_shs_moves,shs_small dmc_vision,dmc_walker_shs,walker_walk_launch dmc_vision,dmc_walker_shs,walker_walk_launch_minimum}"

FORGET="${FORGET:-fixed}"
case "${FORGET}" in
  fixed)  FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed) ;;
  gated)  FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
  retain) FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
  slow)   FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed --shs_ema_tau 0.005) ;;
  fast)   FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed --shs_ema_tau 0.08) ;;
  *) echo "FORGET must be fixed, gated, retain, slow or fast, got ${FORGET}" >&2; exit 2 ;;
esac
SOURCE_ARM="${SOURCE_ARM:-${FORGET}}"
SOURCE_CKPT="${SOURCE_CKPT:-${PROJECT_DIR}/logdir/dmc_walker_walk_shs_${SOURCE_ARM}_seed_${SEED}/latest.pt}"
MOVES="${MOVES:-0}"
case "${MOVES}" in
  1) TARGET_ARM=_moves; MOVE_FLAG=True ;;
  0) TARGET_ARM="";     MOVE_FLAG=False ;;
  *) echo "MOVES must be 0 or 1, got ${MOVES}" >&2; exit 2 ;;
esac
SRC_RUN="$(dirname "${SOURCE_CKPT}")"
case "$(basename "${SRC_RUN}")" in
  seed_*) RECIPE="${RECIPE:-$(basename "$(dirname "$(dirname "${SRC_RUN}")")")_$(basename "$(dirname "${SRC_RUN}")")}" ;;
  *)      RECIPE="${RECIPE:-$(basename "${SRC_RUN}" | sed 's/_seed_[0-9]*$//')}" ;;
esac
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/transfer_walker_run/aime_${MODE}_${FORGET}${TARGET_ARM}_from_${RECIPE}_seed_${SEED}}"

if [[ "${MODE}" == "scratch" ]]; then
  CONFIGS="${SCRATCH_CONFIGS}"
else
  CONFIGS="${TRANSFER_CONFIGS}"
fi

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

echo "Date: $(date)  Host: $(hostname)  Array: ${SLURM_ARRAY_JOB_ID:-NA}/${SLURM_ARRAY_TASK_ID:-NA}"
echo "Transfer: mode=${MODE}  forgetting=${FORGET}  seed=${SEED}  steps=${STEPS}  batch=${BATCH_SIZE}x${BATCH_LENGTH}"
echo "Configs:  ${CONFIGS}"
echo "Source:   ${SOURCE_CKPT}"
echo "Recipes:  ${SOURCE_RECIPES}"
echo "Target:   ${LOGDIR}"
echo "FLAGS:    '${FLAGS}'"
nvidia-smi || echo "nvidia-smi unavailable"

if [[ "${MODE}" != "scratch" && ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 2
fi

unset DISPLAY
render_ok=0
if python - <<'PY'
from dm_control import suite
suite.load("walker", "run").physics.render(64, 64)
print("[gl] egl render OK")
PY
then render_ok=1; fi
if [ "${render_ok}" != "1" ]; then
  echo "[gl] egl failed on $(hostname); falling back to osmesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  python - <<'PY' || { echo "[gl] osmesa render also failed on $(hostname)" >&2; exit 3; }
from dm_control import suite
suite.load("walker", "run").physics.render(64, 64)
print("[gl] osmesa render OK")
PY
fi

if [[ -f "${LOGDIR}/latest.pt" ]]; then
  echo "Resuming target run from ${LOGDIR}/latest.pt"
  ARM_ARGS=(--resume --transfer-mode "${MODE}")
elif [[ "${MODE}" == "scratch" ]]; then
  ARM_ARGS=(--transfer-mode scratch)
  if [[ -f "${SOURCE_CKPT}" ]]; then
    ARM_ARGS+=(--setup-from "${SOURCE_CKPT}")
  else
    echo "Note: ${SOURCE_CKPT} not found; scratch uses the preset architecture." >&2
  fi
else
  ARM_ARGS=(--transfer-mode "${MODE}" --checkpoint "${SOURCE_CKPT}")
fi
if [[ ! -f "${LOGDIR}/latest.pt" ]]; then
  ARM_ARGS+=(--shs_structure_moves "${MOVE_FLAG}" "${FORGET_FLAGS[@]}")
fi

python -u transfer_walker.py --configs ${CONFIGS} \
  --task dmc_walker_run \
  --source-task dmc_walker_walk \
  --source-recipes ${SOURCE_RECIPES} \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODES}" \
  --logdir "${LOGDIR}" \
  "${ARM_ARGS[@]}" \
  ${FLAGS} &
py=$!
trap 'kill -TERM "${py}" 2>/dev/null || true' TERM
set +e
wait "${py}"; status=$?
while (( status > 128 )); do
  wait "${py}"; s=$?
  (( s == 127 )) && break
  status=${s}
done
set -e
exit "${status}"

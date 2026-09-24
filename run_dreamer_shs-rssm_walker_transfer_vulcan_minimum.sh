#!/bin/bash
#SBATCH --job-name=WALKER-SHS-TRANSFER
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=14:00:00                    # 200k run steps at AIME throughput + 20 evals
#SBATCH --account=aip-irina
#SBATCH --array=1-12                       # 3 transfer modes x 4 seeds (mapping below)
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamer_shs_walker_transfer_%A_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamer_shs_walker_transfer_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

# ---- which arm and which seed does this array task run? -------------------
# array id 1..4  -> full         seeds 1..4   (world model + actor/critic from walk)
# array id 5..8  -> world_model  seeds 1..4   (world model only; fresh reward head, actor, critic)
# array id 9..12 -> scratch      seeds 1..4   (no checkpoint; reference)
MODES=(full world_model scratch)
ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED="${SEED:-$(( (ID - 1) % 4 + 1 ))}"
MODE="${MODE:-${MODES[$(( (ID - 1) / 4 ))]}}"

STEPS="${STEPS:-200000}"                   # target-task (walker run) env steps
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"                         # tau ablation: FLAGS="--fixed-regimes --shs_ema_tau 0.1"

# Source checkpoints: the dmc_walker_shs_v2 seeds. The shs_* overrides below are
# the ones that produced them and must not change (K is re-read from the
# checkpoint by the runner; proj_dim/q_rank are checked at load; b0,
# calibrate_sF and ema_tau are not checked, so they are repeated verbatim).
SOURCE_CKPT="${SOURCE_CKPT:-${PROJECT_DIR}/logdir/dmc_walker_shs_v2_seed_${SEED}/latest.pt}"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/transfer_walker_run/aime_${MODE}_seed_${SEED}}"

# ---- environment ----------------------------------------------------------
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

mkdir -p "${PROJECT_DIR}/logs"             # LOGDIR is created by the runner and must be empty
cd "${PROJECT_DIR}"

echo "Date: $(date)  Host: $(hostname)  Array: ${SLURM_ARRAY_JOB_ID:-NA}/${SLURM_ARRAY_TASK_ID:-NA}"
echo "Transfer: mode=${MODE}  seed=${SEED}  steps=${STEPS}  batch=${BATCH_SIZE}x${BATCH_LENGTH}"
echo "Source:   ${SOURCE_CKPT}"
echo "Target:   ${LOGDIR}"
echo "FLAGS:    '${FLAGS}'"
nvidia-smi || echo "nvidia-smi unavailable"

# ---- source checkpoint must exist for the transfer arms -------------------
if [[ "${MODE}" != "scratch" && ! -f "${SOURCE_CKPT}" ]]; then
  echo "Missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 2
fi

# ---- GL preflight: EGL, then OSMesa ---------------------------------------
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

# ---- fresh start or resume of a preempted target run ----------------------
# A requeued task finds its own latest.pt and continues with --resume instead
# of re-loading the walk checkpoint (the runner refuses a non-empty target
# directory otherwise).
if [[ -f "${LOGDIR}/latest.pt" ]]; then
  echo "Resuming target run from ${LOGDIR}/latest.pt"
  ARM_ARGS=(--resume)
elif [[ "${MODE}" == "scratch" ]]; then
  ARM_ARGS=(--transfer-mode scratch)
else
  ARM_ARGS=(--transfer-mode "${MODE}" --checkpoint "${SOURCE_CKPT}")
fi

# ---- run ------------------------------------------------------------------
python -u transfer_walker.py --configs dmc_vision dmc_walker_shs \
  --task dmc_walker_run \
  --source-task dmc_walker_walk \
  --shs_action_dim 6 \
  --shs_proj_dim 256 \
  --shs_b0 0.2 --shs_calibrate_sF 0.1 \
  --shs_K 24 --shs_q_rank 8 --shs_ema_tau 0.01 \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODES}" \
  --logdir "${LOGDIR}" \
  "${ARM_ARGS[@]}" \
  ${FLAGS}
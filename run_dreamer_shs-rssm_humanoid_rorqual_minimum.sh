#!/bin/bash
#SBATCH --job-name=SHS-RSSM
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --time=2-23:59:59
#SBATCH --account=def-bengioy
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-humanoid_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-humanoid_%N-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT


set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/links/projects/def-irina/memole/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/links/D2E}"
SEED="${SEED:-1}"
STEPS="${STEPS:-3000000}"
BATCH_SIZE="${BATCH_SIZE:-96}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
PRECISION="${PRECISION:-32}"
TRAIN_RATIO="${TRAIN_RATIO:-512}"
SHS_Q_RANK="${SHS_Q_RANK:-4}"
SHS_ACTION_DIM="${SHS_ACTION_DIM:-21}"
SHS_LEARN_B0="${SHS_LEARN_B0:-True}"
SHS_B0_STRENGTH="${SHS_B0_STRENGTH:-2.0}"
SHS_CONSOLIDATE_EVERY="${SHS_CONSOLIDATE_EVERY:-50}"
SHS_RSTICK_STOPGRAD="${SHS_RSTICK_STOPGRAD:-True}"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/dmc_humanoid_shs/seed${SEED}}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-600}"

module --force purge
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
export TORCH_LINALG_PREFER_CUSOLVER=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
cd "${PROJECT_DIR}"

python -X faulthandler - <<'PY'
import pathlib
import torch
import shs_rssm

print("SHS package:", pathlib.Path(shs_rssm.__file__).resolve(), flush=True)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available -- check the module stack and the venv "
                       "on this node")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
x = torch.randn(1024, 1024, device="cuda")
print("GPU matmul:", float((x @ x).mean()), flush=True)
PY

heartbeat() {
  while true; do
    sleep "${HEARTBEAT_SECONDS}"
    echo "[heartbeat] $(date --iso-8601=seconds) logdir=${LOGDIR}" >&2
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits >&2 || true
  done
}
heartbeat &
HEARTBEAT_PID=$!
trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true' EXIT INT TERM

echo "Starting Humanoid: seed=${SEED} steps=${STEPS} batch=${BATCH_SIZE}x${BATCH_LENGTH} precision=${PRECISION} train_ratio=${TRAIN_RATIO} q_rank=${SHS_Q_RANK} action_dim=${SHS_ACTION_DIM} learn_b0=${SHS_LEARN_B0} consolidate_every=${SHS_CONSOLIDATE_EVERY} rstick_stopgrad=${SHS_RSTICK_STOPGRAD} logdir=${LOGDIR}"

python -u dreamer.py \
  --configs dmc_humanoid_shs \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --compile False \
  --precision "${PRECISION}" \
  --train_ratio "${TRAIN_RATIO}" \
  --shs_shared_carry True \
  --shs_q_rank "${SHS_Q_RANK}" \
  --shs_action_dim "${SHS_ACTION_DIM}" \
  --shs_learn_b0 "${SHS_LEARN_B0}" \
  --shs_b0_strength "${SHS_B0_STRENGTH}" \
  --shs_consolidate_every_episodes "${SHS_CONSOLIDATE_EVERY}" \
  --shs_rstick_stopgrad "${SHS_RSTICK_STOPGRAD}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_length "${BATCH_LENGTH}" \
  --logdir "${LOGDIR}"
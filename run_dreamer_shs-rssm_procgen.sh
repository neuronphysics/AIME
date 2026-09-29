#!/bin/bash
#SBATCH --job-name=PROCGEN-SHS
#SBATCH --account=aip-irina
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/home/memole/projects/aip-irina/memole/logs/procgen_boot_%A_%a.out
#SBATCH --error=/home/memole/projects/aip-irina/memole/logs/procgen_boot_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
LOGROOT="${LOGROOT:-/home/memole/projects/aip-irina/memole/logs}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/logdir}"
STEPS="${STEPS:-1000000}"
FORGET="${FORGET:-retain}"
MAX_SLICES=20
MAX_FAILS=3
GAMES=(coinrun bigfish fruitbot chaser)
NSEEDS=4
cd "${PROJECT_DIR}"

case "${FORGET}" in
  retain) FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
  gated)  FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
  fixed)  FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed) ;;
  *) echo "FORGET must be retain, gated or fixed, got ${FORGET}" >&2; exit 2 ;;
esac

run_settings() {
  local id="$1" extra
  GAME="${GAMES[$(( id / NSEEDS ))]}"
  SEED=$(( id % NSEEDS ))
  RUN="procgen_${GAME}_shs_${FORGET}"
  LOGDIR="${RUN_ROOT}/${RUN}/seed_${SEED}"
  CMD=(python -u dreamer.py --configs "procgen_${GAME}_shs"
       --task "procgen_${GAME}" --seed "${SEED}" --compile False
       --batch_size 16 --batch_length 64 --steps "${STEPS}"
       --logdir "${LOGDIR}" "${FORGET_FLAGS[@]}")
  if [[ -n "${FLAGS:-}" ]]; then
    read -r -a extra <<< "${FLAGS}"
    CMD+=("${extra[@]}")
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  for id in $(seq 0 $(( ${#GAMES[@]} * NSEEDS - 1 ))); do
    run_settings "${id}"
    printf '%2d  %-9s seed %s ' "${id}" "${GAME}" "${SEED}"
    printf ' %q' "${CMD[@]}"; echo
  done
  exit 0
fi

ID="${SLURM_ARRAY_TASK_ID}"
SLICE="${SLICE:-1}"
FAILS="${FAILS:-0}"
run_settings "${ID}"
SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'Command=\K\S+' || true)"
[[ -f "${SCRIPT}" ]] || SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
scontrol update JobId="${SLURM_JOB_ID}" JobName="PG-${GAME}-${FORGET}-s${SEED}" 2>/dev/null || true

mkdir -p "${LOGROOT}" "${LOGDIR}"
echo "[boot] ${GAME} seed ${SEED} ${FORGET} -> ${LOGROOT}/${RUN}_seed${SEED}.{out,err}"
exec >> "${LOGROOT}/${RUN}_seed${SEED}.out" 2>> "${LOGROOT}/${RUN}_seed${SEED}.err"
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID} | index ${ID} | slice ${SLICE} | ${GAME} seed ${SEED} ${FORGET}"
echo "Logdir: ${LOGDIR}"
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo

if [[ -f "${LOGDIR}/.finished" ]]; then
  echo "Run already finished"
  exit 0
fi
LOCK="${LOGDIR}/.slurm_lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
  OWNER="$(cat "${LOCK}/job" 2>/dev/null || true)"
  if [[ -n "${OWNER}" && "${OWNER}" != "${SLURM_JOB_ID}" ]] && squeue -h -j "${OWNER}" -t R 2>/dev/null | grep -q .; then
    echo "Job ${OWNER} is already training ${LOGDIR}" >&2
    exit 0
  fi
  rm -rf "${LOCK}"
  mkdir "${LOCK}"
fi
echo "${SLURM_JOB_ID}" > "${LOCK}/job"
trap 'rm -rf "${LOCK}"' EXIT

set +u
module purge || true
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a
source "${VENV_DIR}/bin/activate"
set -u
python -c "import sys; assert sys.prefix.startswith('${VENV_DIR}'), f'wrong interpreter: {sys.prefix}'"
unset CUDA_LAUNCH_BLOCKING

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
export OMPI_MCA_pmix="^s1,s2,cray,isolated"
export PMIX_MCA_gds=hash
ulimit -c 0

echo "Python: $(command -v python)"
nvidia-smi || true
GAME="${GAME}" SEED="${SEED}" python - <<'PY'
import os
import numpy as np
import torch
from envs.procgen import ProcGen

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print("GPU:", torch.cuda.get_device_name(0))
env = ProcGen(os.environ["GAME"], (64, 64), seed=int(os.environ["SEED"]))
assert env.action_space.n == 15, env.action_space
obs = env.reset()
assert obs["image"].shape == (64, 64, 3) and obs["image"].max() > 0
obs, reward, done, info = env.step(env.action_space.sample())
print("Procgen OK:", os.environ["GAME"], obs["image"].shape, reward, done)
PY

CHAINED=0
queue_next_slice() {
  local fails="$1" reply
  if (( CHAINED )); then return 0; fi
  if (( SLICE + 1 > MAX_SLICES )); then
    echo "MAX_SLICES=${MAX_SLICES} reached for ${LOGDIR}" >&2
    return 1
  fi
  if reply="$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${ID}" \
      --export="ALL,SLICE=$(( SLICE + 1 )),FAILS=${fails}" "${SCRIPT}" 2>&1)"; then
    CHAINED=1
    echo "Slice $(( SLICE + 1 )) queued: ${reply}"
    return 0
  fi
  echo "sbatch of the next slice failed: ${reply}; resubmit with: sbatch --array=${ID} ${SCRIPT}" >&2
  return 1
}

PY_PID=""
trap 'queue_next_slice 0 || true; kill -USR1 "${PY_PID}" 2>/dev/null || true' USR1
trap 'kill -TERM "${PY_PID}" 2>/dev/null || true' TERM

"${CMD[@]}" &
PY_PID=$!
status=0
while :; do
  if wait "${PY_PID}"; then status=0; else status=$?; fi
  kill -0 "${PY_PID}" 2>/dev/null || break
done
if (( status > 128 )); then
  if wait "${PY_PID}" 2>/dev/null; then status=0; else again=$?; (( again == 127 )) || status="${again}"; fi
fi
trap - USR1 TERM

case "${status}" in
  0)
    date > "${LOGDIR}/.finished"
    echo "Finished ${LOGDIR}" ;;
  75)
    if (( CHAINED )); then
      status=0
    else
      echo "latest.pt is saved but no next slice is queued; resubmit with: sbatch --array=${ID} ${SCRIPT}" >&2
    fi ;;
  78)
    echo "${LOGDIR} holds a checkpoint trained with other settings; use another RUN_ROOT" >&2 ;;
  *)
    if [[ -f "${LOGDIR}/latest.pt" ]] && (( FAILS + 1 < MAX_FAILS )); then
      echo "dreamer.py exited with ${status}; retrying from the last checkpoint" >&2
      queue_next_slice $(( FAILS + 1 )) || true
    fi ;;
esac
exit "${status}"
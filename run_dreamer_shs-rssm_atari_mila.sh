#!/bin/bash
#SBATCH --job-name=shs-atari
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint=40gb|48gb|80gb
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --array=1-48
#SBATCH --output=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_atari_boot_%A_%a.out
#SBATCH --error=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_atari_boot_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"
LOGROOT="${LOGROOT:-${PROJECT_DIR}/logs}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/logdir}"
STEPS="${STEPS:-400000}"
EVAL_EVERY="${EVAL_EVERY:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
MOVES="${MOVES:-1}"
MAX_SLICES=20
MAX_FAILS=3
read -r -a GAMES <<< "${GAMES:-pong breakout alien hero demon_attack frostbite}"
read -r -a FORGETS <<< "${FORGETS:-retain gated}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4}"
TOTAL=$(( ${#FORGETS[@]} * ${#GAMES[@]} * ${#SEEDS[@]} ))
cd "${PROJECT_DIR}"

run_settings() {
  local i=$(( $1 - 1 )) extra
  FORGET="${FORGETS[$(( i % ${#FORGETS[@]} ))]}"
  GAME="${GAMES[$(( i / ${#FORGETS[@]} % ${#GAMES[@]} ))]}"
  SEED="${SEEDS[$(( i / (${#FORGETS[@]} * ${#GAMES[@]}) ))]}"
  case "${FORGET}" in
    retain) FLAGS_RUN=(--shs_ema_gate True --shs_forget_mode evidence) ;;
    gated)  FLAGS_RUN=(--shs_ema_gate True --shs_forget_mode fixed) ;;
    fixed)  FLAGS_RUN=(--shs_ema_gate False --shs_forget_mode fixed) ;;
    *) echo "FORGETS entries must be retain, gated or fixed, got ${FORGET}" >&2; exit 2 ;;
  esac
  if [[ "${MOVES}" == 1 ]]; then
    FLAGS_RUN+=(--shs_structure_moves True)
    RUN="${FORGET}"
  else
    FLAGS_RUN+=(--shs_structure_moves False)
    RUN="${FORGET}_fixedk"
  fi
  LOGDIR="${RUN_ROOT}/atari_${RUN}/${GAME}/seed_${SEED}"
  RUNLOG="${LOGROOT}/atari_${RUN}_${GAME}_seed${SEED}"
  CMD=(python -u dreamer.py --configs atari100k "atari_${GAME}_shs_moves" atari100k_schedule
       --seed "${SEED}" --compile False --batch_size 16 --batch_length 64
       --steps "${STEPS}" --eval_every "${EVAL_EVERY}" --eval_episode_num "${EVAL_EPISODES}"
       --logdir "${LOGDIR}" "${FLAGS_RUN[@]}")
  if [[ -n "${FLAGS:-}" ]]; then
    read -r -a extra <<< "${FLAGS}"
    CMD+=("${extra[@]}")
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "${TOTAL} runs; submit with: sbatch --array=1-${TOTAL} run_dreamer_shs-rssm_atari_mila.sh"
  for id in $(seq 1 "${TOTAL}"); do
    run_settings "${id}"
    status="not started"
    [[ -f "${LOGDIR}/latest.pt" ]] && status="checkpoint"
    [[ -f "${LOGDIR}/.finished" ]] && status="finished"
    printf '%2d  %-13s %-13s seed %s  %s\n' "${id}" "${RUN}" "${GAME}" "${SEED}" "${status}"
    printf '     '; printf ' %q' "${CMD[@]}"; echo
  done
  exit 0
fi

ID="${SLURM_ARRAY_TASK_ID}"
SLICE="${SLICE:-1}"
FAILS="${FAILS:-0}"
if (( ID < 1 || ID > TOTAL )); then
  echo "Array index ${ID} is outside 1-${TOTAL}; nothing to run"
  exit 0
fi
run_settings "${ID}"
SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'Command=\K\S+' || true)"
[[ -f "${SCRIPT}" ]] || SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
scontrol update JobId="${SLURM_JOB_ID}" JobName="atari_${RUN}_${GAME}_s${SEED}" 2>/dev/null || true

mkdir -p "${LOGROOT}" "${LOGDIR}"
echo "[boot] ${GAME} seed ${SEED} ${RUN} -> ${RUNLOG}.{out,err}"
exec >> "${RUNLOG}.out" 2>> "${RUNLOG}.err"
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID} | index ${ID} | slice ${SLICE} | ${GAME} seed ${SEED} ${RUN}"
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
module unload python 2>/dev/null || true
module load anaconda/3
conda activate "${CONDA_ENV}"
module load gcc/9.3.0
module unload anaconda
module load python/3.10
set -u
unset CUDA_LAUNCH_BLOCKING

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export TORCH_DISABLE_ADDR2LINE=1
unset DISPLAY

echo "Python: $(command -v python)"
nvidia-smi || true
python - "${GAME}" <<'PY'
import sys

import numpy as np
import torch
import envs.atari as atari

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print("GPU:", torch.cuda.get_device_name(0))
env = atari.Atari(sys.argv[1], 4, (64, 64), gray=False, noops=30, lives="unused",
                  sticky=False, actions="needed", resize="opencv", seed=0)
env.reset()
n = env.action_space.n
obs, reward, done, info = env.step(np.eye(n, dtype=np.float32)[0])
assert obs["image"].shape == (64, 64, 3), obs["image"].shape
print(f"Atari OK: {sys.argv[1]}, {n} actions")
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
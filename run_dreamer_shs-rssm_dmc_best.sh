#!/bin/bash
#SBATCH --job-name=aime-dmc-best
#SBATCH --account=aip-irina
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-59
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/home/memole/scratch/AIME/logs/dmc_best_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dmc_best_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/logdir}"
STEPS="${STEPS:-1000000}"
MOVES="${MOVES:-1}"
MAX_SLICES=10
MAX_FAILS=3
ARMS=(fixed gated retain)
TASKS=(walker finger quadruped_walk quadruped_run reacher)
NSEEDS=4
cd "${PROJECT_DIR}"

run_settings() {
  local id="$1" extra
  ARM="${ARMS[$(( id / (${#TASKS[@]} * NSEEDS) ))]}"
  TASK="${TASKS[$(( id / NSEEDS % ${#TASKS[@]} ))]}"
  SEED=$(( id % NSEEDS + 1 ))
  case "${TASK}" in
    walker)
      TASK_ID=dmc_walker_walk; DOMAIN=walker; DMC_TASK=walk
      PRESETS=(dmc_walker_shs shs_walker_recipe); RECIPE_FLAGS=() ;;
    finger)
      TASK_ID=dmc_finger_turn_hard; DOMAIN=finger; DMC_TASK=turn_hard
      PRESETS=(dmc_finger_turn_hard_shs_gate)
      RECIPE_FLAGS=(--shs_action_dim 2 --shs_K 16 --shs_q_rank 4 --shs_b0 0.2 --shs_calibrate_sF 0.1 --shs_imag_mode actor_moment) ;;
    quadruped_walk)
      TASK_ID=dmc_quadruped_walk; DOMAIN=quadruped; DMC_TASK=walk
      PRESETS=(dmc_quadruped_walk_shs_gate shs_quadruped_recipe); RECIPE_FLAGS=() ;;
    quadruped_run)
      TASK_ID=dmc_quadruped_run; DOMAIN=quadruped; DMC_TASK=run
      PRESETS=(dmc_quadruped_run_shs_gate shs_quadruped_recipe); RECIPE_FLAGS=(--shs_b0 0.1) ;;
    reacher)
      TASK_ID=dmc_reacher_hard; DOMAIN=reacher; DMC_TASK=hard
      PRESETS=(dmc_reacher_hard_shs); RECIPE_FLAGS=(--shs_action_dim 2 --shs_q_rank 4 --shs_imag_mode actor_moment) ;;
  esac
  case "${ARM}" in
    fixed)  ARM_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed) ;;
    gated)  ARM_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
    retain) ARM_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
  esac
  if [[ "${MOVES}" == 1 ]]; then
    PRESETS+=(shs_move_gate)
    ARM_FLAGS+=(--shs_structure_moves True)
    RUN_TAG="${ARM}"
  else
    ARM_FLAGS+=(--shs_structure_moves False)
    RUN_TAG="${ARM}_fixedk"
  fi
  LOGDIR="${RUN_ROOT}/${TASK_ID}_shs_${RUN_TAG}_seed_${SEED}"
  CMD=(python -u dreamer.py --configs dmc_vision "${PRESETS[@]}"
       --task "${TASK_ID}" --seed "${SEED}" --compile False
       --batch_size 16 --batch_length 64
       --steps "${STEPS}" --eval_every 10000 --eval_episode_num 10
       --logdir "${LOGDIR}" "${RECIPE_FLAGS[@]}" "${ARM_FLAGS[@]}")
  if [[ -n "${FLAGS:-}" ]]; then
    read -r -a extra <<< "${FLAGS}"
    CMD+=("${extra[@]}")
  fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  for id in $(seq 0 $(( ${#ARMS[@]} * ${#TASKS[@]} * NSEEDS - 1 ))); do
    run_settings "${id}"
    printf '%2d  %-15s %-7s seed %s ' "${id}" "${TASK}" "${RUN_TAG}" "${SEED}"
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
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID} | index ${ID} | slice ${SLICE} | ${TASK} ${RUN_TAG} seed ${SEED}"
echo "Logdir: ${LOGDIR}"
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo

if [[ -f "${LOGDIR}/.finished" ]]; then
  echo "Run already finished"
  exit 0
fi
mkdir -p "${LOGDIR}"
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
module --force purge || true
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11 scipy-stack/2024a opencv/4.10.0 mujoco/3.3.0 mpi4py/3.1.6 arrow
source "${VENV_DIR}/bin/activate"
set -u
unset CUDA_LAUNCH_BLOCKING
unset DISPLAY

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

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"

preflight() {
  DOMAIN="${DOMAIN}" DMC_TASK="${DMC_TASK}" python - <<'PY'
import os
import sys
import torch
from dm_control import suite

if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the allocated GPU")
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
suite.load(os.environ["DOMAIN"], os.environ["DMC_TASK"]).physics.render(64, 64)
print("Render OK via", os.environ.get("MUJOCO_GL"))
PY
}

echo "Python: $(command -v python)"
nvidia-smi || true
if ! preflight; then
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=2
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  preflight
fi

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
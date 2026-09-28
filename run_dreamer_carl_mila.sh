#!/bin/bash
#SBATCH --job-name=AIME-CARL
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=40gb|48gb|80gb
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --array=1-4
#SBATCH --output=./logs/slurm-dreamer-shs-rssm-carl-%A_%a.out
#SBATCH --error=./logs/slurm-dreamer-shs-rssm-carl-%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${PROJECT_DIR}"
[[ -f dreamer.py && -f configs.yaml ]] || { echo "PROJECT_DIR must point to the AIME checkout" >&2; exit 2; }

CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"
STEPS="${STEPS:-1000000}"
MAX_FAILS=3
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/logdir}"
read -r -a SETUPS <<< "${SETUPS:-shs:gated shs_moves:gated shs:retain shs_moves:retain}"
read -r -a PRESETS <<< "${CARL_PRESETS:-carl_walker_shs carl_walker_gravity_friction_shs carl_quadruped_walk_shs carl_quadruped_actuator_shs carl_finger_spin_shs}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4}"

RUN_LIST=()
for setup in "${SETUPS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for preset in "${PRESETS[@]}"; do
      RUN_LIST+=("${setup}/${preset}/${seed}")
    done
  done
done

run_settings() {
  local setup extra
  IFS=/ read -r setup PRESET SEED <<< "$1"
  IFS=: read -r ARM FORGET <<< "${setup}"
  CONFIGS=("${PRESET}")
  RUN="${PRESET}"
  FORGET_FLAGS=()
  if [[ "${ARM}" == vanilla ]]; then
    CONFIGS+=(carl_vanilla)
    RUN="${PRESET%_shs}_vanilla"
  else
    case "${PRESET}" in
      carl_walker_*)    CONFIGS+=(shs_walker_recipe) ;;
      carl_quadruped_*) CONFIGS+=(shs_quadruped_recipe) ;;
    esac
    case "${ARM}" in
      shs)       ;;
      shs_moves) CONFIGS+=(carl_moves); RUN="${PRESET}_moves" ;;
      *) echo "Unknown arm '${ARM}' (shs, shs_moves, vanilla)" >&2; exit 2 ;;
    esac
    case "${FORGET:-}" in
      gated)  FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
      retain) FORGET_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
      fixed)  FORGET_FLAGS=(--shs_ema_gate False --shs_forget_mode fixed) ;;
      *) echo "Unknown forgetting '${FORGET:-}' (gated, retain, fixed)" >&2; exit 2 ;;
    esac
    RUN="${RUN}_${FORGET}"
  fi
  LOGDIR="${LOG_ROOT}/${RUN}/seed_${SEED}"
  COMMAND=(python -u dreamer.py --configs "${CONFIGS[@]}"
    --seed "${SEED}" --compile False --steps "${STEPS}" --logdir "${LOGDIR}" "${FORGET_FLAGS[@]}")
  if [[ -n "${FLAGS:-}" ]]; then
    read -r -a extra <<< "${FLAGS}"
    COMMAND+=("${extra[@]}")
  fi
}

failure_count() {
  if [[ -f "${LOGDIR}/.failures" ]]; then wc -l < "${LOGDIR}/.failures"; else echo 0; fi
}

lock_owner() {
  cat "${LOGDIR}/.lane_lock/job" 2>/dev/null || true
}

run_is_open() {
  [[ ! -f "${LOGDIR}/.finished" && ! -f "${LOGDIR}/.config_mismatch" ]] && (( $(failure_count) < MAX_FAILS ))
}

run_progress() {
  local last
  if [[ -f "${LOGDIR}/.finished" ]]; then echo "finished"; return; fi
  if [[ -f "${LOGDIR}/.config_mismatch" ]]; then echo "blocked: checkpoint of other settings"; return; fi
  if (( $(failure_count) >= MAX_FAILS )); then echo "blocked: failed ${MAX_FAILS} times"; return; fi
  last="$(tail -n 1 "${LOGDIR}/metrics.jsonl" 2>/dev/null | grep -o '"step": [0-9.]*' | grep -o '[0-9.]*$' || true)"
  if [[ -n "${last}" ]]; then echo "at frame ${last%.*}"; else echo "not started"; fi
}

claim_run() {
  local lock="${LOGDIR}/.lane_lock" owner
  mkdir -p "${LOGDIR}"
  if ! mkdir "${lock}" 2>/dev/null; then
    owner="$(lock_owner)"
    if [[ -z "${owner}" ]]; then sleep 2; owner="$(lock_owner)"; fi
    if [[ "${owner}" != "${SLURM_JOB_ID}" ]]; then
      if [[ -n "${owner}" ]] && squeue -h -j "${owner}" -t RUNNING 2>/dev/null | grep -q .; then return 1; fi
      rm -rf "${lock}"
      mkdir "${lock}" 2>/dev/null || return 1
    fi
  fi
  echo "${SLURM_JOB_ID}" > "${lock}/job"
  sleep 1
  [[ "$(lock_owner)" == "${SLURM_JOB_ID}" ]]
}

release_run() {
  if [[ -n "${LOGDIR:-}" && "$(lock_owner)" == "${SLURM_JOB_ID:-}" ]]; then rm -rf "${LOGDIR}/.lane_lock"; fi
}

pick_next_run() {
  local entry
  for entry in "${RUN_LIST[@]}"; do
    run_settings "${entry}"
    if run_is_open && claim_run; then return 0; fi
  done
  return 1
}

work_remains() {
  local entry
  for entry in "${RUN_LIST[@]}"; do
    run_settings "${entry}"
    if run_is_open; then return 0; fi
  done
  return 1
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "${#RUN_LIST[@]} runs in priority order, logdirs under ${LOG_ROOT}"
  n=0
  for entry in "${RUN_LIST[@]}"; do
    n=$(( n + 1 ))
    run_settings "${entry}"
    printf '%3d  %-18s %-34s seed %s  %s\n' "${n}" "${ARM}${FORGET:+:${FORGET}}" "${PRESET}" "${SEED}" "$(run_progress)"
    printf '      '; printf ' %q' "${COMMAND[@]}"; printf '\n'
  done
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit with: mkdir -p logs && sbatch run_dreamer_carl_mila.sh   (list runs: DRY_RUN=1 bash run_dreamer_carl_mila.sh)" >&2
  exit 2
fi

LANE="${SLURM_ARRAY_TASK_ID:-1}"
SLICE="${SLICE:-1}"
SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'Command=\K\S+' || true)"
[[ -f "${SCRIPT}" ]] || SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID} | lane ${LANE} | slice ${SLICE} | ${#RUN_LIST[@]} runs"

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

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export DISABLE_RENDER_THREAD_OFFLOADING=1
unset DISPLAY

echo "Python: $(command -v python)"
nvidia-smi || true
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('GPU:', torch.cuda.get_device_name(0))"

carl_preflight() {
  python - "${PRESET}" "${SEED}" <<'PY'
import pathlib
import sys

import numpy as np
from ruamel.yaml import YAML
from envs.carl import CARL

preset = YAML(typ="safe", pure=True).load(pathlib.Path("configs.yaml").read_text())[sys.argv[1]]
for split in ("train", "eval"):
    spec = preset["carl"][f"{split}_contexts"]
    env = CARL(preset["task"].removeprefix("carl_"), action_repeat=preset["action_repeat"], size=(64, 64),
               seed=int(sys.argv[2]), contexts=spec, selector="round_robin")
    count = int(np.prod([len(v) for v in spec.values()]))
    seen = set()
    for _ in range(count):
        obs = env.reset()
        seen.add(int(obs["context_id"]))
        assert obs["image"].shape == (64, 64, 3)
        if "gravity" in spec:
            assert np.isclose(-env._env.env.env.physics.model.opt.gravity[2], env.context["gravity"])
        obs, reward, done, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert np.isfinite(reward) and np.isfinite(obs["state"]).all()
    assert len(seen) == count, (seen, count)
    print(f"{split}: rendered and stepped {count} contexts {spec}")
PY
}

queue_next_slice() {
  local reply
  if reply="$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${LANE}" \
      --export="ALL,SLICE=$(( SLICE + 1 ))" "${SCRIPT}" 2>&1)"; then
    echo "[lane ${LANE}] slice $(( SLICE + 1 )) queued: ${reply}"
    return 0
  fi
  echo "[lane ${LANE}] sbatch failed: ${reply}; resubmit with: sbatch --array=${LANE} ${SCRIPT}" >&2
  return 1
}

STOP=""
PY_PID=""
forward_signal() {
  if [[ -n "${PY_PID}" ]]; then kill "-$1" "${PY_PID}" 2>/dev/null || true; fi
}
trap 'STOP=usr1; forward_signal USR1' USR1
trap 'STOP=term; forward_signal TERM' TERM
trap release_run EXIT

while [[ -z "${STOP}" ]]; do
  if ! pick_next_run; then
    echo "[lane ${LANE}] no run left for this lane"
    break
  fi
  echo "=== $(date) | lane ${LANE} | ${ARM} ${FORGET:-} | ${PRESET} | seed ${SEED} | $(run_progress)"
  printf 'Command:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  scontrol update JobId="${SLURM_JOB_ID}" JobName="carl_${RUN}_s${SEED}" 2>/dev/null || true

  if ! carl_preflight; then
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
    unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
    if ! carl_preflight; then
      echo "$(date) preflight failed on $(hostname)" >> "${LOGDIR}/.failures"
      release_run
      continue
    fi
  fi
  if [[ -n "${STOP}" ]]; then release_run; break; fi

  "${COMMAND[@]}" &
  PY_PID=$!
  status=0
  while :; do
    if wait "${PY_PID}"; then status=0; else status=$?; fi
    kill -0 "${PY_PID}" 2>/dev/null || break
  done
  if (( status > 128 )); then
    if wait "${PY_PID}" 2>/dev/null; then status=0; else again=$?; (( again == 127 )) || status="${again}"; fi
  fi
  PY_PID=""

  case "${status}" in
    0)  date > "${LOGDIR}/.finished"; echo "[lane ${LANE}] finished ${LOGDIR}" ;;
    75) echo "[lane ${LANE}] saved ${LOGDIR}/latest.pt" ;;
    78) date > "${LOGDIR}/.config_mismatch"; echo "[lane ${LANE}] ${LOGDIR} has a checkpoint of other settings; skipped" >&2 ;;
    *)  echo "$(date) exit ${status} on $(hostname)" >> "${LOGDIR}/.failures"
        echo "[lane ${LANE}] exit ${status}, failure $(failure_count) of ${MAX_FAILS} for ${LOGDIR}" >&2 ;;
  esac
  release_run
done

case "${STOP}" in
  usr1)
    if work_remains; then queue_next_slice || exit 75; fi
    exit 0 ;;
  term)
    echo "[lane ${LANE}] stopped by SIGTERM after saving; Slurm requeues it" >&2
    exit 75 ;;
esac
exit 0

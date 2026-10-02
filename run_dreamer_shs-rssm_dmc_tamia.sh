#!/bin/bash
#SBATCH --job-name=shs-rssm-dmc
#SBATCH --account=aip-irina
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=0-23:59:59
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/home/m/memole/links/projects/aip-irina/memole/logs/dmc_tamia_%A_%a.out
#SBATCH --error=/home/m/memole/links/projects/aip-irina/memole/logs/dmc_tamia_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/m/memole/links/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/m/memole/D2E}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/logdir}"
LOGROOT="${LOGROOT:-${PROJECT_DIR}/logs}"
STEPS="${STEPS:-1100000}"
MOVES="${MOVES:-1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-3}"
GPUS_PER_NODE=4
CPUS_PER_RUN=4
MAX_SLICES=30
MAX_FAILS=3
read -r -a ARMS <<< "${ARMS:-gated retain fixed}"
read -r -a TASKS <<< "${TASKS:-walker finger quadruped_walk quadruped_run reacher}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4}"
PACK=$(( RUNS_PER_GPU * GPUS_PER_NODE ))
cd "${PROJECT_DIR}"

RUN_LIST=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      RUN_LIST+=("${arm}/${task}/${seed}")
    done
  done
done
NODES=$(( (${#RUN_LIST[@]} + PACK - 1) / PACK ))

run_settings() {
  local arm task extra
  IFS=/ read -r arm task SEED <<< "$1"
  case "${task}" in
    walker)
      TASK_ID=dmc_walker_walk; DOMAIN=walker; DMC_TASK=walk
      PRESETS=(dmc_walker_shs shs_walker_recipe); RECIPE_FLAGS=() ;;
    finger)
      TASK_ID=dmc_finger_turn_hard; DOMAIN=finger; DMC_TASK=turn_hard
      PRESETS=(dmc_finger_turn_hard_shs_gate)
      RECIPE_FLAGS=(--shs_action_dim 2 --shs_K 20 --shs_q_rank 4 --shs_b0 0.2 --shs_calibrate_sF 0.1 --shs_imag_mode actor_moment) ;;
    quadruped_walk)
      TASK_ID=dmc_quadruped_walk; DOMAIN=quadruped; DMC_TASK=walk
      PRESETS=(dmc_quadruped_walk_shs_gate shs_quadruped_recipe); RECIPE_FLAGS=() ;;
    quadruped_run)
      TASK_ID=dmc_quadruped_run; DOMAIN=quadruped; DMC_TASK=run
      PRESETS=(dmc_quadruped_run_shs_gate shs_quadruped_recipe); RECIPE_FLAGS=(--shs_b0 0.1) ;;
    reacher)
      TASK_ID=dmc_reacher_hard; DOMAIN=reacher; DMC_TASK=hard
      PRESETS=(dmc_reacher_hard_shs); RECIPE_FLAGS=(--shs_action_dim 2 --shs_q_rank 4 --shs_imag_mode actor_moment) ;;
    *) echo "Unknown task ${task} (walker, finger, quadruped_walk, quadruped_run, reacher)" >&2; exit 2 ;;
  esac
  case "${arm}" in
    gated)  ARM_FLAGS=(--shs_ema_gate True --shs_forget_mode fixed) ;;
    retain) ARM_FLAGS=(--shs_ema_gate True --shs_forget_mode evidence) ;;
    *) echo "ARMS entries must be gated or retain, got ${arm}" >&2; exit 2 ;;
  esac
  if [[ "${MOVES}" == 1 ]]; then
    PRESETS+=(shs_move_gate)
    ARM_FLAGS+=(--shs_structure_moves True)
    RUN_TAG="${arm}"
  else
    ARM_FLAGS+=(--shs_structure_moves False)
    RUN_TAG="${arm}_fixedk"
  fi
  TASK="${task}"
  LOGDIR="${RUN_ROOT}/${TASK_ID}_shs_${RUN_TAG}_seed_${SEED}"
  RUNLOG="${LOGROOT}/${TASK_ID}_shs_${RUN_TAG}_seed${SEED}"
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

failure_count() {
  if [[ -f "${LOGDIR}/.failures" ]]; then wc -l < "${LOGDIR}/.failures"; else echo 0; fi
}

run_is_open() {
  [[ ! -f "${LOGDIR}/.finished" && ! -f "${LOGDIR}/.config_mismatch" ]] && (( $(failure_count) < MAX_FAILS ))
}

run_status() {
  local last
  if [[ -f "${LOGDIR}/.finished" ]]; then echo "finished"; return; fi
  if [[ -f "${LOGDIR}/.config_mismatch" ]]; then echo "blocked: checkpoint of other settings"; return; fi
  if (( $(failure_count) >= MAX_FAILS )); then echo "blocked: failed ${MAX_FAILS} times"; return; fi
  last="$(tail -n 1 "${LOGDIR}/metrics.jsonl" 2>/dev/null | grep -o '"step": [0-9.]*' | grep -o '[0-9.]*$' || true)"
  if [[ -n "${last}" ]]; then echo "at frame ${last%.*}"; else echo "not started"; fi
}

node_runs() {
  local first=$(( $1 * PACK ))
  NODE_RUNS=("${RUN_LIST[@]:${first}:${PACK}}")
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "${#RUN_LIST[@]} runs, ${RUNS_PER_GPU} per GPU, ${PACK} per node: sbatch --array=0-$(( NODES - 1 )) run_dreamer_shs-rssm_dmc_best_tamia.sh"
  for node in $(seq 0 $(( NODES - 1 ))); do
    node_runs "${node}"
    for j in "${!NODE_RUNS[@]}"; do
      run_settings "${NODE_RUNS[$j]}"
      printf 'node %d  GPU %d  %-15s %-12s seed %s  %s\n' "${node}" $(( j % GPUS_PER_NODE )) "${TASK}" "${RUN_TAG}" "${SEED}" "$(run_status)"
    done
  done
  exit 0
fi

ID="${SLURM_ARRAY_TASK_ID}"
SLICE="${SLICE:-1}"
if (( ID >= NODES )); then
  echo "Array index ${ID} is outside 0-$(( NODES - 1 )); nothing to run"
  exit 0
fi
node_runs "${ID}"
SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'Command=\K\S+' || true)"
[[ -f "${SCRIPT}" ]] || SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
scontrol update JobId="${SLURM_JOB_ID}" JobName="dmc_node${ID}" 2>/dev/null || true
mkdir -p "${LOGROOT}"
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID} | node ${ID} | slice ${SLICE} | ${#NODE_RUNS[@]} runs, ${RUNS_PER_GPU} per GPU"

set +u
module --force purge || true
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11 scipy-stack/2024a opencv/4.10.0 mujoco/3.3.0
source "${VENV_DIR}/bin/activate"
set -u
unset CUDA_LAUNCH_BLOCKING DISPLAY MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID

export OMP_NUM_THREADS="${CPUS_PER_RUN}"
export MKL_NUM_THREADS="${CPUS_PER_RUN}"
export OPENBLAS_NUM_THREADS="${CPUS_PER_RUN}"
export BLIS_NUM_THREADS="${CPUS_PER_RUN}"
export NUMEXPR_NUM_THREADS="${CPUS_PER_RUN}"
export PYTORCH_NUM_THREADS="${CPUS_PER_RUN}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LP_NUM_THREADS=3

IFS=, read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_IDS[@]} < GPUS_PER_NODE )); then
  echo "Expected ${GPUS_PER_NODE} GPUs, the job has ${#GPU_IDS[@]} (${CUDA_VISIBLE_DEVICES:-none})" >&2
  exit 2
fi

command -v taskset >/dev/null || { echo "taskset is required for CPU binding" >&2; exit 2; }
mapfile -t CPU_SETS < <(python - "${#NODE_RUNS[@]}" "${CPUS_PER_RUN}" <<'PY'
import os
import sys

runs, width = map(int, sys.argv[1:])
cpus = sorted(os.sched_getaffinity(0))
if len(cpus) < runs * width:
    sys.exit(f"Need {runs * width} CPUs, but only {len(cpus)} are available")
for index in range(runs):
    print(",".join(map(str, cpus[index * width:(index + 1) * width])))
PY
)
if (( ${#CPU_SETS[@]} != ${#NODE_RUNS[@]} )); then
  echo "Could not assign ${CPUS_PER_RUN} CPUs to each run" >&2
  exit 2
fi

DOMAINS=""
for entry in "${NODE_RUNS[@]}"; do
  run_settings "${entry}"
  DOMAINS+="${DOMAIN}:${DMC_TASK} "
done
echo "Python: $(command -v python)"
nvidia-smi || true
DOMAINS="${DOMAINS}" python - <<'PY'
import os
import sys
import torch
from dm_control import suite

if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the allocated GPUs")
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "|", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0))
for pair in sorted(set(os.environ["DOMAINS"].split())):
    domain, task = pair.split(":")
    suite.load(domain, task).physics.render(64, 64)
    print("Render OK:", domain, task, "via", os.environ.get("MUJOCO_GL"))
PY

CHAINED=0
queue_next_slice() {
  local reply
  if (( CHAINED )); then return 0; fi
  if (( SLICE + 1 > MAX_SLICES )); then
    echo "MAX_SLICES=${MAX_SLICES} reached for node ${ID}" >&2
    return 1
  fi
  if reply="$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${ID}" \
      --export="ALL,SLICE=$(( SLICE + 1 ))" "${SCRIPT}" 2>&1)"; then
    CHAINED=1
    echo "Slice $(( SLICE + 1 )) queued: ${reply}"
    return 0
  fi
  echo "sbatch of the next slice failed: ${reply}; resubmit with: sbatch --array=${ID} ${SCRIPT}" >&2
  return 1
}

claim_run() {
  local lock="${LOGDIR}/.slurm_lock" owner
  mkdir -p "${LOGDIR}"
  if ! mkdir "${lock}" 2>/dev/null; then
    owner="$(cat "${lock}/job" 2>/dev/null || true)"
    if [[ -n "${owner}" && "${owner}" != "${SLURM_JOB_ID}" ]] && squeue -h -j "${owner}" -t R 2>/dev/null | grep -q .; then
      echo "Job ${owner} is already training ${LOGDIR}" >&2
      return 1
    fi
    rm -rf "${lock}"
    mkdir "${lock}" || return 1
  fi
  echo "${SLURM_JOB_ID}" > "${lock}/job"
}

PIDS=()
DIRS=()
LOCKS=()
release_locks() {
  local d
  for d in "${LOCKS[@]:-}"; do
    [[ -n "${d}" && "$(cat "${d}/.slurm_lock/job" 2>/dev/null || true)" == "${SLURM_JOB_ID}" ]] && rm -rf "${d}/.slurm_lock"
  done
  return 0
}
forward() {
  local pid
  for pid in "${PIDS[@]:-}"; do [[ -n "${pid}" ]] && kill "-$1" "${pid}" 2>/dev/null || true; done
}
trap release_locks EXIT
trap 'queue_next_slice || true; forward USR1' USR1
trap 'forward TERM' TERM

for j in "${!NODE_RUNS[@]}"; do
  run_settings "${NODE_RUNS[$j]}"
  gpu="${GPU_IDS[$(( j % GPUS_PER_NODE ))]}"
  if ! run_is_open; then
    echo "${TASK} ${RUN_TAG} seed ${SEED}: $(run_status), skipped"
    continue
  fi
  claim_run || continue
  LOCKS+=("${LOGDIR}")
  echo "${TASK} ${RUN_TAG} seed ${SEED} on GPU ${gpu}, CPUs ${CPU_SETS[$j]}: $(run_status) -> ${RUNLOG}.{out,err}"
  echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID} | slice ${SLICE} | GPU ${gpu} | CPUs ${CPU_SETS[$j]}" >> "${RUNLOG}.out"
  printf 'Command:' >> "${RUNLOG}.out"; printf ' %q' "${CMD[@]}" >> "${RUNLOG}.out"; echo >> "${RUNLOG}.out"
  taskset -c "${CPU_SETS[$j]}" env CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" >> "${RUNLOG}.out" 2>> "${RUNLOG}.err" &
  PIDS+=("$!")
  DIRS+=("${LOGDIR}")
done

if (( ${#PIDS[@]} == 0 )); then
  echo "No open run on node ${ID}"
  exit 0
fi

failed=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  status=0
  while :; do
    if wait "${pid}"; then status=0; else status=$?; fi
    kill -0 "${pid}" 2>/dev/null || break
  done
  if (( status > 128 )); then
    if wait "${pid}" 2>/dev/null; then status=0; else again=$?; (( again == 127 )) || status="${again}"; fi
  fi
  LOGDIR="${DIRS[$i]}"
  case "${status}" in
    0)  date > "${LOGDIR}/.finished"; echo "finished ${LOGDIR}" ;;
    75) echo "saved ${LOGDIR}/latest.pt" ;;
    78) date > "${LOGDIR}/.config_mismatch"; echo "${LOGDIR} holds a checkpoint trained with other settings; skipped" >&2 ;;
    *)  echo "$(date) exit ${status} on $(hostname)" >> "${LOGDIR}/.failures"; failed=1
        echo "exit ${status}, failure $(failure_count) of ${MAX_FAILS} for ${LOGDIR}" >&2 ;;
  esac
done
trap - USR1 TERM

if (( ! CHAINED )); then
  for entry in "${NODE_RUNS[@]}"; do
    run_settings "${entry}"
    if run_is_open; then queue_next_slice || true; break; fi
  done
fi
exit "${failed}"

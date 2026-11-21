#!/bin/bash
#SBATCH --job-name=shs-dmc-rebuttal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=48gb
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --array=1-36%8
#SBATCH --output=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_dmc_rebuttal_%A_task%a.out
#SBATCH --error=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_dmc_rebuttal_%A_task%a.err

# Rebuttal runs on DMC.  One run per array index: index -> (task, seed), ARM selects the model.
#   index 1-4 walker, 5-8 cheetah, 9-12 hopper, 13-16 pendulum, 17-20 acrobot,
#   21-24 finger_turn_hard, 25-28 reacher_hard, 29-32 quadruped_run, 33-36 quadruped_walk
#
# Every run goes to 1M frames (STEPS) in jobs of one day.  Ten minutes before the time limit
# the job queues its own continuation, dreamer.py saves and exits, and the next job resumes
# from latest.pt; this repeats until dreamer.py reports the run finished (.finished in the log
# directory), at most MAX_SLICES jobs.  A preempted job is requeued by Slurm and resumes the
# same way.  A crash is retried from the last checkpoint, MAX_FAILS launches at most.
#
# Arms (one configuration for every task, schedule in frames; presets shs_rebuttal and rb_* in
# configs.yaml).  Logs under logdir/rebuttal/<task>/<arm>/seed_<n>:
#   dv3           DreamerV3, categorical latents (the baseline of the paper)
#   dv3_gauss     DreamerV3 with Gaussian latents and its neural transition prior
#   rb_k1         one linear-Gaussian action-conditioned regime
#   rb_nostick    K = 16 fixed, no persistence prior
#   rb_sticky     K = 16 fixed, constant stickiness
#   rb_gate       K = 16 fixed, state-dependent persistence gate
#   rb_full       the gate plus merge / delete / birth / split moves
#   rb_ema  rb_power  rb_retain  rb_noforget  rb_frozen     rb_gate with another update of the
#                 regime statistics (plain EMA, power prior, inferred retention, uniform average,
#                 every factor frozen from 200k frames); rb_gate itself uses the
#                 responsibility-gated EMA.  K is fixed and no structure move runs in any of them.
#   rb_handover   rb_full with the memory hand-over of growth moves
#   rb_K<n>       rb_gate with n regimes (regime-count sensitivity), e.g. rb_K4, rb_K32
# Examples:
#   ARM=rb_full sbatch --array=1-4,21-24%8 run_dreamer_shs-rssm_dmc_rebuttal_mila.sh
#   DRY_RUN=1 ARM=rb_k1 SLURM_ARRAY_TASK_ID=21 bash run_dreamer_shs-rssm_dmc_rebuttal_mila.sh  # prints the command
# KEEP_AT (default 500000) makes dreamer.py keep checkpoint_<frames>.pt when a run passes that
# many frames, so that all arms can be compared at that budget later; KEEP_AT= keeps none.
# A fresh run first takes the arm through its whole schedule on random data (selfcheck.py,
# verdict in <logdir>/selfcheck.txt; a verdict other than OK is also copied to
# logs/SELFCHECK_<verdict>_<task>_<arm>_seed<n>.txt).  SELFCHECK=1 (default): training starts
# whatever the check reports.  SELFCHECK=strict: a FAILED check stops the job before training.
# SELFCHECK=0: no check.
# FLAGS adds dreamer.py options; it needs TAG=<name>, which is appended to the arm in the log
# directory so that such a run never shares a directory with the plain arm.
# The arms of the paper (moves, fixedk, ...) stay in run_dreamer_shs-rssm_dmc_mila.sh.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"
SELF_NAME="run_dreamer_shs-rssm_dmc_rebuttal_mila.sh"
MAX_SLICES="${MAX_SLICES:-10}"
MAX_FAILS="${MAX_FAILS:-3}"

TASKS=(walker cheetah hopper pendulum acrobot finger_turn_hard reacher_hard quadruped_run quadruped_walk)
ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( ID < 1 || ID > 36 )); then
  echo "Array index ${ID} is outside the supported range 1-36" >&2
  exit 2
fi
NAME="${NAME:-${TASKS[$(( (ID - 1) / 4 ))]}}"
SEED="${SEED:-$(( (ID - 1) % 4 + 1 ))}"

ARM="${ARM:-}"
TAG="${TAG:-}"
FLAGS="${FLAGS:-}"
ARMS_TEXT="dv3, dv3_gauss, rb_k1, rb_nostick, rb_sticky, rb_gate, rb_full, rb_ema, rb_power, rb_retain, rb_noforget, rb_frozen, rb_handover, rb_K<n>"
case "${ARM}" in
  dv3)           RB_CONFIGS="dmc_vanilla_compare"; ARM_FLAGS="--dyn_discrete 32" ;;
  dv3_gauss)     RB_CONFIGS="dmc_vanilla_compare"; ARM_FLAGS="--dyn_discrete 0" ;;
  rb_k1|rb_nostick|rb_sticky|rb_gate|rb_full)
                 RB_CONFIGS="shs_rebuttal ${ARM}"; ARM_FLAGS="" ;;
  rb_ema|rb_power|rb_retain|rb_noforget|rb_frozen)
                 RB_CONFIGS="shs_rebuttal rb_gate ${ARM}"; ARM_FLAGS="" ;;
  rb_handover)   RB_CONFIGS="shs_rebuttal rb_full rb_handover"; ARM_FLAGS="" ;;
  rb_K[1-9]|rb_K[1-9][0-9])
                 RB_CONFIGS="shs_rebuttal rb_gate"; ARM_FLAGS="--shs_K ${ARM#rb_K}" ;;
  "") echo "Set ARM to one of: ${ARMS_TEXT}" >&2; exit 2 ;;
  moves|fixedk|fixedk_st|moves_st|fixedk_nogate)
      echo "ARM=${ARM} is an arm of run_dreamer_shs-rssm_dmc_mila.sh; this launcher runs: ${ARMS_TEXT}" >&2; exit 2 ;;
  *) echo "ARM must be one of: ${ARMS_TEXT}; got ${ARM}" >&2; exit 2 ;;
esac
PRESET="dmc_${NAME}_shs_moves"

EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
STEPS="${STEPS:-1000000}"
KEEP_AT="${KEEP_AT-500000}"

case "${NAME}" in
  walker)           TASK_ID=dmc_walker_walk ;;
  cheetah)          TASK_ID=dmc_cheetah_run ;;
  hopper)           TASK_ID=dmc_hopper_hop ;;
  pendulum)         TASK_ID=dmc_pendulum_swingup ;;
  acrobot)          TASK_ID=dmc_acrobot_swingup ;;
  finger_turn_hard) TASK_ID=dmc_finger_turn_hard ;;
  reacher_hard)     TASK_ID=dmc_reacher_hard ;;
  quadruped_run)    TASK_ID=dmc_quadruped_run ;;
  quadruped_walk)   TASK_ID=dmc_quadruped_walk ;;
  *) echo "Unknown task ${NAME}" >&2; exit 2 ;;
esac
if [[ -n "${FLAGS}" && -z "${TAG}" ]]; then
  echo "FLAGS needs TAG=<name>, so that the run gets its own log directory" >&2
  exit 2
fi
if [[ -n "${TAG}" && ! "${TAG}" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "TAG may hold letters, digits, dots and dashes only, got '${TAG}'" >&2
  exit 2
fi
if [[ ! "${STEPS}" =~ ^[0-9]+$ ]]; then
  echo "STEPS must be a number of frames, got '${STEPS}'" >&2
  exit 2
fi
if [[ -n "${KEEP_AT}" && ! "${KEEP_AT}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "KEEP_AT must be frame counts separated by commas (or empty), got '${KEEP_AT}'" >&2
  exit 2
fi
SELFCHECK="${SELFCHECK:-1}"
if [[ "${SELFCHECK}" != 0 && "${SELFCHECK}" != 1 && "${SELFCHECK}" != strict ]]; then
  echo "SELFCHECK must be 1 (check, then train whatever it reports), strict (a failed check stops the job) or 0 (no check), got '${SELFCHECK}'" >&2
  exit 2
fi
RUN_NAME="${ARM}${TAG:+_${TAG}}"
CONFIGS="dmc_vision ${RB_CONFIGS}"
ARM_FLAGS="--task ${TASK_ID} ${ARM_FLAGS}"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/rebuttal/${TASK_ID}/${RUN_NAME}/seed_${SEED}}"
RUNLOG="${PROJECT_DIR}/logs/rebuttal_${TASK_ID}_${RUN_NAME}_seed${SEED}"
JOBNAME="rb_${TASK_ID#dmc_}_${RUN_NAME#rb_}_s${SEED}"

read -r -a CONFIG_ARGS <<< "${CONFIGS}"
read -r -a EXTRA_ARGS <<< "${ARM_FLAGS}"
EXTRA_ARGS+=(--steps "${STEPS}")
if [[ -n "${KEEP_AT}" ]]; then
  EXTRA_ARGS+=(--checkpoint_at "${KEEP_AT}")
fi
if [[ -n "${FLAGS}" ]]; then
  read -r -a FLAG_ARGS <<< "${FLAGS}"
  EXTRA_ARGS+=("${FLAG_ARGS[@]}")
fi
COMMAND=(python -u dreamer.py --configs "${CONFIG_ARGS[@]}"
  --seed "${SEED}"
  --compile False
  --batch_size "${BATCH_SIZE}"
  --batch_length "${BATCH_LENGTH}"
  --eval_every "${EVAL_EVERY}"
  --eval_episode_num "${EVAL_EPISODES}"
  --logdir "${LOGDIR}"
  "${EXTRA_ARGS[@]}")

# Frames a finished run was trained for, or nothing.
finished_steps() {
  local n=""
  if [[ -f "${LOGDIR}/.finished" ]]; then
    n="$(sed -n 's/^steps=\([0-9][0-9]*\)$/\1/p' "${LOGDIR}/.finished" | head -n 1)"
    echo "${n:-0}"
  fi
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  STATE="not started"
  [[ -f "${LOGDIR}/latest.pt" ]] && STATE="checkpoint"
  DONE="$(finished_steps)"
  if [[ -n "${DONE}" ]]; then
    if (( DONE >= STEPS )); then STATE="finished"; else STATE="finished at ${DONE} frames, to be continued to ${STEPS}"; fi
  fi
  printf '%-14s %-18s seed %s  %s  [%s]\n     ' "${ARM}" "${NAME}" "${SEED}" "${LOGDIR}" "${STATE}"
  printf ' %q' "${COMMAND[@]}"; printf '\n'
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit with: ARM=${ARM} sbatch --array=<indices> ${SELF_NAME}   (preview: DRY_RUN=1 ARM=${ARM} SLURM_ARRAY_TASK_ID=${ID} bash ${SELF_NAME})" >&2
  exit 2
fi

SLICE="${SLICE:-1}"
FAILS="${FAILS:-0}"
SCRIPT="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'Command=\K\S+' || true)"
[[ -f "${SCRIPT}" ]] || SCRIPT="${PROJECT_DIR}/${SELF_NAME}"

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
echo "[boot] ${ARM} ${NAME} seed ${SEED} -> ${RUNLOG}.{out,err}"
exec >> "${RUNLOG}.out" 2>> "${RUNLOG}.err"

DONE="$(finished_steps)"
if [[ -n "${DONE}" ]] && (( DONE >= STEPS )); then
  echo "=== $(date) | job ${SLURM_JOB_ID} | ${LOGDIR} is finished (${DONE} frames); nothing to do"
  exit 0
fi

# One job per log directory: a lock that names its job, and no second job of the same name.
LOCK="${LOGDIR}/.slurm_lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
  OWNER="$(cat "${LOCK}/job" 2>/dev/null || true)"
  if [[ -n "${OWNER}" && "${OWNER}" != "${SLURM_JOB_ID}" ]] && squeue -h -j "${OWNER}" -t R 2>/dev/null | grep -q .; then
    echo "=== $(date) | job ${SLURM_JOB_ID} | job ${OWNER} is already training ${LOGDIR}; leaving it alone" >&2
    exit 0
  fi
  rm -rf "${LOCK}"
  mkdir "${LOCK}"
fi
echo "${SLURM_JOB_ID}" > "${LOCK}/job"
trap 'rm -rf "${LOCK}"' EXIT
OTHER="$(squeue -h -u "${USER:-$(id -un)}" -n "${JOBNAME}" -t R -o %A 2>/dev/null | grep -vx "${SLURM_JOB_ID}" | head -n 1 || true)"
if [[ -n "${OTHER}" ]]; then
  echo "=== $(date) | job ${SLURM_JOB_ID} | job ${OTHER} with the name ${JOBNAME} is running; leaving ${LOGDIR} to it" >&2
  exit 0
fi

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

scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOBNAME}" 2>/dev/null || true

echo "=== $(date) | $(hostname) | job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task ${SLURM_ARRAY_TASK_ID:-NA} (${SLURM_JOB_ID}) | slice ${SLICE} of at most ${MAX_SLICES} | failures ${FAILS}"
echo "Run:        arm=${ARM} task=${NAME} seed=${SEED} steps=${STEPS}"
echo "Configs:    ${CONFIGS} ${ARM_FLAGS}"
echo "Logdir:     ${LOGDIR}"
echo "Python:     $(command -v python)"
nvidia-smi || true
if [[ -f "${LOGDIR}/latest.pt" ]]; then
  echo "Existing checkpoint found: dreamer.py will resume from ${LOGDIR}/latest.pt"
fi

TASK_KEY="$(python - "${PRESET}" <<'PY'
import sys
from ruamel.yaml import YAML
print(YAML(typ="safe", pure=True).load(open("configs.yaml"))[sys.argv[1]]["task"])
PY
)"
if [[ "${TASK_KEY}" != "${TASK_ID}" ]]; then
  echo "Task of ${PRESET} is ${TASK_KEY}, expected ${TASK_ID}" >&2
  exit 2
fi
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

# The continuation of this run: the same array index and environment in a job that starts when
# this one has ended.
CHAINED=0
queue_next_slice() {
  local fails="$1" reply
  if (( CHAINED )); then return 0; fi
  if (( SLICE + 1 > MAX_SLICES )); then
    echo "MAX_SLICES=${MAX_SLICES} reached for ${LOGDIR}; continue with: ARM=${ARM} sbatch --array=${ID} ${SCRIPT}" >&2
    return 1
  fi
  local try
  for try in 1 2 3; do
    if reply="$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${ID}" \
        --export="ALL,SLICE=$(( SLICE + 1 )),FAILS=${fails}" "${SCRIPT}" 2>&1)"; then
      CHAINED=1
      echo "$(date) slice $(( SLICE + 1 )) queued: ${reply}"
      return 0
    fi
    echo "$(date) sbatch of the next slice failed (attempt ${try} of 3): ${reply}" >&2
    (( try == 3 )) || sleep "${SBATCH_RETRY_WAIT:-20}"
  done
  echo "sbatch of the next slice failed; continue with: ARM=${ARM} sbatch --array=${ID} ${SCRIPT}" >&2
  return 1
}

STOP=""
PY_PID=""
# USR1 arrives ten minutes before the time limit: dreamer.py is told to save and exit, and the
# continuation is queued.  TERM is a preemption or a cancellation: dreamer.py saves and exits,
# nothing is queued here (Slurm requeues a preempted job itself).
trap 'STOP=usr1; [[ -z "${PY_PID}" ]] || kill -USR1 "${PY_PID}" 2>/dev/null || true; queue_next_slice "${FAILS}" || true' USR1
trap 'STOP=term; [[ -z "${PY_PID}" ]] || kill -TERM "${PY_PID}" 2>/dev/null || true' TERM

# Before a fresh run: the arm goes through every phase of its schedule on random data on this
# GPU (calibration, re-seed, sweeps, births, splits, freeze, logging, restoring a saved model).
# The verdict is the first line of <logdir>/selfcheck.txt: OK, INCOMPLETE (a part was not
# reached), FAILED or NO VERDICT (cut off).  Anything but OK is copied to logs/SELFCHECK_*.
# Training starts whatever the verdict is, unless SELFCHECK=strict and the verdict is FAILED.
if [[ "${SELFCHECK}" != 0 && ! -f "${LOGDIR}/latest.pt" ]]; then
  python -u selfcheck.py "${COMMAND[@]:3}" --selfcheck_report "${LOGDIR}/selfcheck.txt" &
  PY_PID=$!
  sc=0
  wait "${PY_PID}" || sc=$?
  PY_PID=""
  if [[ -z "${STOP}" ]]; then
    VERDICT="$(head -n 1 "${LOGDIR}/selfcheck.txt" 2>/dev/null | sed -n 's/^\[selfcheck\] \(OK\|INCOMPLETE\|FAILED\|NO VERDICT\).*/\1/p' || true)"
    VERDICT="${VERDICT:-NO VERDICT}"
    rm -f "${PROJECT_DIR}/logs/SELFCHECK_"*"_${TASK_ID}_${RUN_NAME}_seed${SEED}.txt"
    if [[ "${VERDICT}" != OK ]]; then
      FLAG="${PROJECT_DIR}/logs/SELFCHECK_${VERDICT// /_}_${TASK_ID}_${RUN_NAME}_seed${SEED}.txt"
      cp "${LOGDIR}/selfcheck.txt" "${FLAG}" 2>/dev/null || echo "[selfcheck] NO VERDICT: no report was written" > "${FLAG}"
      echo "[selfcheck] ${ARM}: ${VERDICT} (exit ${sc}); see ${FLAG}" >&2
    fi
    if [[ "${SELFCHECK}" == strict && "${VERDICT}" == FAILED ]]; then
      echo "$(date) SELFCHECK=strict and the check failed: not training.  Submit again once it is fixed, or with SELFCHECK=1 to train regardless." >&2
      exit 1
    fi
    [[ "${VERDICT}" == OK ]] || echo "[selfcheck] training starts anyway (SELFCHECK=${SELFCHECK})" >&2
  fi
fi
if [[ -n "${STOP}" ]]; then
  echo "$(date) stopped by ${STOP} before training started"
  if [[ "${STOP}" == usr1 ]] && (( CHAINED )); then exit 0; fi
  exit 75
fi

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
trap ':' USR1 TERM

case "${status}" in
  0)
    { echo "steps=${STEPS}"; date; } > "${LOGDIR}/.finished"
    echo "$(date) finished ${LOGDIR} (${STEPS} frames)" ;;
  75)
    if (( CHAINED )); then
      echo "$(date) saved ${LOGDIR}/latest.pt; the next slice continues"
      status=0
    elif [[ "${STOP}" == term ]]; then
      echo "$(date) saved ${LOGDIR}/latest.pt after SIGTERM; a preempted job is requeued by Slurm" >&2
    else
      echo "latest.pt is saved but no next slice is queued; continue with: ARM=${ARM} sbatch --array=${ID} ${SCRIPT}" >&2
    fi ;;
  78)
    echo "${LOGDIR} holds a checkpoint trained with other settings; use TAG=<name> or move the directory away" >&2 ;;
  *)
    if [[ "${STOP}" == term ]]; then
      echo "$(date) dreamer.py exited with ${status} after SIGTERM; a preempted job is requeued by Slurm" >&2
    elif [[ -f "${LOGDIR}/latest.pt" ]] && (( FAILS + 1 < MAX_FAILS )); then
      echo "$(date) dreamer.py exited with ${status}; retrying from the last checkpoint (failure $(( FAILS + 1 )) of ${MAX_FAILS})" >&2
      queue_next_slice $(( FAILS + 1 )) || true
    else
      echo "$(date) dreamer.py exited with ${status}; not retried" >&2
    fi ;;
esac
exit "${status}"

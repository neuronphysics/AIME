#!/bin/bash
#SBATCH --job-name=DMC-SHS
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01-21:59:59
#SBATCH --account=def-irina
#SBATCH --array=0-3
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# SHS-RSSM on DeepMind Control (pixels).  Array index = seed when TASK is set,
# otherwise IDX = task*NSEEDS + seed.
#
#   ARM=shs      (default)  --configs dmc_vision dmc_<task>_shs
#   ARM=vanilla             --configs dmc_vision dmc_vanilla_compare
#
#   bash submit_all.sh                      # every task, 4 seeds, named per task
#   TASK=walker NSEEDS=4 sbatch --array=0-3 run_dreamer_shs-rssm_dmc.sh
#   NSEEDS=4 sbatch --array=0-35 run_dreamer_shs-rssm_dmc.sh   # all in one array
#   PILOT=1 TASK=walker SEED=0 sbatch --array=0 run_dreamer_shs-rssm_dmc.sh
#   FLAGS="--shs_live_moves False" bash submit_all.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/links/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/links/D2E}"

TASKS=(walker hopper cheetah acrobot pendulum finger_turn_hard quadruped_walk
       quadruped_run reacher_hard)
declare -A ENV_TASK=([walker]=dmc_walker_walk [hopper]=dmc_hopper_hop [cheetah]=dmc_cheetah_run
                     [acrobot]=dmc_acrobot_swingup [pendulum]=dmc_pendulum_swingup
                     [finger_turn_hard]=dmc_finger_turn_hard [quadruped_walk]=dmc_quadruped_walk
                     [quadruped_run]=dmc_quadruped_run [reacher_hard]=dmc_reacher_hard)
NSEEDS="${NSEEDS:-4}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASK:-${TASKS[$(( IDX / NSEEDS ))]}}"
SEED="${SEED:-$(( IDX % NSEEDS ))}"
ARM="${ARM:-shs}"
PILOT="${PILOT:-0}"
STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

# Job name is fixed when sbatch parses the directives, so rename now that TASK
# and SEED are known. Harmless if it fails (e.g. run outside SLURM).
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="DMC-${TASK}-s${SEED}" || true
fi

case "${ARM}" in
  vanilla) CONFIGS="dmc_vision dmc_vanilla_compare"; TAG="dreamerv3" ;;
  *)       CONFIGS="dmc_vision dmc_${TASK}_shs";     TAG="shs" ;;
esac
if [ "${PILOT}" = "1" ]; then STEPS=20000; TAG="${TAG}_pilot"; fi
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/${TASK}_${TAG}/seed_${SEED}}"

RUN_TAG="shs-rssm"; [ "${ARM}" = "vanilla" ] && RUN_TAG="vanilla"
RUN_NAME="dreamerv3_${RUN_TAG}_${TASK}_${SEED}"
[ "${PILOT}" = "1" ] && RUN_NAME="${RUN_NAME}_pilot"
LOGROOT="${LOGROOT:-${PROJECT_DIR}/logs}"
mkdir -p "${LOGROOT}"
exec > "${LOGROOT}/${RUN_NAME}.out" 2> "${LOGROOT}/${RUN_NAME}.err"

module --force purge || true
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a
module load opencv/4.10.0
module load mpi4py/3.1.6
module load arrow
source "${VENV_DIR}/bin/activate"

unset CUDA_LAUNCH_BLOCKING
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

mkdir -p "${LOGDIR}"
cd "${PROJECT_DIR}"

echo "[run] arm=${ARM} task=${TASK} (${ENV_TASK[$TASK]}) seed=${SEED} steps=${STEPS}"
echo "[run] configs='${CONFIGS}' logdir=${LOGDIR}"
python -u dreamer.py --configs ${CONFIGS} \
  --task "${ENV_TASK[$TASK]}" \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --logdir "${LOGDIR}" ${FLAGS}

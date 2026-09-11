#!/bin/bash
#SBATCH --job-name=DMC-SHS
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01-20:59:59
#SBATCH --account=aip-irina
#SBATCH --array=0-34
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# SHS-RSSM on DeepMind Control (pixels).  Array = TASKS x NSEEDS.
#
#   ARM=shs      (default)  --configs dmc_vision dmc_<task>_shs
#   ARM=vanilla             --configs dmc_vision dmc_vanilla_compare
#
#   sbatch run_dreamer_shs-rssm_dmc.sh
#   ARM=vanilla sbatch run_dreamer_shs-rssm_dmc.sh
#   TASK=walker NSEEDS=4 sbatch --array=0-3 run_dreamer_shs-rssm_dmc.sh
#   PILOT=1 TASK=walker SEED=0 sbatch --array=0 run_dreamer_shs-rssm_dmc.sh
#   FLAGS="--shs_live_moves False" sbatch run_dreamer_shs-rssm_dmc.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

TASKS=(walker hopper cheetah acrobot pendulum finger_turn_hard quadruped_walk)
declare -A ENV_TASK=([walker]=dmc_walker_walk [hopper]=dmc_hopper_hop [cheetah]=dmc_cheetah_run
                     [acrobot]=dmc_acrobot_swingup [pendulum]=dmc_pendulum_swingup
                     [finger_turn_hard]=dmc_finger_turn_hard [quadruped_walk]=dmc_quadruped_walk)
NSEEDS="${NSEEDS:-5}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASK:-${TASKS[$(( IDX / NSEEDS ))]}}"
SEED="${SEED:-$(( IDX % NSEEDS ))}"
ARM="${ARM:-shs}"
PILOT="${PILOT:-0}"
STEPS="${STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

case "${ARM}" in
  vanilla) CONFIGS="dmc_vision dmc_vanilla_compare"; TAG="dreamerv3" ;;
  *)       CONFIGS="dmc_vision dmc_${TASK}_shs";     TAG="shs" ;;
esac
if [ "${PILOT}" = "1" ]; then STEPS=20000; TAG="${TAG}_pilot"; fi
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/${TASK}_${TAG}/seed_${SEED}}"

# The %x_%A_%a files in the submit directory stay as near-empty bootstrap stubs.
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

#!/bin/bash
#SBATCH --job-name=PROCGEN-SHS
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06-23:59:59
#SBATCH --account=aip-irina
#SBATCH --array=0-9
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# SHS-RSSM on Procgen.
#
#   ARM=shs      (default)  --configs procgen procgen_<game>_shs
#   ARM=vanilla             --configs procgen
#
#   sbatch run_dreamer_shs-rssm_procgen.sh
#   ARM=vanilla sbatch run_dreamer_shs-rssm_procgen.sh
#   GAME=coinrun SEED=0 sbatch --array=0 run_dreamer_shs-rssm_procgen.sh
#   PILOT=1 GAME=bigfish SEED=0 sbatch --array=0 run_dreamer_shs-rssm_procgen.sh
#
# Games without a dedicated preset fall back to the generic procgen_shs preset.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

GAMES=(bigfish fruitbot)
NSEEDS="${NSEEDS:-5}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
GAME="${GAME:-${GAMES[$(( IDX / NSEEDS ))]}}"
SEED="${SEED:-$(( IDX % NSEEDS ))}"
ARM="${ARM:-shs}"
PILOT="${PILOT:-0}"
STEPS="${STEPS:-50000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

PRESET="procgen_${GAME}_shs"
if ! grep -q "^${PRESET}:" "${PROJECT_DIR}/configs.yaml"; then PRESET="procgen_shs"; fi
if [ "${ARM}" = "vanilla" ]; then
  CONFIGS="procgen"; TAG="dreamerv3"
else
  CONFIGS="procgen ${PRESET}"; TAG="shs"
fi
if [ "${PILOT}" = "1" ]; then STEPS=20000; TAG="${TAG}_pilot"; fi
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/procgen_${GAME}_${TAG}/seed_${SEED}}"


RUN_TAG="shs-rssm"; [ "${ARM}" = "vanilla" ] && RUN_TAG="vanilla"
RUN_NAME="dreamerv3_${RUN_TAG}_${GAME}_${SEED}"
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

mkdir -p "${LOGDIR}"
cd "${PROJECT_DIR}"

echo "[run] arm=${ARM} game=${GAME} seed=${SEED} steps=${STEPS}"
echo "[run] configs='${CONFIGS}' logdir=${LOGDIR}"
python -u dreamer.py --configs ${CONFIGS} \
  --task "procgen_${GAME}" \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --logdir "${LOGDIR}" ${FLAGS}

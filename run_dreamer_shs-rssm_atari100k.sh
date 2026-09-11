#!/bin/bash
#SBATCH --job-name=ATARI100K-SHS
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00-23:59:59
#SBATCH --account=aip-irina
#SBATCH --array=0-29
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# SHS-RSSM on the Atari100k protocol (4e5 env frames = 100k agent steps).
# Requires the ALE ROMs in the venv (see envs/setup_scripts/atari.sh).
#
#   ARM=shs      (default)  --configs atari100k atari100k_shs atari_<game>_shs
#   ARM=vanilla             --configs atari100k
#
#   sbatch run_dreamer_shs-rssm_atari100k.sh
#   ARM=vanilla sbatch run_dreamer_shs-rssm_atari100k.sh
#   GAME=seaquest SEED=0 sbatch --array=0 run_dreamer_shs-rssm_atari100k.sh
#   PILOT=1 GAME=breakout SEED=0 sbatch --array=0 run_dreamer_shs-rssm_atari100k.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

GAMES=(breakout demon_attack ms_pacman alien seaquest hero)
NSEEDS="${NSEEDS:-5}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
GAME="${GAME:-${GAMES[$(( IDX / NSEEDS ))]}}"
SEED="${SEED:-$(( IDX % NSEEDS ))}"
ARM="${ARM:-shs}"
PILOT="${PILOT:-0}"
STEPS="${STEPS:-400000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

if [ "${ARM}" = "vanilla" ]; then
  CONFIGS="atari100k"; TAG="dreamerv3"
else
  CONFIGS="atari100k atari100k_shs atari_${GAME}_shs"; TAG="shs"
fi
if [ "${PILOT}" = "1" ]; then STEPS=20000; TAG="${TAG}_pilot"; fi
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/atari100k_${GAME}_${TAG}/seed_${SEED}}"


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
  --task "atari_${GAME}" \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --logdir "${LOGDIR}" ${FLAGS}

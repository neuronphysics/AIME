#!/bin/bash
#SBATCH --job-name=ATARI-SHS
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01-23:59:59
#SBATCH --account=aip-irina
#SBATCH --array=1-24
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamer_shs_atari_%A_%a.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamer_shs_atari_%A_%a.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/memole/scratch/AIME}"
VENV_DIR="${VENV_DIR:-/home/memole/D2E}"

GAMES=(pong breakout alien hero demon_attack frostbite)
NSEED="${NSEED:-4}"
ID="${SLURM_ARRAY_TASK_ID:-1}"
if (( ID < 1 || ID > ${#GAMES[@]} * NSEED )); then
  echo "Array index ${ID} is outside the supported range 1-$(( ${#GAMES[@]} * NSEED ))" >&2
  exit 2
fi
GAME="${GAME:-${GAMES[$(( (ID - 1) / NSEED ))]}}"
SEED="${SEED:-$(( (ID - 1) % NSEED + 1 ))}"

ARM="${ARM:-fixedk}"
PRESET="atari_${GAME}_shs_moves"
case "${ARM}" in
  moves)    CONFIGS="atari100k ${PRESET}";               ARM_FLAGS="" ;;
  fixedk)   CONFIGS="atari100k ${PRESET}";               ARM_FLAGS="--shs_structure_moves False" ;;
  moves_st) CONFIGS="atari100k ${PRESET} shs_imag_st";   ARM_FLAGS="" ;;
  vanilla)  CONFIGS="atari100k ${PRESET} atari_vanilla"; ARM_FLAGS="" ;;
  *) echo "ARM must be moves, fixedk, moves_st or vanilla, got ${ARM}" >&2; exit 2 ;;
esac

STEPS="${STEPS:-1000000}"
EVAL_EVERY="${EVAL_EVERY:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
FLAGS="${FLAGS:-}"

LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/atari_${ARM}/${GAME}/seed_${SEED}}"

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1
unset DISPLAY

mkdir -p "${PROJECT_DIR}/logs" "${LOGDIR}"
cd "${PROJECT_DIR}"

echo "Date: $(date)  Host: $(hostname)  Array: ${SLURM_ARRAY_JOB_ID:-NA}/${SLURM_ARRAY_TASK_ID:-NA}"
echo "Run:  arm=${ARM} game=${GAME} seed=${SEED} steps=${STEPS} batch=${BATCH_SIZE}x${BATCH_LENGTH}"
echo "Configs: ${CONFIGS} ${ARM_FLAGS} ${FLAGS}"
echo "Eval: every ${EVAL_EVERY} frames, ${EVAL_EPISODES} episodes"
echo "Log:  ${LOGDIR}"
nvidia-smi || echo "nvidia-smi unavailable"

python - "${GAME}" <<'PY'
import sys
import numpy as np
import torch
import envs.atari as atari

assert torch.cuda.is_available(), "PyTorch cannot see the allocated GPU"
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
env = atari.Atari(sys.argv[1], 4, (64, 64), gray=False, noops=30, lives="unused",
                  sticky=False, actions="needed", resize="opencv", seed=0)
env.reset()
n = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
obs, reward, done, info = env.step(np.zeros(n, np.float32))
assert obs["image"].shape == (64, 64, 3), obs["image"].shape
print(f"[atari] {sys.argv[1]}: {n} actions, frame {obs['image'].shape}, reward {reward}")
PY

read -r -a CONFIG_ARGS <<< "${CONFIGS}"
EXTRA_ARGS=()
if [[ -n "${ARM_FLAGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${ARM_FLAGS}"
fi
if [[ -n "${FLAGS}" ]]; then
  read -r -a FLAG_ARGS <<< "${FLAGS}"
  EXTRA_ARGS+=("${FLAG_ARGS[@]}")
fi

python -u dreamer.py --configs "${CONFIG_ARGS[@]}" \
  --seed "${SEED}" \
  --compile False \
  --batch_size "${BATCH_SIZE}" \
  --batch_length "${BATCH_LENGTH}" \
  --steps "${STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_episode_num "${EVAL_EPISODES}" \
  --logdir "${LOGDIR}" \
  "${EXTRA_ARGS[@]}"

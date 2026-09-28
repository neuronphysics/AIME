#!/bin/bash
#SBATCH --job-name=shs-atari
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=48gb
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=1-24%8
#SBATCH --output=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_atari_%A_task%a.out
#SBATCH --error=/home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/shs_atari_%A_task%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"

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
RUNLOG="${PROJECT_DIR}/logs/atari_${ARM}_${GAME}_seed${SEED}"

mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "${RUNLOG}.out") 2> >(tee -a "${RUNLOG}.err" >&2)

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
export TORCH_LINALG_PREFER_CUSOLVER=1
export TORCH_DISABLE_ADDR2LINE=1
unset DISPLAY

cd "${PROJECT_DIR}"

scontrol update JobId="${SLURM_JOB_ID}" JobName="atari_${ARM}_${GAME}_s${SEED}" 2>/dev/null || true

echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Slurm:      job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}} task=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Run:        arm=${ARM} game=${GAME} seed=${SEED} steps=${STEPS}"
echo "Configs:    ${CONFIGS} ${ARM_FLAGS} ${FLAGS}"
echo "Eval:       every ${EVAL_EVERY} frames, ${EVAL_EPISODES} episodes"
echo "Logdir:     ${LOGDIR}"
echo "Python:     $(command -v python)"
nvidia-smi

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

#!/bin/bash
#SBATCH --job-name=AIME-CARL
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=48gb
#SBATCH --mem=64G
#SBATCH --time=1-17:00:00
#SBATCH --array=0-39%12                                   # 5 grids x {shs, vanilla} x 4 seeds
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/bootstrap/slurm-aime-carl-%A_%a.out
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/bootstrap/slurm-aime-carl-%A_%a.err

# SHS-RSSM on CARL contextual dm_control (pixels), Mila cluster.
# One submission = every CARL grid x {SHS-RSSM, vanilla DreamerV3} x seeds 1..4.
#   index 0-19  : SHS      (preset = idx/4,      seed = idx%4+1)
#   index 20-39 : vanilla  (preset = (idx-20)/4, seed = idx%4+1)
#
#   sbatch run_dreamer_carl_mila.sh                                  # all 40
#   sbatch --array=0-3 run_dreamer_carl_mila.sh                      # carl_walker_shs, seeds 1-4
#   sbatch --array=20-23 run_dreamer_carl_mila.sh                    # walker gravity grid, vanilla Dreamer
#   sbatch --array=8 run_dreamer_carl_mila.sh                        # carl_quadruped_walk_shs seed 1 (canary)
#   VISIBLE=1 sbatch run_dreamer_carl_mila.sh                        # + carl_context_visible on every run
#   STEPS=500000 sbatch run_dreamer_carl_mila.sh
#   FLAGS="--batch_size 8" sbatch run_dreamer_carl_mila.sh
#
# Before the first submit:  mkdir -p /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/bootstrap

set -uo pipefail

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

PROJECT_DIR="${PROJECT_DIR:-/home/mila/z/zahra.sheikhbahaee/scratch/AIME}"
CONDA_ENV="${CONDA_ENV:-trifinger_rl_venv}"
mkdir -p "${PROJECT_DIR}/logs/bootstrap"

# ---- preset / seed from the array index --------------------------------------
PRESETS=(
  carl_walker_shs                     # walker-walk, gravity grid
  carl_walker_gravity_friction_shs    # walker-walk, gravity x friction grid
  carl_quadruped_walk_shs             # quadruped-walk, gravity grid
  carl_quadruped_actuator_shs         # quadruped-walk, actuator_strength grid
  carl_finger_spin_shs                # finger-spin, gravity grid
)
MODELS=(shs vanilla)                  # vanilla = same grid, + carl_vanilla overlay (use_shs False, dyn_discrete 32)
NSEED=4
NPRESET=${#PRESETS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= NPRESET * NSEED * ${#MODELS[@]} )); then
  echo "array index ${IDX} out of range (${NPRESET} presets x ${#MODELS[@]} models x ${NSEED} seeds)" >&2; exit 2
fi
MODEL="${MODELS[$(( IDX / (NPRESET * NSEED) ))]}"
REM=$(( IDX % (NPRESET * NSEED) ))
PRESET="${PRESETS[$(( REM / NSEED ))]}"
SEED="${SEED:-$(( REM % NSEED + 1 ))}"

case "${PRESET}" in
  carl_walker*)             CARL_TASK=dmc_walker ;;
  carl_quadruped*)          CARL_TASK=dmc_quadruped ;;
  carl_finger*)             CARL_TASK=dmc_finger ;;
  *) echo "unknown preset '${PRESET}'" >&2; exit 2 ;;
esac

EXTRA_CONFIGS=""
RUN="${PRESET}"
if [[ "${MODEL}" == "vanilla" ]]; then EXTRA_CONFIGS="carl_vanilla"; RUN="${PRESET%_shs}_vanilla"; fi
if [[ "${VISIBLE:-0}" == "1" ]]; then EXTRA_CONFIGS="${EXTRA_CONFIGS} carl_context_visible"; RUN="${RUN}_visible"; fi

# conda's activate is not `set -u` clean; relax strictness across it only.
set +u
module unload python
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

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_EGL_DEV="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${_EGL_DEV%%,*}"
export EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID}"
export DISABLE_RENDER_THREAD_OFFLOADING=1
unset DISPLAY

cd "${PROJECT_DIR}"

STEPS="${STEPS:-1000000}"
FLAGS="${FLAGS:-}"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/logdir/${RUN}/seed_${SEED}}"
RUNLOG="${PROJECT_DIR}/logs/aime_${RUN}_seed${SEED}"

scontrol update JobId="${SLURM_JOB_ID}" JobName="aime_${RUN}_s${SEED}" 2>/dev/null || true

# Human-readable per-run logs; SLURM cannot expand PRESET/SEED in #SBATCH -o.
exec > "${RUNLOG}.out" 2> "${RUNLOG}.err"

nvidia-smi || echo "nvidia-smi unavailable"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python:  $(command -v python)"
echo "Configs: ${PRESET} ${EXTRA_CONFIGS}  (${CARL_TASK}, model=${MODEL})  seed=${SEED}  steps=${STEPS}"
echo "Logdir:  ${LOGDIR}"

# ---- preflight: the full adapter chain (gym, carl, dm_control, render) -------
carl_preflight () {
python - <<PY
import os, sys, torch
print("Python:", sys.executable)
print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
import importlib.metadata as md, gym, gymnasium, carl
print("gym", gym.__version__, "| gymnasium", gymnasium.__version__, "| carl-bench", md.version("carl-bench"))
import envs.carl as ec
e = ec.CARL("${CARL_TASK}", 2, (64, 64), seed=0, contexts={"gravity": [4.905, 19.62]}, selector="round_robin")
o = e.reset()
g = -e._env.env.env.physics.model.opt.gravity[2]
print("CARL render:", o["image"].shape, "| action", e.action_space.shape, "| context_id", int(o["context_id"]), "| physics gravity", round(float(g), 3))
assert abs(g - 4.905) < 1e-6, "context did not reach the physics"
PY
}

if ! carl_preflight; then
  echo "[gl] egl preflight failed on $(hostname); falling back to osmesa" >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  carl_preflight || { echo "[gl] osmesa also failed on $(hostname)" >&2; exit 3; }
fi

# ---- run ---------------------------------------------------------------------

python -u dreamer.py --configs "${PRESET}" ${EXTRA_CONFIGS} \
  --seed "${SEED}" \
  --compile False \
  --steps "${STEPS}" \
  --logdir "${LOGDIR}" \
  ${FLAGS}
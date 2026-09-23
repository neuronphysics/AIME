#!/bin/bash
#SBATCH --job-name=AIME-CARL
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=1-17:00:00
#SBATCH --array=0-49%2
#SBATCH --output=slurm-aime-carl-%A_%a.out
#SBATCH --error=slurm-aime-carl-%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${PROJECT_DIR}"
[[ -f dreamer.py && -f configs.yaml ]] || { echo "PROJECT_DIR must point to the AIME checkout" >&2; exit 2; }

PRESETS=(
  carl_walker_shs
  carl_walker_gravity_friction_shs
  carl_quadruped_walk_shs
  carl_quadruped_actuator_shs
  carl_finger_spin_shs
)
MODELS=(shs vanilla)
NSEED=5
NPRESET=${#PRESETS[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${IDX}" =~ ^(0|[1-9][0-9]?)$ ]] || (( IDX >= NPRESET * NSEED * ${#MODELS[@]} )); then
  echo "array index '${IDX}' must be an integer from 0 to 49" >&2
  exit 2
fi
MODEL="${MODELS[$(( IDX / (NPRESET * NSEED) ))]}"
REM=$(( IDX % (NPRESET * NSEED) ))
PRESET="${PRESETS[$(( REM / NSEED ))]}"
SEED="${SEED:-$(( REM % NSEED + 1 ))}"
CONFIGS=("${PRESET}")
RUN="${PRESET}"
if [[ "${MODEL}" == "vanilla" ]]; then
  CONFIGS+=(carl_vanilla)
  RUN="${PRESET%_shs}_vanilla"
fi
if [[ "${VISIBLE:-0}" == "1" ]]; then
  CONFIGS+=(carl_context_visible)
  RUN="${RUN}_visible"
fi

STEPS="${STEPS:-1000000}"
LOGDIR="${LOGDIR:-${LOG_ROOT:-${PROJECT_DIR}/logdir}/${RUN}/seed_${SEED}}"
COMMAND=(python -u dreamer.py --configs "${CONFIGS[@]}"
  --seed "${SEED}" --compile False --steps "${STEPS}" --logdir "${LOGDIR}")
if [[ -n "${FLAGS:-}" ]]; then
  read -r -a FLAG_ARGS <<< "${FLAGS}"
  COMMAND+=("${FLAG_ARGS[@]}")
fi
COMMAND+=("$@")

echo "Date: $(date) | Host: $(hostname) | Array index: ${IDX}"
echo "Configs: ${CONFIGS[*]} | Model: ${MODEL} | Seed: ${SEED}"
echo "Project: ${PROJECT_DIR} | Logdir: ${LOGDIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit with sbatch (or use DRY_RUN=1 to inspect without a GPU)." >&2
  exit 2
fi
if [[ -n "${VENV_PATH:-}" && -n "${CONDA_ENV:-}" ]]; then
  echo "Set only one of VENV_PATH and CONDA_ENV." >&2
  exit 2
fi

set +u
if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    module load "${CONDA_MODULE:-miniconda/3}"
  fi
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u
command -v python

unset CUDA_LAUNCH_BLOCKING
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONNOUSERSITE=1
export TORCH_SHOW_CPP_STACKTRACES=1 TORCH_LINALG_PREFER_CUSOLVER=1 TORCH_DISABLE_ADDR2LINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${MUJOCO_GL}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  EGL_CANDIDATE="${CUDA_VISIBLE_DEVICES:-}"
  EGL_CANDIDATE="${EGL_CANDIDATE%%,*}"
  if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" && "${EGL_CANDIDATE}" =~ ^[0-9]+$ ]]; then
    export MUJOCO_EGL_DEVICE_ID="${EGL_CANDIDATE}"
  fi
fi
export DISABLE_RENDER_THREAD_OFFLOADING=1
unset DISPLAY

nvidia-smi || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
git rev-parse HEAD || true
python - <<'PY'
import importlib.metadata as metadata
import sys
import torch
import dreamer

print("Python:", sys.executable)
for package in ("torch", "numpy", "gym", "gymnasium", "carl-bench", "dm-control", "mujoco"):
    print(package, metadata.version(package))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; check the GPU allocation and PyTorch installation")
print("GPU:", torch.cuda.get_device_name(0))
PY

carl_preflight() {
  python - "${PRESET}" "${SEED}" <<'PY'
import pathlib
import sys

import numpy as np
from ruamel.yaml import YAML
from envs.carl import CARL

configs = YAML(typ="safe", pure=True).load(pathlib.Path("configs.yaml").read_text())
preset = configs[sys.argv[1]]
for split in ("train", "eval"):
    spec = preset["carl"][f"{split}_contexts"]
    environment = CARL(
        preset["task"].removeprefix("carl_"),
        action_repeat=preset["action_repeat"],
        size=(64, 64),
        seed=int(sys.argv[2]),
        contexts=spec,
        selector="round_robin",
    )
    context_count = int(np.prod([len(values) for values in spec.values()]))
    seen = set()
    for context_index in range(context_count):
        observation = environment.reset()
        seen.add(int(observation["context_id"]))
        assert observation["image"].shape == (64, 64, 3)
        assert environment.action_space.shape == (preset["shs_action_dim"],)
        if "gravity" in spec:
            gravity = -environment._env.env.env.physics.model.opt.gravity[2]
            assert np.isclose(gravity, environment.context["gravity"]), "context did not reach physics"
        observation, reward, done, info = environment.step(
            np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        assert np.isfinite(reward)
        assert np.isfinite(observation["state"]).all()
    assert len(seen) == context_count, (seen, context_count)
    print(f"{split}: rendered and stepped all {context_count} contexts: {spec}")
PY
}

if ! carl_preflight; then
  if [[ "${MUJOCO_GL}" != "egl" ]]; then
    exit 3
  fi
  echo "Preflight failed with EGL; retrying with OSMesa." >&2
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=3
  unset MUJOCO_EGL_DEVICE_ID EGL_DEVICE_ID
  carl_preflight || { echo "OSMesa preflight also failed; see traceback." >&2; exit 3; }
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec srun "${COMMAND[@]}"

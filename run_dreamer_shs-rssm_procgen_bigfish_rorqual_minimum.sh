#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=01-00:59:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-run-cheetah-seed-1_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-run-cheetah-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
set -euo pipefail

#Start clean
module --force purge

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a   # required by opencv/4.10.0
module load opencv/4.10.0
module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow

# Keep accelerator launches asynchronous. A CUDA_LAUNCH_BLOCKING value inherited
# from a debugging shell can make this workload orders of magnitude slower.
unset CUDA_LAUNCH_BLOCKING

#virtualenv --no-download --clear /home/memole/links/D2E
source /home/memole/links/D2E/bin/activate
### FORCE SINGLE-THREAD BLAS / OMP / TORCH
# ---- VRAM allocator / fragmentation fixes ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8
export CUDA_MODULE_LOADING=LAZY

#pip install --no-index --no-cache-dir --no-deps "torch==2.5.1+computecanada" "sympy==1.13.1+computecanada" "torchvision==0.20.1+computecanada" "torchaudio==2.5.1+computecanada"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
# keep Python from seeing ~/.local
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
python -X faulthandler - <<'PY'
import os, torch
print("python ok")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024,1024, device="cuda")
y = x @ x
print("matmul ok:", y.mean().item())
PY
#python -m pip install --force-reinstall --no-deps mujoco-3.10.0-cp311-cp311-linux_x86_64.whl dm_control-1.0.43-py3-none-any.whl
cd /home/memole/links/projects/def-irina/memole/AIME

python - <<'PY'
import pathlib, shs_rssm
print("SHS package:", pathlib.Path(shs_rssm.__file__).resolve())
print("SHS version:", shs_rssm.__version__)
PY

: <<'PROCGEN_BUILD_NOTES'

# ============================================================================
# One-time Procgen 0.10.7 build instructions for Compute Canada
# ============================================================================

test -d procgen-0.10.7-src/.git || \
    git clone --depth 1 --branch 0.10.7 \
    https://github.com/openai/procgen.git \
    procgen-0.10.7-src

salloc \
    --account=def-irina \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=8G

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load python/3.11
module load scipy-stack/2024a
module load cmake
module load qt/5.15.11

# Activate D2E only after loading all modules.
source /home/memole/links/D2E/bin/activate

# Confirm that installation will go into D2E.
python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy; print('NumPy:', numpy.__version__)"

# Install dependencies from the Alliance wheelhouse.
python -m pip install --no-index \
    "gym<1" \
    "filelock<4" \
    "imageio<3"

# Skip gym3's unavailable graphical moderngl dependency.
python -m pip install --no-index --no-deps "gym3==0.3.3"

python -c "import gym3; print('gym3:', gym3.__file__)"

# Configure the Procgen C++ build.
export PROCGEN_CMAKE_PREFIX_PATH="$(qmake -query QT_INSTALL_PREFIX)/lib/cmake/Qt5"
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-8}"

echo "Qt path: $PROCGEN_CMAKE_PREFIX_PATH"

test -f "$PROCGEN_CMAKE_PREFIX_PATH/Qt5Config.cmake" \
    && echo "Qt5 configuration found"

# Make a clean source copy on the compute node's local storage.
AIME_DIR="$(pwd)"
PROCGEN_BUILD_DIR="$SLURM_TMPDIR/procgen-0.10.7-clean"
PROCGEN_WHEEL_DIR="$SLURM_TMPDIR/procgen-wheels"

mkdir -p "$PROCGEN_BUILD_DIR" "$PROCGEN_WHEEL_DIR"

git -C "$AIME_DIR/procgen-0.10.7-src" archive HEAD \
    | tar -xf - -C "$PROCGEN_BUILD_DIR"

# Compile a Python 3.11 wheel.
cd "$PROCGEN_BUILD_DIR"

python -m pip wheel \
    --no-index \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "$PROCGEN_WHEEL_DIR" \
    .

# Save a persistent copy and install it permanently in D2E.
mkdir -p "$AIME_DIR/procgen-wheels"

cp "$PROCGEN_WHEEL_DIR"/procgen-*.whl \
    "$AIME_DIR/procgen-wheels/"

python -m pip install \
    --no-index \
    --no-deps \
    --force-reinstall \
    "$PROCGEN_WHEEL_DIR"/procgen-*.whl

PROCGEN_BUILD_NOTES
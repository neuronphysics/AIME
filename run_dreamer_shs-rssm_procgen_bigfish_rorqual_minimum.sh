#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=01-00:59:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-procgen-bigfish_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-procgen-bigfish_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
set -euo pipefail

module --force purge

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a
module load opencv/4.10.0
module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow

unset CUDA_LAUNCH_BLOCKING

source /home/memole/links/D2E/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8
export CUDA_MODULE_LOADING=LAZY

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
export TORCH_LINALG_PREFER_CUSOLVER=1
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
cd /home/memole/links/projects/def-irina/memole/AIME

python - <<'PY'
import pathlib, shs_rssm
print("SHS package:", pathlib.Path(shs_rssm.__file__).resolve())
print("SHS version:", shs_rssm.__version__)
PY

: <<'PROCGEN_BUILD_NOTES'


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

source /home/memole/links/D2E/bin/activate

python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy; print('NumPy:', numpy.__version__)"

python -m pip install --no-index \
    "gym<1" \
    "filelock<4" \
    "imageio<3"

python -m pip install --no-index --no-deps "gym3==0.3.3"

python -c "import gym3; print('gym3:', gym3.__file__)"

export PROCGEN_CMAKE_PREFIX_PATH="$(qmake -query QT_INSTALL_PREFIX)/lib/cmake/Qt5"
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK:-8}"

echo "Qt path: $PROCGEN_CMAKE_PREFIX_PATH"

test -f "$PROCGEN_CMAKE_PREFIX_PATH/Qt5Config.cmake" \
    && echo "Qt5 configuration found"

AIME_DIR="$(pwd)"
PROCGEN_BUILD_DIR="$SLURM_TMPDIR/procgen-0.10.7-clean"
PROCGEN_WHEEL_DIR="$SLURM_TMPDIR/procgen-wheels"

mkdir -p "$PROCGEN_BUILD_DIR" "$PROCGEN_WHEEL_DIR"

git -C "$AIME_DIR/procgen-0.10.7-src" archive HEAD \
    | tar -xf - -C "$PROCGEN_BUILD_DIR"

cd "$PROCGEN_BUILD_DIR"

python -m pip wheel \
    --no-index \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "$PROCGEN_WHEEL_DIR" \
    .

mkdir -p "$AIME_DIR/procgen-wheels"

cp "$PROCGEN_WHEEL_DIR"/procgen-*.whl \
    "$AIME_DIR/procgen-wheels/"

python -m pip install \
    --no-index \
    --no-deps \
    --force-reinstall \
    "$PROCGEN_WHEEL_DIR"/procgen-*.whl

PROCGEN_BUILD_NOTES
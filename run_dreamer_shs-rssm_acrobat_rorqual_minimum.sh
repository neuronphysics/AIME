#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=23:59:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-run-acrobot-seed-1_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dreamerv3-shs-rssm-run-acrobot-seed-1_%N-%j.err
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

# acrobot-swingup: 1-D action (torque on the second joint), NOT cheetah's 6
python -u dreamer.py --configs dmc_acrobot_shs --compile False --steps 1000000 --shs_action_dim 1 --shs_learn_b0 True --shs_consolidate_every_episodes 50 --seed 0 --batch_size 96 --batch_length 64 --logdir ./logdir/dmc_acrobot_shs_s0

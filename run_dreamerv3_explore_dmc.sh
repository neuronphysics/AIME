#!/bin/bash
#SBATCH --job-name=DreamerV3-dmc
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=14:59:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/scratch/AIME/logs/dreamerv3-dmc-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dreamerv3-dmc-run-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#Start clean
module load StdEnv/2023
module load gcc/12.3

module load cuda/12.2

module load python/3.11
module load scipy-stack/2024a   # required by opencv/4.10.0

module load opencv/4.10.0

module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow


#virtualenv --no-download --clear /home/memole/D2E
source /home/memole/D2E/bin/activate
#export LD_LIBRARY_PATH=/usr/lib64/nvidia:${LD_LIBRARY_PATH}

### FORCE SINGLE-THREAD BLAS / OMP / TORCH
#export MUJOCO_GL=egl        # If using MuJoCo
#export PYOPENGL_PLATFORM=egl
#export MUJOCO_EGL_DEVICE_ID=0      
#export EGL_DEVICE_ID=0
export MUJOCO_GL=osmesa

###
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1

# keep Python from seeing ~/.local
export PYTHONNOUSERSITE=1

pip install "sentry-sdk>=2.0.0" "gitpython!=3.1.29,>=1.0.0"
python -m pip install "pydantic>=2,<3"
pip install fairscale
python -m pip install "tensorflow"
python -m pip install einx
python -m pip install "timm<1.0.0" --no-deps
python -m pip install dm_control==1.0.28
#python -m pip install git+https://github.com/richzhang/PerceptualSimilarity.git


#pip install -r requirements.txt --no-deps
echo "train dreamerv3 with AGAC exploration ....."

#CUDA_VISIBLE_DEVICES=0 python3 -m dreamerv3.dreamer --configs dmc_vision agac_recursive --task dmc_reacher_hard --logdir ./results/policy/runs/logdir/dmc_reacher_hard
CUDA_VISIBLE_DEVICES=0 python3 -m dreamerv3.dreamer \
  --configs dmc_vision \
  --task dmc_reacher_hard \
  --logdir ./results/policy/runs/logdir/dmc_reacher_hard

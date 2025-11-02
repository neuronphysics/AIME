#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00-13:59
#SBATCH --account=def-jhoey
#SBATCH --output=/home/memole/TEST/AIME/WorldModel-walker-walk-seed-1_%N-%j.out
#SBATCH --error=/home/memole/TEST/AIME/HalfCheetah-walker-walk-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install --upgrade pip
pip install dill --no-index
pip install expecttest
pip install einops
pip install wavio
pip install fairseq
pip install torchgan
pip install cloudpickle
pip install --upgrade tensorboard
pip install tensorboardX
pip install opencv-python

python WorldModel_D2E.py

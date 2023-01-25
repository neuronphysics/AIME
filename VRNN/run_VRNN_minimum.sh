#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00-10:59
#SBATCH --account=def-jhoey
#SBATCH --output=/home/memole/TEST/AIME/Hopper-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/TEST/AIME/Hopper-transit-run-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
pip install --upgrade pip
pip install dill --no-index
pip install torch-optimizer --no-index
#pip install git+https://github.com/WarrenWeckesser/wavio.git
pip install git+https://github.com/cooper-org/cooper.git
pip install --no-index wandb
#pip install torch-geometric
wandb offline
tensorboard --logdir=./tensorlog --port 6006 --bind_all --load_fast false &
#python DataCollectionD2E.py --env_name=HalfCheetah-v2
#python train_eval_offline_D2E.py --env_name=HalfCheetah-v2
python training_vrnn.py
#python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae
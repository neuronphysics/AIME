#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00-13:59
#SBATCH --account=def-jhoey
#SBATCH --output=/home/memole/TEST/HalfCheetah-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/TEST/HalfCheetah-run-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install --upgrade pip
pip install dill --no-index

#python DataCollectionD2E.py --env_name=HalfCheetah-v2
#python train_eval_offline_D2E.py --env_name=HalfCheetah-v2
python train_D2E_eval_online.py --env_name=HalfCheetah-v2 --n_eval_episodes 25 --eval_target 10000 --total_train_steps 100000
#python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae
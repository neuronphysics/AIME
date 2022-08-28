#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=00-02:00
#SBATCH --account=def-jhoey
#SBATCH --output=../brac-run-seed-1_%N-%j.out
#SBATCH --error=../brac-run-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install --upgrade pip
pip install dill --no-index

#python DataCollection.py --env_name=Pendulum-v0
python train_eval_offline.py --env_name=Pendulum-v0

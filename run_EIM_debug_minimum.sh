#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=00-06:59
#SBATCH --account=def-jhoey
#SBATCH --output=../EIM-run-debug-seed-1_%N-%j.out
#SBATCH --error=../EIM-run-debug-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/memole/TEST/AIME/start-with-brac"
pip install --no-index --upgrade pip

pip install --no-index absl-py
python EIM.py

#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00-23:59
#SBATCH --account=def-jhoey
#SBATCH --output=../cheetah-run-seed-1_%N-%j.out
#SBATCH --error=../cheetah-run-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate

python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae  

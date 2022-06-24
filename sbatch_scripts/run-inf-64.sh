#!/bin/bash
#SBATCH --time=00-71:59
#SBATCH --account=def-jhoey
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=cheetah-run-seed-2-inf-64-long.out

module load python/3.8.10
cd ..
source ~/dm_control/bin/activate
python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --max-episode-length 500 --hidden-size 16 --state-size 10
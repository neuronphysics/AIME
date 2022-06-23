#!/bin/bash
#SBATCH --time=00-23:59
#SBATCH --account=def-jhoey
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=cheetah-run-v100-seed-1-reg-2.out

module load python/3.8.10
cd AIME
source ~/dm_control/bin/activate
python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae --max-episode-length 500
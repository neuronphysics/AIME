#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --constraint="dgx&ampere"
#SBATCH --mem=48G
#SBATCH --time=1-4:59:59
#SBATCH -o /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/slurm-adpgmm-dmc_vb-train-RIM-LSTM-%j.out
#SBATCH -e /home/mila/z/zahra.sheikhbahaee/scratch/AIME/logs/slurm-adpgmm-dmc_vb-train-RIM-LSTM-%j.err
# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
module unload python
module load anaconda/3
conda activate trifinger_rl_venv
module load gcc/9.3.0
module unload anaconda
module load python/3.10

CURRENT_PATH=`pwd`

CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.dmc_vb_transition_dynamics_trainer --data_dir /home/mila/z/zahra.sheikhbahaee/scratch/transition_datasets/transition_data/
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=2 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=24G      
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-output/1-GPU-%N-%j.out
#SBATCH --account=def-phumane
# --mail-user=prateek.humane@gmail.com
# --mail-type=ALL

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
pip install tensorboard scikit-learn matplotlib triton

echo "starting training..."
time python train_dpgmm_vae_CC.py 
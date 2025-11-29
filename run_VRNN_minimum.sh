#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100_80gb:1 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01-20:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/scratch/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc/12.3 python/3.11 mujoco/3.3.0 cuda/12.2
module load StdEnv/2023  
module load mpi4py/3.1.6
module load arrow
module load opencv/4.10.0
module load scipy-stack/2024a
module load imkl/2025.2.0
module load rust
module load cmake
module load intel


#virtualenv --no-download --clear /home/memole/D2E
source /home/memole/D2E/bin/activate
#pip install -r requirements.txt --no-deps
#gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/medium /home/memole/scratch/AIME/transition_data/dmc_vb/humanoid_walk
echo "pretrain VQVAE ....."
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.pretrain_vqvae
echo "finished pretraining and start training world model dpgmm vrnn model... " 
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.dmc_vb_transition_dynamics_trainer  

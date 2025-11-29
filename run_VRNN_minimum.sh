#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01-20:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/scratch/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/scratch/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#Start clean
module load StdEnv/2023
module load gcc/12.3

module load cuda/12.2

module load python/3.11
module load scipy-stack/2024a   # required by opencv/4.10.0

module load opencv/4.10.0

module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow

#virtualenv --no-download --clear /home/memole/D2E
source /home/memole/D2E/bin/activate
pip install "sentry-sdk>=2.0.0" "gitpython!=3.1.29,>=1.0.0"
python -m pip install "pydantic>=2,<3"
pip install fairscale
#pip install tensorflow
#pip install --user einx
#pip install -r requirements.txt --no-deps
#gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/medium /home/memole/scratch/AIME/transition_data/dmc_vb/humanoid_walk
echo "pretrain VQVAE ....."
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.pretrain_vqvae
echo "finished pretraining and start training world model dpgmm vrnn model... " 
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.dmc_vb_transition_dynamics_trainer  

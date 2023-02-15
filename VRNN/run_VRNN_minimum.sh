#!/bin/bash
#
#SBATCH --nodes=6
#SBATCH --gres=gpu:v100:4 # Request 4 GPU "generic resources”.
#SBATCH --ntasks-per-node=4 # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02-23:59
#SBATCH --account=def-jhoey
#SBATCH --output=/home/memole/TEST/AIME/Hopper-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/TEST/AIME/Hopper-transit-run-seed-1_%N-%j.err
module load StdEnv/2020
module load python/3.8.10
module load scipy-stack
source /home/memole/dm_control/bin/activate



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/memole/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
pip install --upgrade pip
pip install dill --no-index
pip install einops --no-index
pip install torch-optimizer --no-index
#pip install git+https://github.com/WarrenWeckesser/wavio.git
pip install git+https://github.com/cooper-org/cooper.git
#pip install git+https://github.com/facebookresearch/fairseq.git
pip install --no-index wandb
#pip install torch-geometric
wandb offline
tensorboard --logdir=./tensorlog --port 6006 --bind_all --load_fast false &
#python DataCollectionD2E.py --env_name=HalfCheetah-v2
#python train_eval_offline_D2E.py --env_name=HalfCheetah-v2
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun python training_vrnn.py  --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS  --batch_size 35 --max_epochs 600

#python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae
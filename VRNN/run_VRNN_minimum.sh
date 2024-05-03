#!/bin/bash
#
#SBATCH --nodes=7
#SBATCH --gres=gpu:v100:4 # Request 4 GPU "generic resources”.
#SBATCH --ntasks-per-node=4 # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=01-15:59
#SBATCH --account=def-jhoey
#SBATCH --output=Hopper-transit-run-seed-1_%N-%j.out
#SBATCH --error=Hopper-transit-run-seed-1_%N-%j.err
source /home/sstevec/vir_env/gen_env/bin/activate
module load StdEnv/2020 gcc/9.3.0 opencv/4.7.0
module load mujoco
module load scipy-stack/2022a
module load python/3.10

#python DataCollectionD2E.py --env_name=HalfCheetah-v2
#python train_eval_offline_D2E.py --env_name=HalfCheetah-v2
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun python training_vrnn.py  --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS  --batch_size 35 --max_epochs 750

#python main.py --id cheetah-run-seed-1 --seed 1 --env cheetah-run --use-regular-vae
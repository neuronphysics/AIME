#!/bin/bash
#SBATCH --job-name=aime-world-model
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00-13:59
#SBATCH --output=logs/world-model-%N-%j.out
#SBATCH --error=logs/world-model-%N-%j.err

# ============================================================================
# AIME World Model Training Script
#
# This script trains the complete DPGMM-VRNN world model using the new
# modular structure (src/).
#
# Usage:
#   sbatch scripts/train_world_model.sh
#
# Environment variables you can set:
#   AIME_ENV_NAME: Name of conda/virtualenv environment (default: dm_control)
#   AIME_DOMAIN: DMC domain (default: walker)
#   AIME_TASK: DMC task (default: walk)
#   AIME_SEED: Random seed (default: 1)
# ============================================================================

# Load modules (adjust for your cluster)
module load StdEnv/2020 python/3.8.10 scipy-stack

# Activate Python environment
AIME_ENV_NAME=${AIME_ENV_NAME:-"dm_control"}
source ~/envs/${AIME_ENV_NAME}/bin/activate

# Set MuJoCo paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Configure environment
AIME_DOMAIN=${AIME_DOMAIN:-"walker"}
AIME_TASK=${AIME_TASK:-"walk"}
AIME_SEED=${AIME_SEED:-1}

# Print configuration
echo "=========================================="
echo "AIME World Model Training"
echo "=========================================="
echo "Domain: ${AIME_DOMAIN}"
echo "Task: ${AIME_TASK}"
echo "Seed: ${AIME_SEED}"
echo "Python: $(which python)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Train world model using new structure
python -m src.world_model.train \
    --domain ${AIME_DOMAIN} \
    --task ${AIME_TASK} \
    --seed ${AIME_SEED} \
    --num_epochs 500 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --save_dir checkpoints/${AIME_DOMAIN}-${AIME_TASK}-seed-${AIME_SEED}

echo "Training complete!"

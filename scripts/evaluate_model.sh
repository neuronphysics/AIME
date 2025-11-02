#!/bin/bash
#SBATCH --job-name=aime-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00-02:00
#SBATCH --output=logs/eval-%N-%j.out
#SBATCH --error=logs/eval-%N-%j.err

# ============================================================================
# AIME Model Evaluation Script
#
# This script evaluates a trained world model on DMC tasks.
#
# Usage:
#   sbatch scripts/evaluate_model.sh /path/to/checkpoint.pt
#
# Environment variables:
#   AIME_ENV_NAME: Environment name (default: dm_control)
#   AIME_DOMAIN: DMC domain (default: walker)
#   AIME_TASK: DMC task (default: walk)
#   AIME_NUM_EPISODES: Number of evaluation episodes (default: 100)
# ============================================================================

# Check if checkpoint path provided
if [ -z "$1" ]; then
    echo "Error: Please provide checkpoint path"
    echo "Usage: sbatch scripts/evaluate_model.sh /path/to/checkpoint.pt"
    exit 1
fi

CHECKPOINT_PATH=$1

# Load modules
module load StdEnv/2020 python/3.8.10 scipy-stack

# Activate environment
AIME_ENV_NAME=${AIME_ENV_NAME:-"dm_control"}
source ~/envs/${AIME_ENV_NAME}/bin/activate

# Set MuJoCo paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Configure
AIME_DOMAIN=${AIME_DOMAIN:-"walker"}
AIME_TASK=${AIME_TASK:-"walk"}
AIME_NUM_EPISODES=${AIME_NUM_EPISODES:-100}

echo "=========================================="
echo "AIME Model Evaluation"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Domain: ${AIME_DOMAIN}"
echo "Task: ${AIME_TASK}"
echo "Episodes: ${AIME_NUM_EPISODES}"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Run evaluation
python -m src.world_model.evaluate \
    --checkpoint ${CHECKPOINT_PATH} \
    --domain ${AIME_DOMAIN} \
    --task ${AIME_TASK} \
    --num_episodes ${AIME_NUM_EPISODES} \
    --save_videos true \
    --video_dir videos/eval-$(basename ${CHECKPOINT_PATH} .pt)

echo "Evaluation complete!"

#!/bin/bash
#
# Tier 1: Ultra-Fast Debug Script (< 1 min/epoch)
# Use: Quick bug finding, gradient diagnostics, feature testing
# Environment: Cartpole (10x faster than humanoid)
#

set -e

# Experiment config
ENV_NAME="${AIME_ENV_NAME:-aime_env}"
DOMAIN="${AIME_DOMAIN:-cartpole}"
TASK="${AIME_TASK:-swingup}"
SEED="${AIME_SEED:-1}"
DEBUG="${AIME_DEBUG:-0}"

# Fast debug hyperparameters
BATCH_SIZE=16
VIDEO_LENGTH=10      # Shortened from 50
IMG_SIZE=64          # Reduced from 84
LATENT_DIM=18        # Reduced from 36
HIDDEN_DIM=256       # Reduced from 512
MAX_COMPONENTS=8     # Reduced from 15
NUM_EPOCHS=50
LR=3e-4

# Paths
EXP_NAME="${DOMAIN}-${TASK}-tier1-seed${SEED}"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-aime-tier1-debug}"

# CUDA settings for debugging
if [ "$DEBUG" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    echo "🐛 Debug mode enabled: CUDA_LAUNCH_BLOCKING=1"
fi

echo "=========================================="
echo "🚀 Tier 1: Ultra-Fast Debug Training"
echo "=========================================="
echo "Environment: ${DOMAIN}-${TASK}"
echo "Video shape: (${VIDEO_LENGTH}, 3, ${IMG_SIZE}, ${IMG_SIZE})"
echo "Model: latent_dim=${LATENT_DIM}, hidden_dim=${HIDDEN_DIM}, max_components=${MAX_COMPONENTS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# Create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}

# Run training
CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} python -m VRNN.run_perceiver_io_dmc_vb \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --seed ${SEED} \
    --video_length ${VIDEO_LENGTH} \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --context_dim $((HIDDEN_DIM / 2)) \
    --max_components ${MAX_COMPONENTS} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${EXP_NAME} \
    --save_every 10 \
    --eval_every 5 \
    --log_every 10 \
    --num_workers 4 \
    "$@"

echo ""
echo "✅ Tier 1 training complete!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

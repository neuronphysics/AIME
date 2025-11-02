#!/bin/bash
#
# Tier 1: Ultra-Fast Debug Script (< 1 min/epoch)
# Use: Quick bug finding, gradient diagnostics, feature testing
# Environment: Cartpole (10x faster than humanoid)
#
# Prerequisites:
#   1. Collect data first: bash scripts/collect_tier1_data.sh
#   2. Or use existing HDF5 file with --data_path
#

set -e

# Experiment config
ENV_NAME="${AIME_ENV_NAME:-aime_env}"
DOMAIN="${AIME_DOMAIN:-cartpole}"
TASK="${AIME_TASK:-swingup}"
SEED="${AIME_SEED:-1}"
DEBUG="${AIME_DEBUG:-0}"
DATA_PATH="${AIME_DATA_PATH:-data/tier1/${DOMAIN}_${TASK}.hdf5}"

# Fast debug hyperparameters
BATCH_SIZE=16
SEQUENCE_LENGTH=10   # Shortened from 50
IMG_SIZE=64          # Reduced from 84
NUM_EPOCHS=50
LR=3e-4

# Model hyperparameters (smaller for Tier 1)
NUM_LATENTS=128
NUM_LATENT_CHANNELS=256
NUM_ENCODER_LAYERS=2
NUM_ATTENTION_HEADS=4
CODE_DIM=128
NUM_CODES=512
DOWNSAMPLE=4
BASE_CHANNELS=32

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
echo "Data: ${DATA_PATH}"
echo "Sequence length: ${SEQUENCE_LENGTH}"
echo "Image size: ${IMG_SIZE}×${IMG_SIZE}"
echo "Model: ${NUM_LATENTS} latents, ${NUM_LATENT_CHANNELS} channels"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "=========================================="

# Check if data exists
if [ ! -f "${DATA_PATH}" ]; then
    echo ""
    echo "❌ Data file not found: ${DATA_PATH}"
    echo ""
    echo "Please collect data first:"
    echo "  bash scripts/collect_tier1_data.sh"
    echo ""
    echo "Or specify data path:"
    echo "  AIME_DATA_PATH=/path/to/data.hdf5 bash $0"
    exit 1
fi

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# Create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}

# Run training (use new Perceiver-only training script)
python -m legacy.VRNN.run_perceiver_io_dmc_vb \
    --base_path $(dirname ${DATA_PATH}) \
    --task ${DOMAIN}_${TASK} \
    --sequence_length ${SEQUENCE_LENGTH} \
    --img_height ${IMG_SIZE} \
    --img_width ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --num_latents ${NUM_LATENTS} \
    --num_latent_channels ${NUM_LATENT_CHANNELS} \
    --num_encoder_layers ${NUM_ENCODER_LAYERS} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --code_dim ${CODE_DIM} \
    --num_codes ${NUM_CODES} \
    --downsample ${DOWNSAMPLE} \
    --base_channels ${BASE_CHANNELS} \
    --out_dir ${CHECKPOINT_DIR} \
    --seed ${SEED} \
    --run_name ${EXP_NAME} \
    --num_workers 4 \
    --save_every 10 \
    --vis_every 10 \
    $([ "$DEBUG" = "1" ] && echo "--use_wandb" || echo "") \
    "$@"

echo ""
echo "✅ Tier 1 training complete!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

#!/bin/bash
#
# Tier 2: Medium Validation Script (5-10 min/epoch)
# Use: Validating fixes at scale, hyperparameter tuning
# Environment: Reacher (8x faster than humanoid)
#

set -e

# Experiment config
ENV_NAME="${AIME_ENV_NAME:-aime_env}"
DOMAIN="${AIME_DOMAIN:-reacher}"
TASK="${AIME_TASK:-easy}"
SEED="${AIME_SEED:-1}"

# Medium validation hyperparameters
BATCH_SIZE=32
VIDEO_LENGTH=20      # Moderate length
IMG_SIZE=84          # Standard resolution
LATENT_DIM=36        # Full latent
HIDDEN_DIM=512       # Full hidden
MAX_COMPONENTS=15    # Full DPGMM
NUM_EPOCHS=200
LR=2e-4

# Paths
EXP_NAME="${DOMAIN}-${TASK}-tier2-seed${SEED}"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-aime-tier2-validation}"

echo "=========================================="
echo "🔬 Tier 2: Medium Validation Training"
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

# Run training with standard settings
python -m VRNN.run_perceiver_io_dmc_vb \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --seed ${SEED} \
    --video_length ${VIDEO_LENGTH} \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --context_dim 256 \
    --max_components ${MAX_COMPONENTS} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${EXP_NAME} \
    --save_every 20 \
    --eval_every 10 \
    --log_every 50 \
    --num_workers 8 \
    --use_amp \
    "$@"

echo ""
echo "✅ Tier 2 training complete!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

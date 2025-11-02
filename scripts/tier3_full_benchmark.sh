#!/bin/bash
#
# Tier 3: Full Benchmark Script (production settings)
# Use: Final validation before publication/deployment
# Environment: Humanoid (full complexity)
#

set -e

# Experiment config
ENV_NAME="${AIME_ENV_NAME:-aime_env}"
DOMAIN="${AIME_DOMAIN:-humanoid}"
TASK="${AIME_TASK:-walk}"
SEED="${AIME_SEED:-1}"

# Full benchmark hyperparameters (as designed)
BATCH_SIZE=32
VIDEO_LENGTH=50      # Full sequence length
IMG_SIZE=84          # Standard DMC vision benchmark
LATENT_DIM=36
HIDDEN_DIM=512
CONTEXT_DIM=256
MAX_COMPONENTS=15
NUM_EPOCHS=500
LR=2e-4

# Paths
EXP_NAME="${DOMAIN}-${TASK}-tier3-seed${SEED}"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-aime-tier3-benchmark}"

echo "=========================================="
echo "🎯 Tier 3: Full Benchmark Training"
echo "=========================================="
echo "Environment: ${DOMAIN}-${TASK}"
echo "Video shape: (${VIDEO_LENGTH}, 3, ${IMG_SIZE}, ${IMG_SIZE})"
echo "Model: latent_dim=${LATENT_DIM}, hidden_dim=${HIDDEN_DIM}, max_components=${MAX_COMPONENTS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "This will take several hours/days depending on hardware"
echo "=========================================="

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

# Create checkpoint directory
mkdir -p ${CHECKPOINT_DIR}

# Run full training with all optimizations
python -m VRNN.run_perceiver_io_dmc_vb \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --seed ${SEED} \
    --video_length ${VIDEO_LENGTH} \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --context_dim ${CONTEXT_DIM} \
    --max_components ${MAX_COMPONENTS} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${EXP_NAME} \
    --save_every 50 \
    --eval_every 25 \
    --log_every 100 \
    --num_workers 8 \
    --use_amp \
    --gradient_accumulation_steps 2 \
    "$@"

echo ""
echo "✅ Tier 3 training complete!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"

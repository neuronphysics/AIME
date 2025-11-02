#!/bin/bash
#
# Collect Tier 2 data (reacher, medium validation)
#

set -e

DOMAIN="${AIME_DOMAIN:-reacher}"
TASK="${AIME_TASK:-easy}"
NUM_EPISODES="${AIME_NUM_EPISODES:-200}"
IMG_SIZE=84
OUTPUT_DIR="data/tier2"

echo "Collecting Tier 2 data: ${DOMAIN}-${TASK}"
echo "Episodes: ${NUM_EPISODES}"
echo "Image size: ${IMG_SIZE}×${IMG_SIZE}"

mkdir -p ${OUTPUT_DIR}

MUJOCO_GL=egl python scripts/collect_dmc_data.py \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps 500 \
    --img_size ${IMG_SIZE} \
    --output ${OUTPUT_DIR}/${DOMAIN}_${TASK}.hdf5

echo ""
echo "✅ Tier 2 data ready!"
echo "Run training with: bash scripts/tier2_medium_validation.sh"

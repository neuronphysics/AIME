#!/bin/bash
#
# Collect Tier 1 data (cartpole, fast iteration)
#

set -e

DOMAIN="${AIME_DOMAIN:-cartpole}"
TASK="${AIME_TASK:-swingup}"
NUM_EPISODES="${AIME_NUM_EPISODES:-50}"
IMG_SIZE=64
OUTPUT_DIR="data/tier1"

echo "Collecting Tier 1 data: ${DOMAIN}-${TASK}"
echo "Episodes: ${NUM_EPISODES}"
echo "Image size: ${IMG_SIZE}×${IMG_SIZE}"

mkdir -p ${OUTPUT_DIR}

MUJOCO_GL=egl python scripts/collect_dmc_data.py \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps 200 \
    --img_size ${IMG_SIZE} \
    --output ${OUTPUT_DIR}/${DOMAIN}_${TASK}.hdf5

echo ""
echo "✅ Tier 1 data ready!"
echo "Run training with: bash scripts/tier1_ultrafast_debug.sh"

#!/bin/bash
#
# Collect Tier 3 data (humanoid, full benchmark)
#

set -e

DOMAIN="${AIME_DOMAIN:-humanoid}"
TASK="${AIME_TASK:-walk}"
NUM_EPISODES="${AIME_NUM_EPISODES:-1000}"
IMG_SIZE=84
OUTPUT_DIR="data/tier3"

echo "Collecting Tier 3 data: ${DOMAIN}-${TASK}"
echo "Episodes: ${NUM_EPISODES}"
echo "Image size: ${IMG_SIZE}×${IMG_SIZE}"
echo "⚠️  This will take a while..."

mkdir -p ${OUTPUT_DIR}

MUJOCO_GL=egl python scripts/collect_dmc_data.py \
    --domain ${DOMAIN} \
    --task ${TASK} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps 1000 \
    --img_size ${IMG_SIZE} \
    --output ${OUTPUT_DIR}/${DOMAIN}_${TASK}.hdf5

echo ""
echo "✅ Tier 3 data ready!"
echo "Run training with: bash scripts/tier3_full_benchmark.sh"

#!/bin/bash
#SBATCH --job-name=VRNN-MetaWorld
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00-12:00:00
#SBATCH --output=logs/vrnn-metaworld_%N-%j.out
#SBATCH --error=logs/vrnn-metaworld_%N-%j.err

set -euo pipefail

# ============================================================================
# VRNN World Model Training on Meta-World
# ============================================================================
# This script trains the DPGMM-VRNN world model on Meta-World robotic
# manipulation environments.
#
# Prerequisites:
#   1. Collect data first using:
#      python -m VRNN.collect_metaworld_data --task reach-v3 --episodes 1000
#
#   2. Install metaworld:
#      pip install metaworld
# ============================================================================

# Configuration
TASK_NAME="${TASK_NAME:-reach-v3}"
DATA_DIR="${DATA_DIR:-./transition_data}"
POLICY_LEVEL="${POLICY_LEVEL:-random}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
LEARNING_RATE="${LEARNING_RATE:-0.0007}"

# GPU settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# Limit threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=========================================="
echo "VRNN Training on Meta-World"
echo "=========================================="
echo "Task: ${TASK_NAME}"
echo "Data directory: ${DATA_DIR}"
echo "Policy level: ${POLICY_LEVEL}"
echo "Epochs: ${N_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "=========================================="

# Check if data exists
if [ ! -d "${DATA_DIR}/metaworld/${TASK_NAME}" ]; then
    echo "Warning: Data directory not found: ${DATA_DIR}/metaworld/${TASK_NAME}"
    echo "You may need to collect data first using:"
    echo "  python -m VRNN.collect_metaworld_data --task ${TASK_NAME} --episodes 1000 --output_dir ${DATA_DIR}/metaworld"
fi

cd "$(dirname "$0")"

# Train VRNN
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.train_metaworld_vrnn \
    --task_name "${TASK_NAME}" \
    --data_dir "${DATA_DIR}" \
    --policy_level "${POLICY_LEVEL}" \
    --n_epochs "${N_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --sequence_length "${SEQUENCE_LENGTH}" \
    --learning_rate "${LEARNING_RATE}"

echo "Training complete!"

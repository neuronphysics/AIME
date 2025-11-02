#!/bin/bash
# Monitor training progress

LOG_FILE="${1:-tier1_long_training.log}"

echo "=========================================="
echo "Training Progress Monitor"
echo "=========================================="
echo ""

# Get latest epoch stats
echo "Last 5 Epoch Summaries:"
grep -E "^Epoch [0-9]+/[0-9]+:" "$LOG_FILE" | tail -5

echo ""
echo "=========================================="
echo "Training curves (visualize externally):"
echo "  Loss:     $(grep -oP 'Loss: \K[0-9.]+' "$LOG_FILE" | tail -10 | tr '\n' ', ')"
echo "  Accuracy: $(grep -oP 'Accuracy: \K[0-9.]+%' "$LOG_FILE" | tail -10 | tr '\n' ', ')"
echo "=========================================="
echo ""
echo "To monitor in real-time: tail -f $LOG_FILE | grep 'Epoch [0-9]'"
echo "Target: Train Acc >90%, Test Acc >80%"

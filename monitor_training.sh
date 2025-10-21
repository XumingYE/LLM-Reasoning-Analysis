#!/bin/bash
# Monitor the training progress

echo "=== DeBERTa-v3-large Training Monitor ==="
echo "Latest log entries:"
echo ""
tail -30 training_deberta_large.log
echo ""
echo "=== Training Process Status ==="
ps aux | grep "train_deberta_with_features.py" | grep -v grep
echo ""
echo "=== GPU Status (cuda:1) ==="
nvidia-smi --id=1 --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

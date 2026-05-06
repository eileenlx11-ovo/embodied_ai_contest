#!/bin/bash
set -e

export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Config can be overridden at `docker run` time:
#   docker run ... -e CONFIG=configs/imagenet_resnet50.yaml embodied-ai
CONFIG="${CONFIG:-configs/imagenet_resnet50.yaml}"

echo "=== Config: $CONFIG ==="

echo "=== Step 1: Training ==="
python scripts/train.py --config "$CONFIG"

echo "=== Step 2: Validation ==="
python scripts/evaluate.py --checkpoint checkpoints/best.pth --config "$CONFIG"

echo "=== Step 3: Prediction (test set) ==="
python scripts/predict.py --checkpoint checkpoints/best.pth --config "$CONFIG" --output submit/result.csv

echo "=== Step 4: CSV Validation ==="
python scripts/validate_csv.py --csv submit/result.csv

echo "=== Done ==="

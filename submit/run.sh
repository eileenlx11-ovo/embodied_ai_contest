#!/bin/bash
set -e

export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

echo "=== Step 1: Training ==="
python scripts/train.py --config configs/final_config.yaml

echo "=== Step 2: Validation ==="
python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/final_config.yaml

echo "=== Step 3: Prediction (test set) ==="
python scripts/predict.py --checkpoint checkpoints/best.pth --config configs/final_config.yaml --output submit/result.csv

echo "=== Step 4: CSV Validation ==="
python scripts/validate_csv.py --csv submit/result.csv

echo "=== Done ==="

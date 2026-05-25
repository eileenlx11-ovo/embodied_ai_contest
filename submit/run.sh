#!/bin/bash
set -e

export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Overridable env vars (docker run -e KEY=value ...):
#   MODE     full | predict | eval     (default: full)
#   CONFIG   path to yaml             (default: configs/imagenet_resnet50.yaml)
#   CKPT     path to .pth checkpoint  (default: checkpoints/best.pth)
#   TTA      none | hflip             (default: hflip; predict step only)
#   OUTPUT   result.csv path          (default: submit/result.csv)
MODE="${MODE:-full}"
CONFIG="${CONFIG:-configs/imagenet_resnet50.yaml}"
CKPT="${CKPT:-checkpoints/best.pth}"
TTA="${TTA:-hflip}"
OUTPUT="${OUTPUT:-submit/result.csv}"

echo "=== MODE: $MODE | CONFIG: $CONFIG | CKPT: $CKPT | TTA: $TTA ==="

if [ "$MODE" = "full" ]; then
    echo "=== Step 1: Training ==="
    python scripts/train.py --config "$CONFIG"
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "eval" ]; then
    echo "=== Step 2: Validation ==="
    python scripts/evaluate.py --checkpoint "$CKPT" --config "$CONFIG"
fi

echo "=== Step 3: Prediction (test set) ==="
python scripts/predict.py --checkpoint "$CKPT" --config "$CONFIG" --tta "$TTA" --output "$OUTPUT"

echo "=== Step 4: CSV Validation ==="
python scripts/validate_csv.py --csv "$OUTPUT"

echo "=== Done ==="

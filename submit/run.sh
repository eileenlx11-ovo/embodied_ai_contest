#!/bin/bash
# Reproduce the submitted ResNet-50 result on ImageNet-1K (noisy-label track).
# Pipeline: train (from scratch) -> validate -> predict (HFlip TTA) -> check CSV.
set -e

export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Submitted recipe (full data + Mixup/CutMix). Override CONFIG/CKPT_DIR only to re-run a variant.
CONFIG="${CONFIG:-configs/imagenet_resnet50_full_mixcut.yaml}"
CKPT_DIR="${CKPT_DIR:-checkpoints_full_mixcut}"

# Dataset location. Mount ImageNet-1K (with train/ val/ test/ subdirs) and point
# DATA_ROOT at it, e.g.  docker run --gpus all -e DATA_ROOT=/data/imagenet ...
DATA_ROOT="${DATA_ROOT:-/data/imagenet}"
RUN_CONFIG="configs/_runtime.yaml"
sed "s#^  root:.*#  root: ${DATA_ROOT}#" "$CONFIG" > "$RUN_CONFIG"
echo "=== Config: $RUN_CONFIG  (data.root -> ${DATA_ROOT}) ==="

echo "=== Step 1/4: Training (ResNet-50, full ImageNet-1K, 100 epochs) ==="
python scripts/train.py --config "$RUN_CONFIG"

echo "=== Step 2/4: Validation (top-1 on val set) ==="
python scripts/evaluate.py --checkpoint "${CKPT_DIR}/best.pth" --config "$RUN_CONFIG"

echo "=== Step 3/4: Prediction on test set (HFlip TTA) ==="
python scripts/predict.py --checkpoint "${CKPT_DIR}/best.pth" --config "$RUN_CONFIG" --output submit/result.csv

echo "=== Step 4/4: CSV format validation ==="
python scripts/validate_csv.py --csv submit/result.csv

echo "=== Done. Submission file: submit/result.csv ==="

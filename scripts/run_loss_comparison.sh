#!/bin/bash
# Loss comparison experiments for Week 3
# Run on AutoDL with RTX 4090 (or any GPU with >=8GB VRAM)
# Usage: bash scripts/run_loss_comparison.sh

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
LOG_DIR=./logs/week3_loss_comparison
mkdir -p $LOG_DIR

echo "=== Loss Comparison Experiments ==="
echo "Data: 10% subset | Model: ResNet-50 | Epochs: 30 | LR: 0.1"
echo "Start: $(date)"
echo ""

# 1. CE + Label Smoothing 0.1 (baseline)
echo "[1/5] CE + LS(0.1)..."
$PYTHON scripts/train.py --config configs/imagenet_resnet50.yaml \
    --subset 0.1 --epochs 30 --warmup-epochs 3 --eval-interval 5 \
    > $LOG_DIR/01_ce_ls01.out 2>&1
tail -1 $LOG_DIR/01_ce_ls01.out

# 2. SCE (alpha=0.1, beta=1.0)
echo "[2/5] SCE a=0.1 b=1.0..."
$PYTHON scripts/train.py --config configs/imagenet_resnet50.yaml \
    --subset 0.1 --epochs 30 --warmup-epochs 3 --eval-interval 5 \
    --loss sce --sce-alpha 0.1 --sce-beta 1.0 \
    > $LOG_DIR/02_sce_a01_b10.out 2>&1
tail -1 $LOG_DIR/02_sce_a01_b10.out

# 3. SCE (alpha=0.5, beta=0.5)
echo "[3/5] SCE a=0.5 b=0.5..."
$PYTHON scripts/train.py --config configs/imagenet_resnet50.yaml \
    --subset 0.1 --epochs 30 --warmup-epochs 3 --eval-interval 5 \
    --loss sce --sce-alpha 0.5 --sce-beta 0.5 \
    > $LOG_DIR/03_sce_a05_b05.out 2>&1
tail -1 $LOG_DIR/03_sce_a05_b05.out

# 4. GCE (q=0.7)
echo "[4/5] GCE q=0.7..."
$PYTHON scripts/train.py --config configs/imagenet_resnet50.yaml \
    --subset 0.1 --epochs 30 --warmup-epochs 3 --eval-interval 5 \
    --loss gce --gce-q 0.7 \
    > $LOG_DIR/04_gce_q07.out 2>&1
tail -1 $LOG_DIR/04_gce_q07.out

# 5. ELR (beta=0.9, lam=3.0)
echo "[5/5] ELR b=0.9 lam=3.0..."
$PYTHON scripts/train.py --config configs/imagenet_resnet50.yaml \
    --subset 0.1 --epochs 30 --warmup-epochs 3 --eval-interval 5 \
    --loss elr \
    > $LOG_DIR/05_elr_b09_l30.out 2>&1
tail -1 $LOG_DIR/05_elr_b09_l30.out

echo ""
echo "=== Summary ==="
for f in $LOG_DIR/*.out; do
    name=$(basename $f .out)
    best=$(grep "Best Val" $f | tail -1)
    echo "  $name: $best"
done
echo "End: $(date)"

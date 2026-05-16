#!/bin/bash
set -e
cd /root/autodl-tmp

echo "$(date): Starting val fix"

# Step 1: Remove broken val symlinks
rm -rf /root/autodl-tmp/imagenet_10p/val

# Step 2: Download the zip again
echo "Downloading ImageNet zip..."
/root/miniconda3/bin/kaggle competitions download -c imagenet-object-localization-challenge -p /root/autodl-tmp/

# Step 3: Extract only val + CSV
echo "Extracting val images..."
unzip -qo /root/autodl-tmp/imagenet-object-localization-challenge.zip "ILSVRC/Data/CLS-LOC/val/*" "LOC_val_solution.csv" -d /root/autodl-tmp/val_extract/

# Step 4: Organize val into class folders
echo "Organizing val into class folders..."
/root/miniconda3/bin/python /root/autodl-tmp/organize_val.py

# Step 5: Update symlink
rm -f /root/autodl-tmp/imagenet_full/ILSVRC/Data/CLS-LOC/val
ln -sf /root/autodl-tmp/imagenet_10p/val /root/autodl-tmp/imagenet_full/ILSVRC/Data/CLS-LOC/val

# Step 6: Delete zip and temp
rm -f /root/autodl-tmp/imagenet-object-localization-challenge.zip
rm -rf /root/autodl-tmp/val_extract

# Step 7: Verify
echo "Verifying..."
VAL_CLASSES=$(ls /root/autodl-tmp/imagenet_10p/val/ | wc -l)
VAL_IMAGES=$(find /root/autodl-tmp/imagenet_10p/val/ -type f | wc -l)
echo "Val: $VAL_CLASSES classes, $VAL_IMAGES images"

# Step 8: Restart training
echo "Restarting training with --resume..."
cd /root/autodl-tmp/embodied_ai_contest
nohup /root/miniconda3/bin/python scripts/train.py --config configs/imagenet_resnet50.yaml --epochs 100 --warmup-epochs 5 --eval-interval 5 --resume checkpoints/latest.pth > logs/full_imagenet_1.28M_100ep.out 2>&1 &

echo "$(date): Val fix complete, training resumed"

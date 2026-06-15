"""
C2: Teacher-loss based sample filtering.

Uses the baseline_v1 checkpoint (77.53% EMA model) to compute per-sample
CE loss on the full training set (with val-style transforms, no augmentation).
Outputs the top-K% highest-loss samples as an exclusion list.

Usage:
  python scripts/clean_c2_teacher_loss.py \
      --config configs/imagenet_resnet50.yaml \
      --checkpoint checkpoints/best.pth \
      --data-root /root/autodl-tmp/imagenet_full/ILSVRC/Data/CLS-LOC \
      --top-pct 10 \
      --output data/exclude_c2.txt \
      --sanity-output data/c2_top1000_sanity.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.build_model import build_model
from src.data.transforms import get_val_transforms
from src.utils.ema import ModelEMA


def load_model(cfg, checkpoint_path, device):
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print(f"[INFO] Loaded EMA state_dict (epoch={ckpt.get('epoch')}, best_acc={ckpt.get('best_acc')})")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded model state_dict (no EMA found)")
    else:
        model.load_state_dict(ckpt)
        print("[INFO] Loaded raw state_dict")

    model.to(device).eval()
    return model


def compute_losses(model, loader, device, num_samples):
    losses = np.zeros(num_samples, dtype=np.float32)
    idx_offset = 0

    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=use_amp):
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
            bs = images.size(0)
            losses[idx_offset:idx_offset + bs] = per_sample_loss.cpu().numpy()
            idx_offset += bs

            if (batch_idx + 1) % 500 == 0:
                print(f"  [{idx_offset}/{num_samples}] mean_loss={losses[:idx_offset].mean():.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--top-pct", type=float, default=10.0,
                        help="Percentage of highest-loss samples to exclude")
    parser.add_argument("--per-class-cap", type=float, default=15.0,
                        help="Max pct of samples to remove per class (prevents over-cleaning hard classes)")
    parser.add_argument("--output", default="data/exclude_c2.txt")
    parser.add_argument("--sanity-output", default="data/c2_top1000_sanity.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_model(cfg, args.checkpoint, device)

    val_transform = get_val_transforms(cfg)
    train_dataset = ImageFolder(
        root=os.path.join(args.data_root, "train"),
        transform=val_transform,
    )
    num_samples = len(train_dataset)
    print(f"[INFO] Train set: {num_samples} samples")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("[INFO] Computing per-sample losses...")
    losses = compute_losses(model, loader, device, num_samples)

    # Per-class capped filtering: global top-K% but no class loses more than cap%
    samples = train_dataset.samples
    global_cutoff = np.percentile(losses, 100.0 - args.top_pct)

    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, (path, target) in enumerate(samples):
        class_indices[target].append(i)

    high_loss_indices = []
    class_stats = {}
    for cls, indices in class_indices.items():
        cls_losses = losses[indices]
        above_global = [idx for idx, l in zip(indices, cls_losses) if l > global_cutoff]
        max_remove = int(len(indices) * args.per_class_cap / 100.0)
        if len(above_global) > max_remove:
            above_global_sorted = sorted(above_global, key=lambda i: losses[i], reverse=True)
            above_global = above_global_sorted[:max_remove]
        high_loss_indices.extend(above_global)
        if above_global:
            class_stats[cls] = (len(above_global), len(indices))

    high_loss_indices = np.array(high_loss_indices)

    print(f"\n[STATS] Total: {num_samples}")
    print(f"  Mean loss: {losses.mean():.4f}")
    print(f"  Median loss: {np.median(losses):.4f}")
    print(f"  Global cutoff (top {args.top_pct}%): {global_cutoff:.4f}")
    print(f"  Max loss: {losses.max():.4f}")
    actual_pct = 100.0 * len(high_loss_indices) / num_samples
    print(f"  Excluding: {len(high_loss_indices)} samples ({actual_pct:.2f}%, "
          f"target={args.top_pct}%, per-class cap={args.per_class_cap}%)")
    capped_classes = sum(1 for removed, total in class_stats.values()
                         if removed >= int(total * args.per_class_cap / 100.0))
    print(f"  Classes hitting cap: {capped_classes}/{len(class_stats)}")
    if capped_classes > 0:
        capped_details = [
            (cls, removed, total)
            for cls, (removed, total) in class_stats.items()
            if removed >= int(total * args.per_class_cap / 100.0)
        ]
        capped_details.sort(key=lambda x: x[1], reverse=True)
        print(f"  Capped classes (top 20):")
        for cls, removed, total in capped_details[:20]:
            synset = train_dataset.classes[cls]
            print(f"    class {cls} ({synset}): removed {removed}/{total} ({100.0*removed/total:.1f}%)")

    samples = train_dataset.samples
    excluded_paths = []
    for idx in high_loss_indices:
        path, _ = samples[idx]
        parts = Path(path).parts
        train_idx = next(i for i, p in enumerate(parts) if p == "train")
        rel = "/".join(parts[train_idx:])
        excluded_paths.append(rel)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(excluded_paths)) + "\n", encoding="utf-8")
    print(f"  -> {out_path}")

    # Sanity check: top-1000 highest loss samples (from excluded set)
    sanity_path = Path(args.sanity_output)
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    top1000 = sorted(high_loss_indices, key=lambda i: losses[i], reverse=True)[:1000]
    with open(sanity_path, "w", encoding="utf-8") as f:
        f.write("rank,loss,class_id,filename\n")
        for rank, idx in enumerate(top1000, 1):
            path, target = samples[idx]
            cls_id = Path(path).parent.name
            fname = Path(path).name
            f.write(f"{rank},{losses[idx]:.4f},{cls_id},{fname}\n")
    print(f"  Sanity check (top-1000) -> {sanity_path}")


if __name__ == "__main__":
    main()

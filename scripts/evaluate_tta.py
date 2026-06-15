#!/usr/bin/env python3
"""Validation set evaluation with optional HFlip TTA."""
import sys
import os
import argparse
import yaml
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.data.imagenet_dataset import get_dataloaders
from src.models.build_model import build_model
from src.utils.metrics import accuracy
from src.trainers.base_trainer import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imagenet_resnet50.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tta", type=str, default="none",
                        choices=["none", "hflip"],
                        help="Test-time augmentation: 'hflip' averages softmax(original) + softmax(flipped)")
    parser.add_argument("--weight_source", type=str, default="auto",
                        choices=["auto", "raw", "ema"],
                        help="Force loading raw or ema weights; auto follows best_source or fallback priority")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use autocast fp16 on CUDA (default on)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    _, val_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    # Load checkpoint with explicit weight_source control
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        best_source = ckpt.get("best_source")

        # Explicit override takes priority
        if args.weight_source == "raw" and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            print(f"Loaded raw model weights (forced)")
        elif args.weight_source == "ema" and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"])
            print(f"Loaded EMA weights (forced)")
        # Auto mode: follow best_source if set, else fallback
        elif args.weight_source == "auto":
            if best_source == "raw" and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                print(f"Loaded raw model weights (best_source=raw)")
            elif best_source == "ema" and "ema" in ckpt:
                model.load_state_dict(ckpt["ema"])
                print(f"Loaded EMA weights (best_source=ema)")
            elif "ema" in ckpt:
                model.load_state_dict(ckpt["ema"])
                print(f"Loaded EMA weights (fallback)")
            elif "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                print(f"Loaded raw model weights (fallback)")
            else:
                model.load_state_dict(ckpt)
                print(f"Loaded raw state_dict")
        else:
            raise ValueError(f"Requested {args.weight_source} not found in checkpoint")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded raw state_dict")

    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0

    print(f"Evaluating on {len(val_loader)} batches | TTA: {args.tta} | AMP: {use_amp}")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                if args.tta == "hflip":
                    # Average softmax probabilities of original and horizontally flipped
                    out = model(images)
                    probs = F.softmax(out.float(), dim=-1)
                    out_flipped = model(torch.flip(images, dims=[-1]))
                    probs = probs + F.softmax(out_flipped.float(), dim=-1)
                    # probs now contains sum of two softmax outputs; argmax still correct
                    preds = probs.argmax(dim=-1)
                else:
                    out = model(images)
                    preds = out.argmax(dim=-1)

                # Compute accuracy using the predictions
                correct_top1 = (preds == targets).float().sum()
                # For top-5, use the averaged logits/probs if TTA, else raw output
                if args.tta == "hflip":
                    _, top5_preds = probs.topk(5, dim=-1)
                else:
                    _, top5_preds = out.topk(5, dim=-1)
                correct_top5 = top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).float().sum()

                batch_size = targets.size(0)
                top1_sum += correct_top1.item()
                top5_sum += correct_top5.item()
                total += batch_size

    top1_acc = 100.0 * top1_sum / total
    top5_acc = 100.0 * top5_sum / total

    print(f"\nValidation Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Total samples: {total}")


if __name__ == "__main__":
    main()

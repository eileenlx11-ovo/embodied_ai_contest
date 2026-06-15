"""Per-class top-1 accuracy on ImageNet val.

Extends the single-number eval to 1000 per-class accuracies so we can
quantify exactly how the C1-removed classes (two maillot synsets + ear)
score, and whether removal dragged down neighboring classes.

Usage:
    python scripts/eval_per_class.py --config <cfg> --checkpoint <ckpt> \
        --output reports/perclass_<tag>.csv
"""
import sys
import os
import argparse
import csv
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.data.imagenet_dataset import get_dataloaders
from src.models.build_model import build_model
from src.trainers.base_trainer import resolve_device

# C1-removed synsets (class-level deletion): two maillot dupes + ear (cereal spike)
C1_REMOVED = {"n03710637": "maillot", "n03710721": "maillot/tank_suit", "n13133613": "ear"}


def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
        return "raw_state_dict"
    src = ckpt.get("best_source")
    if src == "raw" and "model" in ckpt:
        model.load_state_dict(ckpt["model"]); return "model(best_source=raw)"
    if src == "ema" and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"]); return "EMA(best_source=ema)"
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"]); return "EMA"
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"]); return "model"
    model.load_state_dict(ckpt); return "raw_state_dict"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--worst", type=int, default=20, help="how many worst classes to print")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(args.device)

    _, val_loader = get_dataloaders(cfg)
    # ImageFolder is wrapped; reach the underlying dataset for class names
    base = val_loader.dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    classes = base.classes  # synset ids, sorted; target index aligns to this
    num_classes = len(classes)

    model = build_model(cfg).to(device)
    tag = load_ckpt(model, args.checkpoint, device)
    print(f"Loaded {tag} from {args.checkpoint}")
    model.eval()

    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1).cpu()
            for t, p in zip(targets, preds):
                total[t] += 1
                if t == p:
                    correct[t] += 1

    acc = (correct.float() / total.clamp(min=1).float() * 100)
    overall = correct.sum().item() / total.sum().item() * 100
    print(f"\nOverall top-1: {overall:.2f}%  ({correct.sum().item()}/{total.sum().item()})")

    print("\n=== C1-removed classes ===")
    for i, syn in enumerate(classes):
        if syn in C1_REMOVED:
            print(f"  {syn} ({C1_REMOVED[syn]}): {acc[i]:.1f}%  ({correct[i]}/{total[i]})")

    order = acc.argsort()
    print(f"\n=== {args.worst} worst classes ===")
    for i in order[:args.worst].tolist():
        flag = "  <-- C1-removed" if classes[i] in C1_REMOVED else ""
        print(f"  {classes[i]}: {acc[i]:.1f}%  ({correct[i]}/{total[i]}){flag}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_index", "synset", "correct", "total", "top1_acc"])
            for i in range(num_classes):
                w.writerow([i, classes[i], int(correct[i]), int(total[i]), f"{acc[i]:.4f}"])
        print(f"\nPer-class CSV saved to {args.output}")


if __name__ == "__main__":
    main()

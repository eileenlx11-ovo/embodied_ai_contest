import sys
import os
import argparse
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.data.imagenet_dataset import get_dataloaders
from src.models.build_model import build_model
from src.utils.metrics import accuracy
from src.trainers.base_trainer import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="auto=自动检测, cuda=强制GPU, cpu=强制CPU")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = resolve_device(args.device)

    _, val_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")

    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            top1_sum += acc1 * batch_size
            top5_sum += acc5 * batch_size
            total += batch_size

    print(f"Validation Results:")
    print(f"  Top-1 Accuracy: {top1_sum / total:.2f}%")
    print(f"  Top-5 Accuracy: {top5_sum / total:.2f}%")
    print(f"  Total samples: {total}")


if __name__ == "__main__":
    main()

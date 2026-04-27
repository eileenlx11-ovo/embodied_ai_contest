import sys
import os
import argparse
import yaml
import csv
import torch
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.build_model import build_model
from src.data.transforms import get_val_transforms
from src.trainers.base_trainer import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="submit/result.csv")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="auto=自动检测, cuda=强制GPU, cpu=强制CPU")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    transform = get_val_transforms(cfg)

    # CIFAR-10 demo mode: predict on val set
    if cfg["data"]["dataset"] == "cifar10":
        from torchvision.datasets import CIFAR10
        from torch.utils.data import DataLoader
        dataset = CIFAR10(root=cfg["data"]["root"], train=False, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            idx = 0
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    for p in preds:
                        writer.writerow([f"image_{idx:05d}.jpg", f"{p.item():04d}"])
                        idx += 1
        print(f"Predictions saved to {args.output} ({idx} samples)")
        return

    # ImageNet mode: predict on test directory
    test_dir = args.test_dir or os.path.join(cfg["data"]["root"], "test")
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder as IF
    import glob
    from PIL import Image

    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.JPEG")) +
                         glob.glob(os.path.join(test_dir, "*.jpg")) +
                         glob.glob(os.path.join(test_dir, "*.png")))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        with torch.no_grad():
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device)
                output = model(img_t)
                _, pred = output.max(1)
                filename = os.path.basename(path)
                writer.writerow([filename, f"{pred.item():04d}"])

    print(f"Predictions saved to {args.output} ({len(image_paths)} samples)")


if __name__ == "__main__":
    main()

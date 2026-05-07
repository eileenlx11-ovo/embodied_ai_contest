import sys
import os
import argparse
import yaml
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.build_model import build_model
from src.data.transforms import get_val_transforms
from src.trainers.base_trainer import resolve_device


class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.paths = sorted(
            glob.glob(os.path.join(image_dir, "*.JPEG"))
            + glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.paths[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="submit/result.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    model = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")
    model.eval()

    transform = get_val_transforms(cfg)
    num_workers = cfg["data"].get("num_workers", 4)

    if cfg["data"]["dataset"] == "cifar10":
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=cfg["data"]["root"], train=False, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        rows = []
        with torch.no_grad():
            idx = 0
            for images, _ in loader:
                images = images.to(device)
                _, preds = model(images).max(1)
                for p in preds:
                    rows.append([f"image_{idx:05d}.jpg", f"{p.item():04d}"])
                    idx += 1

        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"Predictions saved to {args.output} ({len(rows)} samples)")
        return

    test_dir = args.test_dir or os.path.join(cfg["data"]["root"], "test")
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)

    dataset = TestImageDataset(test_dir, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    print(f"Predicting {len(dataset)} images from {test_dir}")
    rows = []
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device, non_blocking=True)
            _, preds = model(images).max(1)
            for fname, p in zip(filenames, preds):
                rows.append([fname, f"{p.item():04d}"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Predictions saved to {args.output} ({len(rows)} samples)")


if __name__ == "__main__":
    main()

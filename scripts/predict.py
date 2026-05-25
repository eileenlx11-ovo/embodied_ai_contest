import sys
import os
import argparse
import yaml
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.build_model import build_model
from src.data.transforms import get_val_transforms
from src.trainers.base_trainer import resolve_device


def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        tag = "EMA"
    else:
        model.load_state_dict(ckpt["model"])
        tag = "model"
    return tag


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
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="One or more checkpoint paths; multiple = ensemble (equal weight, softmax-averaged).")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="submit/result.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tta", type=str, default="hflip",
                        choices=["none", "hflip"],
                        help="Test-time augmentation: 'hflip' doubles inference time for ~+0.2-0.5%% top-1.")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use autocast fp16 on CUDA (default on).")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    model = build_model(cfg).to(device)
    model.eval()

    transform = get_val_transforms(cfg)
    num_workers = cfg["data"].get("num_workers", 4)
    num_classes = cfg["model"]["num_classes"]

    # CIFAR10 branch kept simple — no TTA / ensemble (use first ckpt only).
    if cfg["data"]["dataset"] == "cifar10":
        from torchvision.datasets import CIFAR10
        tag = load_weights(model, args.checkpoint[0], device)
        print(f"Loaded {tag} weights from {args.checkpoint[0]}")
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

    n = len(dataset)
    filenames_in_order = [os.path.basename(p) for p in dataset.paths]
    # Accumulator on CPU to avoid GPU OOM when N is large (100k × 1000 × 4B ≈ 400MB).
    logits_sum = torch.zeros(n, num_classes, dtype=torch.float32)

    print(f"Predicting {n} images from {test_dir}")
    print(f"  Ensemble: {len(args.checkpoint)} checkpoint(s) | TTA: {args.tta} | AMP: {use_amp}")

    for ckpt_idx, ckpt_path in enumerate(args.checkpoint):
        tag = load_weights(model, ckpt_path, device)
        print(f"[{ckpt_idx+1}/{len(args.checkpoint)}] Loaded {tag} from {ckpt_path}")
        offset = 0
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"ckpt{ckpt_idx+1}"):
                images = images.to(device, non_blocking=True)
                bsz = images.size(0)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(images)
                    probs = F.softmax(out.float(), dim=-1)
                    if args.tta == "hflip":
                        out_f = model(torch.flip(images, dims=[-1]))
                        probs = probs + F.softmax(out_f.float(), dim=-1)
                logits_sum[offset:offset+bsz] += probs.detach().cpu()
                offset += bsz

    preds = logits_sum.argmax(dim=1).tolist()
    rows = [[fname, f"{p:04d}"] for fname, p in zip(filenames_in_order, preds)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Predictions saved to {args.output} ({len(rows)} samples)")


if __name__ == "__main__":
    main()

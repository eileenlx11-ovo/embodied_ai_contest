"""
C1: Remove training samples from classes with known dataset-level bugs.

Based on confusion_top10.md analysis (baseline 77.53% ckpt):
- n03710637 (maillot) + n03710721 (maillot): duplicate class names, 45 mutual confusions
- n13133613 (ear): homonym — mixes "ear of corn" with "human ear", 20 misclassifications

Outputs: data/exclude_c1.txt (one relative path per line, e.g. "train/n03710637/xxx.JPEG")
"""

import argparse
import os
from pathlib import Path


BUGGY_CLASSES = ["n03710637", "n03710721", "n13133613"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="ImageNet root (contains train/)")
    parser.add_argument("--output", default="data/exclude_c1.txt")
    args = parser.parse_args()

    train_root = Path(args.data_root) / "train"
    excluded = []

    for cls_id in BUGGY_CLASSES:
        cls_dir = train_root / cls_id
        if not cls_dir.exists():
            print(f"[WARN] class dir not found: {cls_dir}")
            continue
        files = sorted(s for s in cls_dir.iterdir() if s.is_file())
        for s in files:
            excluded.append(f"train/{cls_id}/{s.name}")
        print(f"  {cls_id}: {len(files)} samples excluded")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(excluded) + "\n", encoding="utf-8")
    print(f"\nC1 total: {len(excluded)} samples -> {out_path}")


if __name__ == "__main__":
    main()

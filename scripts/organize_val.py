"""
将 flat val 目录按类别组织为 class 子文件夹，同时生成 class_index.json

用法:
  C:/Python312/python.exe scripts/organize_val.py \
      --val-dir ./data/imagenet/val_flat \
      --output-dir ./data/imagenet/val \
      --meta-mat ./data/imagenet/ImageNet-1K/ILSVRC2012_devkit_t12/data/meta.mat \
      --ground-truth ./data/imagenet/ImageNet-1K/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt \
      --class-index-output ./data/imagenet_class_index.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_meta_mapping(meta_mat_path: str):
    """从 meta.mat 提取 {ILSVRC2012_ID: WNID} 和 {WNID: class_name}"""
    mat = loadmat(meta_mat_path)
    synsets = mat["synsets"]

    id_to_wnid = {}
    wnid_to_name = {}

    for i in range(synsets.shape[0]):
        entry = synsets[i, 0]
        ils_id = int(entry["ILSVRC2012_ID"][0, 0])
        wnid = str(entry["WNID"][0])
        words = str(entry["words"][0])

        id_to_wnid[ils_id] = wnid
        if ils_id <= 1000:  # only low-level synsets
            wnid_to_name[wnid] = words.split(",")[0].strip()

    return id_to_wnid, wnid_to_name


def organize_val(val_dir: str, output_dir: str, id_to_wnid: dict, ground_truth_path: str):
    """将 flat val 图片按类别移动/复制到 class 子文件夹"""
    val_dir = Path(val_dir)
    output_dir = Path(output_dir)

    # 读取 ground truth (每行一个 ILSVRC2012_ID)
    with open(ground_truth_path, "r") as f:
        labels = [int(line.strip()) for line in f]

    # 获取排序后的 val 图片列表
    val_images = sorted(
        [f for f in val_dir.iterdir() if f.suffix.lower() in (".jpeg", ".jpg", ".png")]
    )

    if len(val_images) != len(labels):
        print(f"[WARN] 图片数量 ({len(val_images)}) != 标签数量 ({len(labels)})")

    # 创建 class 子文件夹并复制图片
    for img_path, label_id in zip(val_images, labels):
        wnid = id_to_wnid.get(label_id, f"unknown_{label_id}")
        class_dir = output_dir / wnid
        class_dir.mkdir(parents=True, exist_ok=True)

        dst = class_dir / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)

    # 统计
    class_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    total = sum(sum(1 for _ in d.iterdir()) for d in class_dirs)
    print(f"[OK] val 已组织: {len(class_dirs)} 个类别, {total} 张图片 -> {output_dir}")


def generate_class_index(wnid_to_name: dict, output_path: str):
    """生成 imagenet_class_index.json 格式: {wnid: [wnid, class_name]}"""
    class_index = {}
    for wnid, name in sorted(wnid_to_name.items()):
        class_index[wnid] = [wnid, name]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_index, f, ensure_ascii=False, indent=2)

    print(f"[OK] class_index.json 已保存: {output_path} ({len(class_index)} 个类别)")


def main():
    parser = argparse.ArgumentParser(description="组织 ImageNet val 目录并生成 class_index.json")
    parser.add_argument("--val-dir", default="./data/imagenet/val_flat")
    parser.add_argument("--output-dir", default="./data/imagenet/val")
    parser.add_argument("--meta-mat", default="./data/imagenet/ImageNet-1K/ILSVRC2012_devkit_t12/data/meta.mat")
    parser.add_argument("--ground-truth", default="./data/imagenet/ImageNet-1K/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
    parser.add_argument("--class-index-output", default="./data/imagenet_class_index.json")
    args = parser.parse_args()

    # 1. 加载映射
    print("加载 meta.mat 中的类别映射...")
    id_to_wnid, wnid_to_name = load_meta_mapping(args.meta_mat)
    print(f"  低层 synset (1-1000): {sum(1 for k in id_to_wnid if k <= 1000)}")
    print(f"  高层 synset (>1000): {sum(1 for k in id_to_wnid if k > 1000)}")

    # 2. 生成 class_index.json
    print("\n生成 class_index.json...")
    generate_class_index(wnid_to_name, args.class_index_output)

    # 3. 组织 val 目录
    print("\n组织 val 目录...")
    organize_val(args.val_dir, args.output_dir, id_to_wnid, args.ground_truth)

    return 0


if __name__ == "__main__":
    sys.exit(main())

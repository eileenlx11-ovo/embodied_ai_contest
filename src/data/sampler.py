"""
ClassBalancedSampler — 长尾分布采样策略.

策略:
  1. ClassBalancedSampler: 按 1/count^beta 加权采样，尾部类别更频繁出现
  2. compute_loss_weights: 为 CE / SCE loss 提供 per-class 权重（用于 re-weighting）

用法:
  from src.data.sampler import ClassBalancedSampler, build_class_weights

  weights, stats = build_class_weights("docs/week1/class_counts.csv", beta=0.5)
  sampler = ClassBalancedSampler(dataset, weights)
  loader = DataLoader(dataset, batch_size=128, sampler=sampler)
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler


def build_class_weights(
    class_counts_csv: str,
    beta: float = 0.5,
) -> Tuple[Dict[str, float], Dict]:
    """
    从 class_counts.csv 构建 per-class 权重。

    weight_c = (1 / count_c) ^ beta

    beta = 0   → uniform (no balancing)
    beta = 0.5 → sqrt-inverse (推荐, 温和平衡)
    beta = 1.0 → full inverse (尾部类被大幅过采样)

    Returns:
        weights:  {class_id: weight}  归一化后 sum = num_classes
        stats:   {imbalance_ratio, max/min count, weight_range, ...}
    """
    classes = []
    with open(class_counts_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes.append((row["class_id"], row["class_name"], int(row["num_images"])))

    counts = np.array([c[2] for c in classes], dtype=np.float64)
    raw_weights = (1.0 / counts) ** beta
    normalized = raw_weights / raw_weights.sum() * len(classes)

    weights = {}
    for i, (class_id, class_name, count) in enumerate(classes):
        weights[class_id] = {
            "class_name": class_name,
            "num_images": count,
            "weight": round(float(normalized[i]), 6),
            "raw_weight": round(float(raw_weights[i]), 8),
        }

    stats = {
        "num_classes": len(classes),
        "total_images": int(counts.sum()),
        "beta": beta,
        "max_count": int(counts.max()),
        "min_count": int(counts.min()),
        "mean_count": round(float(counts.mean()), 1),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
        "weight_max": round(float(normalized.max()), 6),
        "weight_min": round(float(normalized.min()), 6),
        "weight_ratio": round(float(normalized.max() / normalized.min()), 2),
    }

    return weights, stats


def export_sampling_weights(
    class_counts_csv: str,
    output_json: str,
    beta_values: Optional[List[float]] = None,
):
    """
    生成 sampling_weights.json，交付给 A 和 C。

    包含多个 beta 值的权重（供对比实验）:
      - beta=0.0: uniform baseline
      - beta=0.5: 温和平衡（推荐，适合 ImageNet-1K 的 1.78x 不平衡）
      - beta=1.0: 全逆频（激进，尾部类被大幅上采样）
    """
    if beta_values is None:
        beta_values = [0.0, 0.5, 1.0]

    output = {"_comment": "ClassBalancedSampler weights for ImageNet-1K"}

    for beta in beta_values:
        weights, stats = build_class_weights(class_counts_csv, beta=beta)
        key = f"beta_{beta}"
        output[key] = {
            "stats": stats,
            "weights": weights,
        }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


class ClassBalancedSampler(Sampler):
    """
    基于类别权重的采样器，尾部类别以更高概率被抽取。

    与 WeightedRandomSampler 的区别：
      - 直接接受 class_id → weight 映射，自动处理 ImageFolder 的 target 索引
      - 支持与 torchvision ImageFolder 无缝集成
    """

    def __init__(
        self,
        dataset,
        class_weights: Dict[str, float],
        class_to_idx: Optional[Dict[str, int]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Args:
            dataset: ImageFolder 实例（需要 dataset.targets 和 dataset.class_to_idx）
            class_weights: {class_id: normalized_weight} 来自 build_class_weights
            class_to_idx: 可选，覆盖 dataset 自带的 class_to_idx
            num_samples: 每 epoch 采样数，默认 = len(dataset)
            replacement: 是否允许重复采样（长尾场景推荐 True）
        """
        if class_to_idx is None:
            class_to_idx = getattr(dataset, "class_to_idx", None)
        if class_to_idx is None:
            raise ValueError("需要 class_to_idx 映射或 dataset.class_to_idx")

        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise ValueError("dataset 需要有 targets 属性（ImageFolder 自带）")

        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement

        # idx → class_id 反向映射
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        sample_weights = []
        for t in targets:
            class_id = idx_to_class.get(t)
            w = class_weights.get(class_id, {}).get("weight", 1.0) if isinstance(
                class_weights.get(class_id), dict
            ) else class_weights.get(class_id, 1.0)
            sample_weights.append(float(w))

        self.weights = torch.tensor(sample_weights, dtype=torch.float64)

    def __iter__(self):
        return iter(
            torch.multinomial(
                self.weights, self.num_samples, self.replacement
            ).tolist()
        )

    def __len__(self):
        return self.num_samples


def compute_loss_weights(
    class_weights: Dict,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    将 per-class 采样权重转换为 CE loss 的 class_weight tensor。

    Loss weight = sampling_weight * scale
    形状: (num_classes,) 可直接传入 nn.CrossEntropyLoss(weight=...)

    Args:
        class_weights: build_class_weights() 返回的 weights dict
        scale: 缩放因子，默认 1.0

    Returns:
        Tensor of shape (1000,)  dtype=float32
    """
    # 按 class_id 字母序排列（与 ImageFolder class_to_idx 一致）
    sorted_ids = sorted(class_weights.keys())
    w = []
    for cid in sorted_ids:
        entry = class_weights[cid]
        if isinstance(entry, dict):
            w.append(entry["weight"])
        else:
            w.append(float(entry))
    t = torch.tensor(w, dtype=torch.float32)
    return t * scale / t.mean()


# ---------------------------------------------------------------------------
# CLI — 可直接运行生成权重文件
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="生成 ClassBalancedSampler 权重")
    parser.add_argument("--class-counts", default="docs/week1/class_counts.csv")
    parser.add_argument("--output", default="docs/week3/sampling_weights.json")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    args = parser.parse_args()

    if not os.path.exists(args.class_counts):
        base = Path(__file__).resolve().parent.parent.parent
        args.class_counts = str(base / args.class_counts)
        args.output = str(base / args.output)

    result = export_sampling_weights(args.class_counts, args.output, args.beta)

    # 打印摘要
    for beta_key in sorted(result.keys()):
        if beta_key.startswith("_"):
            continue
        s = result[beta_key]["stats"]
        print(
            f"beta={s['beta']:.1f}  |  "
            f"weight [{s['weight_min']:.6f}, {s['weight_max']:.6f}]  "
            f"ratio={s['weight_ratio']:.2f}x  |  "
            f"data imbalance={s['imbalance_ratio']:.2f}x"
        )

    print(f"\nWeights saved to {args.output}")

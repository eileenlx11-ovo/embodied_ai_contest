"""
类别统计分析 — 可复用模块

功能：
  count_images_per_class()    — 遍历目录统计每类图片数
  compute_imbalance_metrics() — 计算不平衡比、均值、中位数等
  get_top_bottom_classes()    — 返回 Top-N 最多/最少类别
  plot_class_distribution()   — 生成 log-scale 直方图
  export_class_counts_csv()   — 导出 class_counts.csv

用法：
  from src.analysis.class_stats import (
      count_images_per_class,
      compute_imbalance_metrics,
      get_top_bottom_classes,
      plot_class_distribution,
      export_class_counts_csv,
  )
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np


# ---------- 类别名映射 ----------

def load_class_index(class_index_path: str) -> Dict[str, str]:
    """
    加载 imagenet_class_index.json → {class_id: class_name}
    文件格式: {"n01440764": ["n01440764", "tench"], ...}
    """
    with open(class_index_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        if isinstance(first_val, list):
            return {k: v[1] for k, v in raw.items()}
        else:
            return raw
    return {}


def get_default_class_index_url() -> str:
    """ImageNet 官方 class_index.json 的常用镜像地址"""
    return "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"


# ---------- 核心统计 ----------

def count_images_per_class(
    data_root: str,
    split: str = "train",
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"),
) -> Dict[str, int]:
    """
    遍历数据集目录，统计每个类别的图片数量

    Args:
        data_root: ImageNet 根目录（如 ./data/imagenet）
        split: "train" or "val"

    Returns:
        {class_id: count} 字典，按 count 降序排列
    """
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"目录不存在: {split_dir}")

    class_counts = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in extensions)
        class_counts[class_dir.name] = count

    # 降序排列
    return dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))


def compute_imbalance_metrics(class_counts: Dict[str, int]) -> Dict:
    """
    计算类别不平衡相关指标

    Returns:
        {
            "num_classes": int,
            "total_images": int,
            "max_count": int,
            "min_count": int,
            "mean": float,
            "median": float,
            "std": float,
            "imbalance_ratio": float,      # max / min
            "q1": float,                    # 25% 分位数
            "q3": float,                    # 75% 分位数
        }
    """
    counts = np.array(list(class_counts.values()))

    return {
        "num_classes": len(counts),
        "total_images": int(counts.sum()),
        "max_count": int(counts.max()),
        "min_count": int(counts.min()),
        "mean": round(float(counts.mean()), 1),
        "median": round(float(np.median(counts)), 1),
        "std": round(float(counts.std()), 1),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
        "q1": round(float(np.percentile(counts, 25)), 1),
        "q3": round(float(np.percentile(counts, 75)), 1),
    }


def get_top_bottom_classes(
    class_counts: Dict[str, int],
    n: int = 20,
    class_names: Optional[Dict[str, str]] = None,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    获取 Top-N 最多和 Top-N 最少类别

    Returns:
        top: [(class_id, class_name, count), ...] 降序
        bottom: [(class_id, class_name, count), ...] 升序
    """
    items = list(class_counts.items())  # 已按 count 降序
    names = class_names or {}

    top = []
    for class_id, count in items[:n]:
        name = names.get(class_id, "")
        top.append((class_id, name, count))

    bottom = []
    for class_id, count in items[-n:][::-1]:  # 反转使升序
        name = names.get(class_id, "")
        bottom.append((class_id, name, count))

    return top, bottom


# ---------- 可视化 ----------

def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: str,
    metrics: Optional[Dict] = None,
    title: str = "ImageNet-1K Class Distribution",
):
    """
    生成类别分布直方图（log-scale），保存为 PNG

    Args:
        class_counts: {class_id: count}
        output_path: PNG 输出路径（如 ./figures/class_distribution.png）
        metrics: 可选，compute_imbalance_metrics() 的返回值，用于标注
        title: 图表标题
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    counts = np.array(list(class_counts.values()))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ---- 子图1: 直方图（log-scale x轴）----
    ax1 = axes[0]
    ax1.hist(counts, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Images per Class (log scale)", fontsize=12)
    ax1.set_ylabel("Number of Classes", fontsize=12)
    ax1.set_title(f"{title}\nHistogram (log-scale bins)", fontsize=13)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.axvline(np.median(counts), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(counts):.0f}")
    ax1.axvline(np.mean(counts), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(counts):.0f}")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # ---- 子图2: 排序图（降序散点 or bar）----
    ax2 = axes[1]
    sorted_counts = np.sort(counts)[::-1]
    x_range = np.arange(1, len(sorted_counts) + 1)
    ax2.fill_between(x_range, sorted_counts, alpha=0.6, color="steelblue")
    ax2.plot(x_range, sorted_counts, linewidth=1.0, color="darkblue")
    ax2.set_xlabel("Class Rank (by image count)", fontsize=12)
    ax2.set_ylabel("Images per Class", fontsize=12)
    ax2.set_title(f"{title}\nSorted Class Distribution", fontsize=13)
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    # 标注统计指标
    if metrics:
        textstr = (
            f"Classes: {metrics['num_classes']}\n"
            f"Total images: {metrics['total_images']:,}\n"
            f"Max/Min ratio: {metrics['imbalance_ratio']}x\n"
            f"Mean: {metrics['mean']:.0f}, Median: {metrics['median']:.0f}\n"
            f"Max: {metrics['max_count']}, Min: {metrics['min_count']}"
        )
        ax2.text(
            0.98, 0.95, textstr,
            transform=ax2.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 直方图已保存: {output_path}")


# ---------- 导出 ----------

def export_class_counts_csv(
    class_counts: Dict[str, int],
    output_path: str,
    class_names: Optional[Dict[str, str]] = None,
):
    """
    导出 class_counts.csv

    格式:
        class_id,class_name,num_images
        n01440764,tench,1300
        ...
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = class_names or {}
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "num_images"])
        for class_id, count in class_counts.items():
            name = names.get(class_id, "")
            writer.writerow([class_id, name, count])

    print(f"[OK] CSV 已保存: {output_path}")


def export_top_bottom_report(
    top: List[Tuple[str, str, int]],
    bottom: List[Tuple[str, str, int]],
    output_path: str,
    metrics: Optional[Dict] = None,
):
    """导出 Top/Bottom 类别报告（文本 + JSON）"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("ImageNet-1K Long-Tail Analysis Report")
    lines.append("=" * 70)

    if metrics:
        lines.append(f"\nImbalance Metrics:")
        lines.append(f"  Classes:       {metrics['num_classes']}")
        lines.append(f"  Total images:  {metrics['total_images']:,}")
        lines.append(f"  Max count:     {metrics['max_count']}")
        lines.append(f"  Min count:     {metrics['min_count']}")
        lines.append(f"  Mean:          {metrics['mean']}")
        lines.append(f"  Median:        {metrics['median']}")
        lines.append(f"  Std:           {metrics['std']}")
        lines.append(f"  Q1 / Q3:       {metrics['q1']} / {metrics['q3']}")
        lines.append(f"  Imbalance ratio (max/min): {metrics['imbalance_ratio']}x")
        lines.append(f"  {'⚠ Highly imbalanced!' if metrics['imbalance_ratio'] > 2 else 'Relatively balanced.'}")

    lines.append(f"\nTop-{len(top)} classes (most images):")
    lines.append("-" * 50)
    for rank, (cls_id, cls_name, cnt) in enumerate(top, 1):
        lines.append(f"  {rank:3d}. {cls_id}  {cls_name:<25s}  {cnt:6d}")

    lines.append(f"\nBottom-{len(bottom)} classes (least images):")
    lines.append("-" * 50)
    for rank, (cls_id, cls_name, cnt) in enumerate(bottom, 1):
        lines.append(f"  {rank:3d}. {cls_id}  {cls_name:<25s}  {cnt:6d}")

    report_text = "\n".join(lines)

    # 保存文本报告
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] 文本报告已保存: {txt_path}")

    # 也打印到控制台
    print(report_text)

    # 保存 JSON
    json_path = output_path.with_suffix(".json")
    json_data = {
        "metrics": metrics,
        "top_classes": [(cls_id, cls_name, cnt) for cls_id, cls_name, cnt in top],
        "bottom_classes": [(cls_id, cls_name, cnt) for cls_id, cls_name, cnt in bottom],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON 报告已保存: {json_path}")

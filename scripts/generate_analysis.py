"""
一键生成方向一统计分析三件套：

  data/metadata/class_counts.csv       — 每类图片数
  figures/class_distribution.png       — log-scale 直方图
  figures/longtail_report.txt/json     — Top-20 最多/最少类别 + 不平衡指标

用法:
  C:/Python312/python.exe scripts/generate_analysis.py \
      --data-root ./data/imagenet \
      --class-index ./data/imagenet_class_index.json \
      --output-dir ./figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from analysis.class_stats import (
    count_images_per_class,
    compute_imbalance_metrics,
    get_top_bottom_classes,
    plot_class_distribution,
    export_class_counts_csv,
    export_top_bottom_report,
    load_class_index,
    get_default_class_index_url,
)


def main():
    parser = argparse.ArgumentParser(description="ImageNet-1K 类别统计分析")
    parser.add_argument("--data-root", default="./data/imagenet", help="ImageNet 根目录")
    parser.add_argument("--class-index", default=None, help="imagenet_class_index.json 路径（可选）")
    parser.add_argument("--output-dir", default="./figures", help="图表输出目录")
    parser.add_argument("--metadata-dir", default="./data/metadata", help="CSV/JSON 输出目录")
    parser.add_argument("--top-n", type=int, default=20, help="Top/Bottom N 类别数")
    parser.add_argument("--split", default="train", help="统计哪个 split（train/val）")
    args = parser.parse_args()

    # ---- 1. 加载类别名映射 ----
    class_names = {}
    if args.class_index and os.path.exists(args.class_index):
        class_names = load_class_index(args.class_index)
        print(f"[INFO] 已加载 {len(class_names)} 个类别名")
    else:
        print(f"[WARN] 未提供 class_index.json，CSV 中 class_name 列为空")
        print(f"       可从 {get_default_class_index_url()} 下载")

    # ---- 2. 统计每类图片数 ----
    print(f"\n统计类别分布: {args.data_root}/{args.split}/ ...")
    class_counts = count_images_per_class(args.data_root, args.split)

    if not class_counts:
        print(f"[ERROR] 未找到图片，请确认数据集路径正确: {args.data_root}/{args.split}/")
        sys.exit(1)

    # ---- 3. 计算不平衡指标 ----
    metrics = compute_imbalance_metrics(class_counts)

    # ---- 4. Top/Bottom N 类别 ----
    top, bottom = get_top_bottom_classes(class_counts, n=args.top_n, class_names=class_names)

    # ---- 5. 导出 CSV ----
    csv_path = os.path.join(args.metadata_dir, "class_counts.csv")
    export_class_counts_csv(class_counts, csv_path, class_names)

    # ---- 6. 绘制直方图 ----
    png_path = os.path.join(args.output_dir, "class_distribution.png")
    plot_class_distribution(class_counts, png_path, metrics=metrics)

    # ---- 7. 导出长尾报告 ----
    report_path = os.path.join(args.output_dir, "longtail_report")
    export_top_bottom_report(top, bottom, report_path, metrics=metrics)

    # ---- 8. 摘要 ----
    print(f"\n{'='*50}")
    print(f"不平衡比 (max/min): {metrics['imbalance_ratio']}x")
    if metrics['imbalance_ratio'] > 2:
        print(f">>> 建议启用长尾处理策略（ClassBalancedSampler / Re-weighting）")
    else:
        print(f">>> 分布较均匀，可暂不启用长尾处理")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

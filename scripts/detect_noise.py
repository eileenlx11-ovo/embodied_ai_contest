"""
Week 2 — 一键噪声检测 + 生成 Top-20 噪声报告

用法:
  C:/Python312/python.exe scripts/detect_noise.py \
      --data-root ./data/imagenet \
      --class-index ./data/imagenet_class_index.json \
      --output-dir ./docs
"""

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from analysis.noise_analysis import (
    scan_all_classes,
    rank_noisy_classes,
    get_loss_tuning_hints,
)
from analysis.class_stats import load_class_index


def main():
    parser = argparse.ArgumentParser(description="ImageNet-1K 噪声检测与 Top-20 报告")
    parser.add_argument("--data-root", default="./data/imagenet", help="ImageNet 根目录")
    parser.add_argument("--class-index", default=None, help="imagenet_class_index.json 路径")
    parser.add_argument("--output-dir", default="./docs", help="报告输出目录")
    parser.add_argument("--split", default="train", help="扫描哪个 split")
    parser.add_argument("--sample-size", type=int, default=50, help="每类抽样数")
    parser.add_argument("--max-classes", type=int, default=None, help="限制扫描类别数（调试用）")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N 噪声类别")
    args = parser.parse_args()

    # 加载类别名
    class_names = {}
    if args.class_index and os.path.exists(args.class_index):
        class_names = load_class_index(args.class_index)
        print(f"[INFO] 已加载 {len(class_names)} 个类别名")
    else:
        print("[WARN] 未提供 class_index.json，报告中将无类别名")

    # 全扫描
    print(f"\n开始扫描: {args.data_root}/{args.split}/")
    print(f"每类抽样: {args.sample_size} 张")
    if args.max_classes:
        print(f"限制: 前 {args.max_classes} 个类别")
    print()

    results = scan_all_classes(
        args.data_root,
        args.split,
        class_names=class_names,
        sample_size=args.sample_size,
        max_classes=args.max_classes,
    )

    if not results:
        print("[ERROR] 未找到任何类别")
        sys.exit(1)

    # Top/Bottom
    top, bottom = rank_noisy_classes(results, top_n=args.top_n)

    # 确保输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 保存 JSON
    json_path = os.path.join(args.output_dir, "noise_scan.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "top_noisy": top,
            "bottom_noisy": bottom,
            "full_results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] JSON 已保存: {json_path}")

    # 保存 Markdown 报告
    md_path = os.path.join(args.output_dir, "noise_top20.md")
    _write_markdown_report(md_path, top, bottom, results, class_names, sample=args.sample_size)

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, "noise_all_classes.csv")
    _write_csv(csv_path, results)


def _write_markdown_report(md_path, top, bottom, all_results, class_names, sample=200):
    lines = []
    lines.append("# ImageNet-1K 噪声分析报告 (Week 2)")
    lines.append("")
    lines.append(f"> 扫描范围: {len(all_results)} 类别, 每类抽样 {sample} 张")
    lines.append(f"> 检测信号: 文件损坏 / 灰度图 / 小尺寸 / 低对比度 / 近空白 / 颜色离群")
    lines.append("")

    # 总体统计
    scores = [r["noise_score"] for r in all_results]
    lines.append("## 总体统计")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 平均噪声评分 | {sum(scores)/len(scores):.2f} |")
    lines.append(f"| 最高噪声评分 | {max(scores):.2f} |")
    lines.append(f"| 最低噪声评分 | {min(scores):.2f} |")
    lines.append(f"| 中位数噪声评分 | {sorted(scores)[len(scores)//2]:.2f} |")
    lines.append("")

    # Top-20 噪声类别
    lines.append(f"## Top-{len(top)} 噪声类别（最需关注）")
    lines.append("")
    lines.append("| 排名 | class_id | 类别名 | 噪声评分 | 总图数 | 损坏 | 灰度 | 小图 | 低对比度 | 近空白 | 颜色离群 | Loss建议 |")
    lines.append("|------|----------|--------|----------|--------|------|------|------|----------|--------|----------|----------|")
    for rank, r in enumerate(top, 1):
        name = r.get("class_name", "") or r["class_id"]
        lines.append(
            f"| {rank} | {r['class_id']} | {name} | {r['noise_score']:.1f} | "
            f"{r['total']} | {r['corrupted']} | {r['grayscale']} | {r['small']} | "
            f"{r['low_contrast']} | {r['near_blank']} | {r['color_outlier']} | "
            f"{get_loss_tuning_hints(r['noise_score'])} |"
        )
    lines.append("")

    # Bottom 最干净类别
    lines.append(f"## Bottom-{len(bottom)} 最干净类别（参考对照）")
    lines.append("")
    lines.append("| 排名 | class_id | 类别名 | 噪声评分 | 总图数 |")
    lines.append("|------|----------|--------|----------|--------|")
    for rank, r in enumerate(bottom, 1):
        name = r.get("class_name", "") or r["class_id"]
        lines.append(f"| {rank} | {r['class_id']} | {name} | {r['noise_score']:.1f} | {r['total']} |")
    lines.append("")

    # Loss 调参建议汇总
    lines.append("## Loss 调参建议（交付 C）")
    lines.append("")
    high_noise = [r for r in top if r["noise_score"] > 10]
    mid_noise = [r for r in top if 5 <= r["noise_score"] <= 10]
    low_noise = [r for r in top if r["noise_score"] < 5]

    lines.append(f"- **高噪声类别 ({len(high_noise)} 个, 评分 >10)**: 这些类别建议启用强噪声鲁棒 Loss")
    lines.append(f"  - 推荐: ELR + SCE 混合, alpha=1.0, beta=0.1, warmup=10 epochs")
    lines.append(f"  - 涉及类别: {', '.join(r['class_id'] for r in high_noise[:5])}{'...' if len(high_noise) > 5 else ''}")
    lines.append("")
    lines.append(f"- **中等噪声类别 ({len(mid_noise)} 个, 评分 5-10)**: 适当提高 SCE 权重")
    lines.append(f"  - 推荐: SCE alpha=0.5-0.7, beta=0.3-0.5 或 GCE q=0.5-0.7")
    lines.append(f"  - 涉及类别: {', '.join(r['class_id'] for r in mid_noise[:5])}{'...' if len(mid_noise) > 5 else ''}")
    lines.append("")
    lines.append(f"- **低噪声类别 ({len(low_noise)} 个, 评分 <5)**: 对训练影响不大，正常 CE Loss 即可")
    lines.append(f"  - 推荐: SCE alpha=0.1, beta=1.0 或直接 CE Loss")
    lines.append("")

    lines.append("## 后续建议")
    lines.append("")
    lines.append("1. 等 A 的 Baseline (ResNet-50) 完成后，运行混淆矩阵分析，交叉验证高噪声类别")
    lines.append("2. 对 Top-20 噪声类别做人工抽样复查（参考 `docs/noise_samples/` 中的 grid 图）")
    lines.append("3. 可尝试对这些类别做数据清洗（去重/去水印/修正标签）后再训练")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown 报告已保存: {md_path}")


def _write_csv(csv_path, results):
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "class_name", "total", "noise_score",
            "corrupted", "grayscale", "small", "low_contrast",
            "near_blank", "color_outlier", "color_var"
        ])
        for r in results:
            writer.writerow([
                r["class_id"],
                r.get("class_name", ""),
                r["total"],
                r["noise_score"],
                r["corrupted"],
                r["grayscale"],
                r["small"],
                r["low_contrast"],
                r["near_blank"],
                r["color_outlier"],
                r["color_var"],
            ])
    print(f"[OK] CSV 已保存: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())

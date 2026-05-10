"""
Week 2 — 噪声样本可视化 Grid

对 Top-N 噪声类别，生成 3x3 或 5x5 样本网格，供人工复查

用法:
  C:/Python312/python.exe scripts/visualize_noise.py \
      --noise-json ./docs/noise_scan.json \
      --data-root ./data/imagenet \
      --output-dir ./docs/noise_samples \
      --top-n 20 \
      --grid 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def create_sample_grid(
    class_id: str,
    class_name: str,
    image_dir: str,
    noisy_paths: list,
    output_path: str,
    grid_size: int = 5,
    random_sample: int = 50,
):
    """
    生成一个类别的噪声样本网格图

    Args:
        class_id: 类别 ID（如 n02089078）
        class_name: 类别名
        image_dir: 该类别的图片目录
        noisy_paths: 已知噪声图片路径列表
        output_path: PNG 输出路径
        grid_size: 每行/每列图片数（grid_size × grid_size）
        random_sample: 从目录随机抽样数
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import random
    random.seed(42)

    # 收集候选图片: 噪声图优先，再随机补充
    candidates = list(noisy_paths)

    # 从目录中随机抽样补充
    all_images = sorted(
        str(f) for f in Path(image_dir).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
    )
    random.shuffle(all_images)

    for path in all_images:
        if len(candidates) >= grid_size * grid_size:
            break
        if path not in candidates:
            candidates.append(path)

    n = min(len(candidates), grid_size * grid_size)
    if n == 0:
        print(f"  [SKIP] {class_id}: 无图片")
        return

    # 绘制网格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < n:
            try:
                img = Image.open(candidates[i]).convert("RGB")
                ax.imshow(img)
                # 噪声标记
                is_noisy = candidates[i] in noisy_paths
                border_color = "red" if is_noisy else "green"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3 if is_noisy else 1)
            except Exception:
                ax.text(0.5, 0.5, "LOAD ERR", ha="center", va="center", fontsize=8, color="red")
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{class_id} | {class_name}" if class_name else class_id
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {class_id}: {n} 张 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="噪声样本可视化 Grid")
    parser.add_argument("--noise-json", default="./docs/noise_scan.json", help="detect_noise.py 输出的 JSON")
    parser.add_argument("--data-root", default="./data/imagenet", help="ImageNet 根目录")
    parser.add_argument("--output-dir", default="./docs/noise_samples", help="Grid 图输出目录")
    parser.add_argument("--split", default="train", help="数据 split")
    parser.add_argument("--top-n", type=int, default=20, help="可视化前 N 个噪声类别")
    parser.add_argument("--grid", type=int, default=5, help="网格大小 (grid×grid)")
    args = parser.parse_args()

    # 加载扫描结果
    with open(args.noise_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_noisy = data.get("top_noisy", [])[:args.top_n]

    if not top_noisy:
        print("[ERROR] 未找到噪声数据")
        sys.exit(1)

    split_dir = Path(args.data_root) / args.split

    print(f"生成 Top-{len(top_noisy)} 噪声类别可视化 Grid ({args.grid}×{args.grid})...\n")

    for rank, cls_info in enumerate(top_noisy, 1):
        class_id = cls_info["class_id"]
        class_name = cls_info.get("class_name", "")
        image_dir = split_dir / class_id

        if not image_dir.exists():
            print(f"  [SKIP] {class_id}: 目录不存在")
            continue

        noisy_paths = cls_info.get("noisy_paths", [])
        fname = f"{rank:02d}_{class_id}.png"
        output_path = os.path.join(args.output_dir, fname)

        create_sample_grid(
            class_id=class_id,
            class_name=class_name,
            image_dir=str(image_dir),
            noisy_paths=noisy_paths,
            output_path=output_path,
            grid_size=args.grid,
        )

    print(f"\n全部完成，共 {len(top_noisy)} 个类别 → {args.output_dir}/")


if __name__ == "__main__":
    sys.exit(main())

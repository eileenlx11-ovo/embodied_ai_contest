"""
Week 2 — 噪声样本可视化 Grid

对 Top-N 噪声类别，生成 5x5 样本网格（无红绿框，干净版本）
排序按人工复查后的真实噪声严重程度

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

# 人工复查排名（按真实噪声严重程度）
MANUAL_RANK = {
    "n04041544": 1,   # radio — 高噪声
    "n03777754": 2,   # modem — 中等噪声
    "n01930112": 3,   # nematode — 中等噪声
    "n03759954": 4,   # microphone
    "n04270147": 5,   # spatula
    "n04127249": 6,   # safety pin
    "n03532672": 7,   # hook
    "n03691459": 8,   # loudspeaker
    "n02708093": 9,   # analog clock
    "n06785654": 10,  # crossword puzzle
    "n04040759": 11,  # radiator
    "n04141076": 12,  # sax
    "n03297495": 13,  # espresso maker
    "n02110627": 14,  # affenpinscher
    "n04228054": 15,  # ski
    "n03041632": 16,  # cleaver
    "n03492542": 17,  # hard disc
    "n04485082": 18,  # tripod
    "n04238763": 19,  # slide rule
    "n02672831": 20,  # accordion
}

MANUAL_LABEL = {
    "n04041544": "HIGH noise ~60%",
    "n03777754": "MED noise",
    "n01930112": "MED noise",
    "n03759954": "LOW (label ambiguity)",
    "n04270147": "LOW (subject mismatch)",
    "n04127249": "LOW (label ambiguity)",
    "n03532672": "LOW (label ambiguity)",
    "n03691459": "LOW (white bg FP)",
    "n02708093": "LOW (white bg FP)",
    "n06785654": "LOW (white bg FP)",
    "n04040759": "LOW (white bg FP)",
    "n04141076": "LOW (white bg FP)",
    "n03297495": "CLEAN (white bg FP)",
    "n02110627": "CLEAN (white bg FP)",
    "n04228054": "CLEAN (white bg FP)",
    "n03041632": "CLEAN (white bg FP)",
    "n03492542": "CLEAN (white bg FP)",
    "n04485082": "CLEAN (white bg FP)",
    "n04238763": "CLEAN (gray real)",
    "n02672831": "CLEAN (white bg FP)",
}


def create_sample_grid(
    class_id: str,
    class_name: str,
    image_dir: str,
    output_path: str,
    manual_label: str,
    grid_size: int = 5,
):
    """生成干净样本网格（无红绿框）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    import random
    random.seed(42)

    all_images = sorted(
        str(f) for f in Path(image_dir).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
    )
    random.shuffle(all_images)

    candidates = all_images[:grid_size * grid_size]
    n = len(candidates)

    if n == 0:
        print(f"  [SKIP] {class_id}: 无图片")
        return

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < n:
            try:
                img = Image.open(candidates[i]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "LOAD ERR", ha="center", va="center", fontsize=8, color="red")
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{class_id} | {class_name}\n[Manual review: {manual_label}]" if class_name else class_id
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {class_id}: {n} 张 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="噪声样本可视化 Grid（人工复查后干净版）")
    parser.add_argument("--noise-json", default="./docs/noise_scan.json")
    parser.add_argument("--data-root", default="./data/imagenet")
    parser.add_argument("--output-dir", default="./docs/noise_samples")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--grid", type=int, default=5)
    args = parser.parse_args()

    with open(args.noise_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_noisy = data.get("top_noisy", [])[:args.top_n]

    if not top_noisy:
        print("[ERROR] 未找到噪声数据")
        sys.exit(1)

    # 按人工复查排名重排
    top_noisy.sort(key=lambda x: MANUAL_RANK.get(x["class_id"], 99))

    split_dir = Path(args.data_root) / args.split

    print(f"生成 Top-{len(top_noisy)} 噪声类别可视化 Grid（人工复查后排序）...\n")

    for rank, cls_info in enumerate(top_noisy, 1):
        class_id = cls_info["class_id"]
        class_name = cls_info.get("class_name", "")
        image_dir = split_dir / class_id

        if not image_dir.exists():
            print(f"  [SKIP] {class_id}: 目录不存在")
            continue

        manual_label = MANUAL_LABEL.get(class_id, "")
        fname = f"{rank:02d}_{class_id}.png"
        output_path = os.path.join(args.output_dir, fname)

        create_sample_grid(
            class_id=class_id,
            class_name=class_name,
            image_dir=str(image_dir),
            output_path=output_path,
            manual_label=manual_label,
            grid_size=args.grid,
        )

    print(f"\n全部完成，共 {len(top_noisy)} 个类别 → {args.output_dir}/")


if __name__ == "__main__":
    sys.exit(main())

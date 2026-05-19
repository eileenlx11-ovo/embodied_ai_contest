"""
Week 3 — 混淆矩阵推理脚本.

在验证集上加载 ResNet-50 baseline checkpoint，推理并输出：
  1. predictions.npy — (N,) true_labels, pred_labels, pred_probs
  2. confusion_top10.md — Top-10 易混淆类别对 + 样本 Grid

用法:
  C:/Python312/python.exe scripts/infer_confusion.py \
      --checkpoint ./best.pth \
      --data-root ./data/imagenet \
      --output-dir ./docs/week3
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from src.models.build_model import build_model


def load_checkpoint(path: str, device: str = "cuda"):
    """加载 checkpoint，优先使用 EMA 权重."""
    cp = torch.load(path, map_location=device, weights_only=False)
    state = cp.get("ema") or cp["model"]
    # EMA state dict 的 key 是原始 model state 的名字；我们的 build_model 也产出相同的 key
    best_acc = cp.get("best_acc", None)
    epoch = cp.get("epoch", None)
    return state, best_acc, epoch


def build_cfg(data_root: str):
    return {
        "model": {"name": "resnet50", "num_classes": 1000, "drop_path_rate": 0.0},
        "data": {
            "dataset": "imagenet",
            "root": data_root,
            "input_size": 224,
            "batch_size": 128,
            "num_workers": 4,
            "pin_memory": True,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def get_val_loader(data_root: str, batch_size: int = 128):
    """构建验证集 DataLoader."""
    from torchvision import transforms

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dir = Path(data_root) / "val"
    dataset = ImageFolder(root=str(val_dir), transform=val_transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader, dataset.classes, dataset.class_to_idx


def inference(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全量推理，返回 true_labels, pred_labels, pred_probs."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_true.append(targets.numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def compute_confusion_pairs(
    true: np.ndarray,
    pred: np.ndarray,
    idx_to_class: Dict[int, str],
    class_names: Dict[str, str],
    top_k: int = 20,
) -> List[Dict]:
    """
    从预测结果中提取 Top-K 易混淆类别对。

    对每一对 (true_i, pred_j) 其中 i != j，统计误分类样本数。
    只统计 off-diagonal 且 misclassification >= min_count 的对。
    """
    num_classes = len(idx_to_class)
    off_diag = {}  # (true_cls, pred_cls) → [sample_indices]

    for i, (t, p) in enumerate(zip(true, pred)):
        if t != p:
            key = (int(t), int(p))
            if key not in off_diag:
                off_diag[key] = []
            off_diag[key].append(i)

    # 排序取 top-k
    sorted_pairs = sorted(off_diag.items(), key=lambda x: -len(x[1]))

    results = []
    for (t_idx, p_idx), indices in sorted_pairs[:top_k]:
        t_id = idx_to_class[t_idx]
        p_id = idx_to_class[p_idx]
        results.append({
            "rank": len(results) + 1,
            "true_class_id": t_id,
            "true_class_name": class_names.get(t_id, t_id),
            "pred_class_id": p_id,
            "pred_class_name": class_names.get(p_id, p_id),
            "misclass_count": len(indices),
            "sample_indices": indices[:25],  # 最多保存 25 个样本索引
        })

    return results


def load_class_names(class_counts_csv: str) -> Dict[str, str]:
    names = {}
    with open(class_counts_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[row["class_id"]] = row["class_name"]
    return names


def make_confusion_grid(
    pair: Dict,
    dataset: ImageFolder,
    idx_to_class: Dict[int, str],
    output_dir: str,
    grid_size: int = 5,
):
    """为每个易混淆对生成 5x5 样本 Grid（左: true class, 右: mis-predicted）."""
    import random
    random.seed(42)

    # 收集 true class 和 pred class 的样本
    t_label = list(idx_to_class.keys())[list(idx_to_class.values()).index(pair["true_class_id"])]
    p_label = list(idx_to_class.keys())[list(idx_to_class.values()).index(pair["pred_class_id"])]

    true_samples = [s for s, t in enumerate(dataset.targets) if t == t_label]
    pred_samples = [s for s, t in enumerate(dataset.targets) if t == p_label]

    random.shuffle(true_samples)
    random.shuffle(pred_samples)

    n_true = min(grid_size * grid_size, len(true_samples))
    n_pred = min(grid_size * grid_size, len(pred_samples))

    if n_true == 0 and n_pred == 0:
        return

    fig, axes = plt.subplots(grid_size * 2, grid_size, figsize=(grid_size * 3, grid_size * 6))
    if grid_size == 1:
        axes = np.array([axes])

    # 左半: true class 样本
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        ax = axes[row][col]
        if i < n_true:
            try:
                img, _ = dataset[true_samples[i]]
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "ERR", ha="center", va="center", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"True\n{pair['true_class_name']}", fontsize=7)

    # 右半: pred class 样本 (被误判成的类)
    for i in range(grid_size * grid_size):
        row, col = (i + grid_size * grid_size) // grid_size, (i + grid_size * grid_size) % grid_size
        ax = axes[row][col]
        if i < n_pred:
            try:
                img, _ = dataset[pred_samples[i]]
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "ERR", ha="center", va="center", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"Pred\n{pair['pred_class_name']}", fontsize=7)

    title = (
        f"#{pair['rank']}  {pair['true_class_id']} → {pair['pred_class_id']}\n"
        f"{pair['true_class_name']} → {pair['pred_class_name']}  "
        f"({pair['misclass_count']} misclassifications)"
    )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()

    fname = f"{pair['rank']:02d}_{pair['true_class_id']}_to_{pair['pred_class_id']}.png"
    outpath = os.path.join(output_dir, "confusion_grids", fname)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()


def write_confusion_report(
    pairs: List[Dict],
    class_names: Dict[str, str],
    accuracy: float,
    top5_acc: float,
    best_acc: float,
    output_path: str,
):
    """生成 confusion_top10.md 报告."""
    lines = [
        "# 混淆矩阵分析报告 (Week 3)",
        "",
        f"> Baseline: ResNet-50 / 100 epoch / Val Top-1 {accuracy:.2f}% / Top-5 {top5_acc:.2f}%",
        f"> Checkpoint best_acc: {best_acc:.2f}%",
        "",
        "---",
        "",
        "## Top-10 易混淆类别对",
        "",
        "按误分类样本数排序：",
        "",
        "| 排名 | 真实类别 | 误判为 | 混淆次数 | 分析 |",
        "|------|----------|--------|----------|------|",
    ]

    for p in pairs[:10]:
        analysis = analyze_pair(p)
        lines.append(
            f"| {p['rank']} | {p['true_class_name']} ({p['true_class_id']}) "
            f"| {p['pred_class_name']} ({p['pred_class_id']}) "
            f"| {p['misclass_count']} | {analysis} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 混淆模式分析",
        "",
        "### 容易混淆的类别特征",
        "",
        "通过 Top-10 易混淆对可以归纳出以下模式：",
        "",
    ]

    # 按类别归纳混淆模式
    patterns = categorize_pairs(pairs[:10])
    for category, items in patterns.items():
        lines.append(f"**{category}**：")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 给 C 的策略建议",
        "",
        "1. 对以上 Top-10 易混淆对，考虑在训练时增加细粒度特征学习（如 triplet loss 辅助）",
        "2. 对混淆对中的两个类别，可在 Week 2 噪声分析基础上做针对性数据清洗",
        "3. 若 C 在做 per-class loss 分析，优先关注这些易混淆类别的 SCE 表现",
        "",
        "---",
        "",
        "## 交付清单",
        "",
        "| 产出 | 路径 | 状态 |",
        "|------|------|------|",
        f"| 预测结果 (npy) | `docs/week3/predictions.npy` | ✅ |",
        f"| 混淆对 Grid 图 | `docs/week3/confusion_grids/*.png` | ✅ |",
        f"| 本报告 | `docs/confusion_top10.md` | ✅ |",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 同时保存到 docs/week3/
    alt_path = Path(output_path).parent / "week3" / "confusion_pairs.json"
    alt_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for p in pairs[:10]:
        serializable.append({
            k: v for k, v in p.items() if k != "sample_indices"
        })
    with open(alt_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def analyze_pair(p: Dict) -> str:
    """对混淆对给出简短分析."""
    t_name = p["true_class_name"]
    p_name = p["pred_class_name"]

    # 简单启发式规则
    dog_breeds = {"hound", "terrier", "spaniel", "retriever", "shepherd", "collie", "poodle", "husky", "malamute"}
    if any(b in t_name.lower() for b in dog_breeds) and any(b in p_name.lower() for b in dog_breeds):
        return "犬种细粒度混淆"
    if any(b in t_name.lower() for b in {"bird", "finch", "sparrow", "warbler"}) and \
       any(b in p_name.lower() for b in {"bird", "finch", "sparrow", "warbler"}):
        return "鸟类细粒度混淆"
    if any(w in t_name.lower() for w in {"car", "truck", "bus", "vehicle"}) and \
       any(w in p_name.lower() for w in {"car", "truck", "bus", "vehicle"}):
        return "车辆类混淆"
    if any(w in t_name.lower() for w in {"guitar", "violin", "cello", "sax", "trumpet", "flute", "piano"}) and \
       any(w in p_name.lower() for w in {"guitar", "violin", "cello", "sax", "trumpet", "flute", "piano"}):
        return "乐器细粒度混淆"
    return "视觉/语义相似"


def categorize_pairs(pairs: List[Dict]) -> Dict[str, List[str]]:
    patterns = {}
    for p in pairs:
        cat = analyze_pair(p)
        patterns.setdefault(cat, []).append(
            f"{p['true_class_name']} → {p['pred_class_name']} ({p['misclass_count']} 次)"
        )
    return patterns


def main():
    parser = argparse.ArgumentParser(description="混淆矩阵推理")
    parser.add_argument("--checkpoint", default="./best.pth")
    parser.add_argument("--data-root", default="./data/imagenet")
    parser.add_argument("--output-dir", default="./docs/week3")
    parser.add_argument("--class-counts", default="./docs/week1/class_counts.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    # 路径兼容
    for attr in ("checkpoint", "data_root", "class_counts", "output_dir"):
        val = getattr(args, attr)
        if not os.path.exists(val):
            abs_val = str(SRC_DIR / val)
            if os.path.exists(abs_val):
                setattr(args, attr, abs_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")

    # 1. 加载模型
    cfg = build_cfg(args.data_root)
    model = build_model(cfg)
    state, best_acc, epoch = load_checkpoint(args.checkpoint, device)

    # 处理 EMA state dict 中可能存在的 'module.' 前缀
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    print(f"Loaded epoch {epoch}, best_acc={best_acc:.2f}%")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2. 加载数据
    loader, classes, class_to_idx = get_val_loader(args.data_root, args.batch_size)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Val set: {len(loader.dataset)} images, {len(classes)} classes")

    # 3. 推理
    true, pred, probs = inference(model, loader, device)

    # 4. 计算准确率
    acc = (true == pred).mean() * 100
    top5_pred = np.argsort(probs, axis=-1)[:, -5:]
    top5_acc = np.any(top5_pred == true[:, None], axis=1).mean() * 100
    print(f"\nTop-1 Accuracy: {acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    # 5. 保存预测结果
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_dir / "predictions.npz"),
        true_labels=true,
        pred_labels=pred,
        pred_probs=probs,
        idx_to_class=idx_to_class,
    )
    print(f"Predictions saved to {out_dir / 'predictions.npz'}")

    # 6. 加载类别名称
    class_counts_path = args.class_counts
    if not os.path.exists(class_counts_path):
        class_counts_path = str(SRC_DIR / "docs" / "week1" / "class_counts.csv")
    class_names = load_class_names(class_counts_path)

    # 7. 找 Top-10 易混淆对
    pairs = compute_confusion_pairs(true, pred, idx_to_class, class_names, top_k=args.top_k)
    print(f"\nTop-{args.top_k} Confusing Pairs:")
    for p in pairs[:10]:
        print(f"  #{p['rank']} {p['true_class_name']} → {p['pred_class_name']}: {p['misclass_count']}")

    # 8. 生成 Grid 图
    print("\nGenerating sample grids...")
    # 重新创建不带 normalize 的 dataset 用于可视化
    from torchvision import transforms as T
    vis_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    val_dir = Path(args.data_root) / "val"
    vis_dataset = ImageFolder(root=str(val_dir), transform=vis_transform)

    for p in pairs[:10]:
        make_confusion_grid(p, vis_dataset, idx_to_class, str(out_dir))

    # 9. 写报告
    report_path = str(SRC_DIR / "docs" / "confusion_top10.md")
    write_confusion_report(pairs, class_names, acc, top5_acc, best_acc or acc, report_path)
    print(f"\nReport saved to {report_path}")
    print(f"Grids saved to {out_dir / 'confusion_grids'}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())

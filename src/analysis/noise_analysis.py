"""
噪声分析模块 — 多信号噪声检测（不依赖已训练模型）

支持的检测信号:
  1. 文件损坏检测     — PIL 能否正常解码
  2. 图像质量检测     — 分辨率过低 / 对比度异常 / 近空白图
  3. 颜色分布离群检测 — 类内颜色直方图偏差
  4. 格式异常检测     — 灰度图 / 异常宽高比

评分设计:
  noise_score ∈ [0, 100]，直观含义 = "估计问题图片占比 (%)"
  - <2%  → 极干净
  - 2-5% → 低噪声
  - 5-10% → 中等噪声
  - 10-20% → 显著噪声
  - >20% → 高噪声，建议重点清洗

用法:
  from src.analysis.noise_analysis import scan_class, scan_all_classes, rank_noisy_classes
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageStat


# ---------- 单张图片检测 ----------

def check_image_valid(path: str) -> Dict:
    """
    检测单张图片是否存在技术问题
    Returns: {is_valid, width, height, is_grayscale, is_low_contrast, is_small, error}
    """
    result = {
        "path": path,
        "is_valid": True,
        "width": 0,
        "height": 0,
        "is_grayscale": False,
        "is_low_contrast": False,
        "is_small": False,
        "is_near_blank": False,
        "aspect_ratio": 0.0,
        "error": None,
    }

    try:
        with Image.open(path) as img:
            img.load()
            result["width"], result["height"] = img.size
            result["aspect_ratio"] = img.size[0] / max(img.size[1], 1)

            if img.mode == "L":
                result["is_grayscale"] = True
            elif img.mode == "P" or (img.mode == "RGB" and _is_effectively_grayscale(img)):
                result["is_grayscale"] = True

            if img.size[0] < 64 or img.size[1] < 64:
                result["is_small"] = True

            stat = ImageStat.Stat(img)
            if len(stat.stddev) >= 3:
                avg_std = sum(stat.stddev[:3]) / 3
            else:
                avg_std = stat.stddev[0] if stat.stddev else 0
            if avg_std < 15:
                result["is_low_contrast"] = True
            if avg_std < 5:
                result["is_near_blank"] = True

    except Exception as e:
        result["is_valid"] = False
        result["error"] = str(e)

    return result


def _is_effectively_grayscale(img: Image.Image, threshold: float = 5.0) -> bool:
    """检测 RGB 图是否实质上是灰度图"""
    if img.mode != "RGB":
        return False
    w, h = img.size
    crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).resize((50, 50))
    arr = np.array(crop, dtype=np.float32)
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
    diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2]).mean()
    return diff_rg < threshold and diff_rb < threshold


# ---------- 颜色特征 ----------

def compute_color_histogram(path: str, bins: int = 32) -> Optional[np.ndarray]:
    """计算归一化 RGB 颜色直方图（96-dim 向量）"""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((128, 128))
            arr = np.array(img, dtype=np.float32)
            hist = []
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=True)
                hist.append(h)
            return np.concatenate(hist)
    except Exception:
        return None


# ---------- 类级别扫描 ----------

def scan_class(
    class_dir: str,
    sample_size: int = 200,
) -> Dict:
    """
    扫描一个类别的所有图片，返回噪声指标

    抽样 sample_size 张做图片质量检测，结果外推到全类别。
    noise_score = 估计的问题图片占比 (%)
    """
    class_dir = Path(class_dir)
    class_id = class_dir.name

    image_files = sorted(
        f for f in class_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")
    )

    if not image_files:
        return {"class_id": class_id, "total": 0, "error": "empty"}

    total = len(image_files)

    # 均匀抽样 sample_size 张
    step = max(1, total // sample_size)
    sample_indices = list(range(0, total, step))[:sample_size]
    sample_paths = [image_files[i] for i in sample_indices]
    actual_sample = len(sample_paths)

    # 逐张检测
    corrupted = 0
    grayscale_count = 0
    small_count = 0
    low_contrast_count = 0
    blank_count = 0
    noisy_samples = []

    color_features = []
    for img_path in sample_paths:
        path_str = str(img_path)
        result = check_image_valid(path_str)

        issues = []

        if not result["is_valid"]:
            corrupted += 1
            issues.append("corrupted")
        else:
            if result["is_grayscale"]:
                grayscale_count += 1
                issues.append("grayscale")
            if result["is_small"]:
                small_count += 1
                issues.append("small")
            if result["is_low_contrast"]:
                low_contrast_count += 1
                issues.append("low_contrast")
            if result["is_near_blank"]:
                blank_count += 1
                issues.append("near_blank")

            feat = compute_color_histogram(path_str)
            if feat is not None:
                color_features.append(feat)

        if issues:
            noisy_samples.append((path_str, issues))

    # 颜色离群检测
    outlier_count = 0
    color_var = 0.0
    if len(color_features) > 5:
        features = np.array(color_features)
        centroid = features.mean(axis=0)
        distances = np.linalg.norm(features - centroid, axis=1)
        color_var = float(distances.mean())
        mean_d, std_d = distances.mean(), distances.std()
        if std_d > 0:
            threshold = mean_d + 2.0 * std_d
            outlier_count = int((distances > threshold).sum())

    # 综合噪声评分 = 估计问题图片占比 (%)
    # 损坏图片: 权重 1.0（每张损坏图贡献 100/actual_sample %）
    # 近空白图: 权重 0.8
    # 低对比度 + 小图: 权重 0.5
    # 灰度图: 权重 0.2
    if actual_sample == 0:
        noise_score = 0.0
    else:
        weighted_issues = (
            corrupted * 1.0 +
            blank_count * 0.8 +
            low_contrast_count * 0.5 +
            small_count * 0.5 +
            grayscale_count * 0.2 +
            outlier_count * 0.3
        )
        noise_score = round(weighted_issues / actual_sample * 100, 2)

    # 估算全类别问题数
    est_corrupted_total = int(corrupted / actual_sample * total) if actual_sample else 0

    return {
        "class_id": class_id,
        "total": total,
        "sampled": actual_sample,
        "corrupted": corrupted,
        "est_total_corrupted": est_corrupted_total,
        "grayscale": grayscale_count,
        "small": small_count,
        "low_contrast": low_contrast_count,
        "near_blank": blank_count,
        "color_outlier": outlier_count,
        "color_var": round(color_var, 4),
        "noise_score": noise_score,
        "noisy_paths": [p for p, _ in noisy_samples[:15]],
    }


# ---------- 全数据集扫描 ----------

def scan_all_classes(
    data_root: str,
    split: str = "train",
    class_names: Optional[Dict[str, str]] = None,
    sample_size: int = 200,
    max_classes: Optional[int] = None,
) -> List[Dict]:
    """遍历所有类别执行 scan_class，返回按 noise_score 降序排列的列表"""
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"目录不存在: {split_dir}")

    class_dirs = sorted(
        d for d in split_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if max_classes:
        class_dirs = class_dirs[:max_classes]

    names = class_names or {}
    results = []
    for i, class_dir in enumerate(class_dirs):
        result = scan_class(str(class_dir), sample_size=sample_size)
        result["class_name"] = names.get(result["class_id"], "")
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(class_dirs)}] 已扫描...")

    results.sort(key=lambda x: x["noise_score"], reverse=True)
    return results


# ---------- 工具 ----------

def rank_noisy_classes(results: List[Dict], top_n: int = 20) -> Tuple[List[Dict], List[Dict]]:
    """返回 top_n 最高噪声和最干净的类别"""
    top = results[:top_n]
    bottom = results[-top_n:][::-1]
    return top, bottom


def get_loss_tuning_hints(noise_score: float) -> str:
    """根据噪声评分给出 SCE / GCE Loss 调参建议"""
    if noise_score < 2:
        return "极低噪声，CE Loss 或 SCE alpha=0.1, beta=1.0"
    elif noise_score < 5:
        return "低噪声，SCE alpha=0.3, beta=0.8 或 GCE q=0.7"
    elif noise_score < 10:
        return "中等噪声，SCE alpha=0.5, beta=0.5 或 GCE q=0.5"
    elif noise_score < 20:
        return "显著噪声，SCE alpha=0.7, beta=0.3，建议 warmup=5 epoch"
    else:
        return "高噪声，ELR + SCE 混合，alpha=1.0, beta=0.1，warmup=10 epoch"

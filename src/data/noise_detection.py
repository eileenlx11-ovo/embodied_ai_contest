"""
噪声检测方法模块
实现基于 Loss 的噪声检测：
  原理: 模型先记忆干净样本（小 loss），后记忆噪声样本（大 loss）
  方法:
    1. 在 warm-up 阶段（5-10 epoch）记录每个样本的累计 loss
    2. warm-up 后按 loss 排序:
       - 小 loss (bottom 20%) → 高置信度干净样本
       - 大 loss (top 20%) → 疑似噪声样本
    3. 可视化 loss 分布直方图
    4. 输出: clean_indices, noisy_indices

用法：
  from src.data.noise_detection import LossBasedNoiseDetector
  detector = LossBasedNoiseDetector()
  detector.record_losses(epoch, sample_indices, losses)
  clean_idx, noisy_idx = detector.detect()
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LossBasedNoiseDetector:
    """
    基于 Loss 的噪声检测器
    适合在训练 warm-up 阶段使用
    """

    def __init__(
        self,
        num_samples: int,
        warmup_epochs: int = 5,
        noise_ratio: float = 0.2,
        clean_ratio: float = 0.2,
    ):
        """
        Args:
            num_samples: 总样本数
            warmup_epochs: 预热轮数，收集这些 epoch 的 loss 后执行检测
            noise_ratio: 大 loss 样本比例（疑似噪声），默认 top 20%
            clean_ratio: 小 loss 样本比例（高置信度干净），默认 bottom 20%
        """
        self.num_samples = num_samples
        self.warmup_epochs = warmup_epochs
        self.noise_ratio = noise_ratio
        self.clean_ratio = clean_ratio

        # 每个样本的累计 loss 列表
        self.loss_history = defaultdict(list)
        self.mean_losses: Optional[np.ndarray] = None

    def record_losses(self, sample_indices: np.ndarray, losses: np.ndarray):
        """记录一个 batch 的样本 loss"""
        for idx, loss in zip(sample_indices, losses):
            self.loss_history[int(idx)].append(float(loss))

    def detect(self) -> Tuple[List[int], List[int], np.ndarray]:
        """
        执行噪声检测
        Returns:
            clean_indices: 相对干净的样本索引列表
            noisy_indices: 疑似噪声的样本索引列表
            mean_losses: 所有样本的平均 loss 数组
        """
        # 计算每个样本的平均 loss
        all_indices = sorted(self.loss_history.keys())
        mean_losses = np.array([np.mean(self.loss_history[i]) for i in all_indices])

        # 按 loss 排序
        sorted_order = np.argsort(mean_losses)

        n_clean = int(len(all_indices) * self.clean_ratio)
        n_noisy = int(len(all_indices) * self.noise_ratio)

        clean_indices = [all_indices[i] for i in sorted_order[:n_clean]]
        noisy_indices = [all_indices[i] for i in sorted_order[-n_noisy:]]
        all_mean_losses = np.array([np.mean(self.loss_history.get(i, [0])) for i in range(self.num_samples)])

        self.mean_losses = all_mean_losses

        return clean_indices, noisy_indices, all_mean_losses

    def get_loss_stats(self) -> Dict:
        """获取 loss 分布的统计信息"""
        if self.mean_losses is None:
            raise ValueError("请先调用 detect() 方法")

        valid = self.mean_losses[self.mean_losses > 0]
        if len(valid) == 0:
            return {"error": "无有效 loss 数据"}

        return {
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "median": float(np.median(valid)),
            "bimodality_hint": "可能双峰" if (valid.std() / valid.mean()) > 0.5 else "单峰",
        }

    def reset(self):
        """清空记录，准备下一轮"""
        self.loss_history.clear()
        self.mean_losses = None

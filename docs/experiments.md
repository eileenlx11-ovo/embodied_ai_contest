# Experiment Log

> 所有实验结果记录。格式：日期 / 配置 / 结果 / 备注

## Hardware

| Machine | GPU | VRAM | Notes |
|---------|-----|------|-------|
| AutoDL (乔) | RTX 4090 D | 24GB | 月租，主力训练 |
| 本地 (乔) | RTX 4060 | 8GB | 辅助实验 |

## Baseline Experiments

### ResNet-50 + CE + Label Smoothing 0.1

| Date | Subset | Epochs | Train Acc | Val Top-1 | Val Top-5 | Notes |
|------|--------|--------|-----------|-----------|-----------|-------|
| 5/15 | 10% | 30 | TBD | TBD | TBD | 数据修复后首次正式运行 |
| TBD | 100% | 100 | TBD | TBD | TBD | 正式 baseline |

## Loss Comparison (10% subset, 30 epochs, ResNet-50)

| Loss | Params | Train Acc | Val Top-1 | Val Top-5 | Notes |
|------|--------|-----------|-----------|-----------|-------|
| CE + LS(0.1) | smoothing=0.1 | TBD | TBD | TBD | baseline |
| SCE | α=0.1, β=1.0 | TBD | TBD | TBD | |
| SCE | α=0.5, β=0.5 | TBD | TBD | TBD | |
| GCE | q=0.7 | TBD | TBD | TBD | |
| ELR | β=0.9, λ=3.0 | TBD | TBD | TBD | |

## Full Training Runs

| Date | Model | Loss | Epochs | Val Top-1 | Val Top-5 | Checkpoint | Notes |
|------|-------|------|--------|-----------|-----------|------------|-------|
| TBD | ResNet-50 | Best Loss | 100 | TBD | TBD | best.pth | 正式提交模型 |

## Key Findings

1. **数据集不是长尾分布**：不平衡比仅 1.78（732~1300 张/类），无需特殊长尾处理
2. **数据质量高**：损坏 0 张，近空白 0 张，主要噪声来自标签歧义（radio ~60%）
3. **10% subset 不足以评估 val**：12816 张训练图 + 30 epochs 不够泛化到 50000 val

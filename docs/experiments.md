# Experiment Log

> 所有实验结果记录。格式：日期 / 配置 / 结果 / 备注

## Hardware

| Machine | GPU | VRAM | Notes |
|---------|-----|------|-------|
| AutoDL (乔) | RTX 4090 D | 24GB | 月租，全部正式训练在此完成 |
| 本地 (乔) | RTX 4060 | 8GB | 仅做数据分析，未参与正式训练 |

> ⚠️ 技术报告 / 答辩切勿对外提"4060 训练"——所有 baseline / 消融结果均产自 4090D，应据实陈述。

## Baseline Experiments

### ResNet-50 + CE + Label Smoothing 0.1 (full ImageNet-1K, 100 epochs)

| Date | Subset | Epochs | Train Acc | Val Top-1 | Val Top-5 | Notes |
|------|--------|--------|-----------|-----------|-----------|-------|
| 2026-05-18 | 100% | 100 | 79.50% | **77.53%** | **93.63%** | 正式 baseline（best.pth, log: train_full.log）|

配置：`configs/imagenet_resnet50.yaml`
- SGD 0.1, weight_decay 1e-4, momentum 0.9
- cosine schedule, 5 epoch warmup
- batch 128 × accum 2 = effective 256
- AMP + channels_last + EMA(0.9999)
- 单 epoch ~1800s，total ~50h on 4090D

✅ 已超过备赛计划的"目标线"76%，距"冲刺线"78% 仅差 0.47 pp。

### 历史失败实验（不计入结果）

| Date | 日志 | 表象 | 真实原因 |
|------|------|------|----------|
| 2026-05-15 | full_resnet50_ce_100ep.out | Train 90.45% / Val 0.10% | val 目录结构未按 ImageFolder 修复，类索引错配 |
| 2026-05-15 | full_imagenet_1.28M_100ep.out | epoch 7 处 FileNotFoundError | 同上，val 路径缺失 |
| 2026-05-15 | full_resnet50_ce_100ep_ema_fixed.out | epoch 7 处中断 | EMA fix 测试，未完成 |

## Loss Comparison

⚠️ **2026-05-11~12 的 SCE/GCE/ELR 实验全部无效**：当时 val 目录未修复，所有结果 Val Top-1 ≈ 0.1~4%（与随机猜测同量级，与训练 loss/acc 表现完全脱钩）。不能据此判断任何 loss 函数的优劣，需在 val 修复后重做。

| Loss | Params | 历史"结果"（无效） | 待重做 |
|------|--------|--------------------|--------|
| CE + LS(0.1) | smoothing=0.1 | — | baseline 已 100ep |
| SCE | α=0.1, β=1.0 | 0.16% (invalid) | ☐ |
| SCE | α=0.5, β=0.5 | 4.07% (invalid) | ☐ |
| SCE | α=0.7, β=1.0 | invalid | ☐ |
| GCE | q=0.5/0.7/0.9 | 0.14~0.15% (invalid) | ☐ |
| ELR | β=0.9, λ=3.0 | incomplete | ☐ |

### 重做方案（建议）
- 10% subset + 30 epoch 太短，3 个不同 loss 都没收敛到可比较点
- 建议 30% subset + 40 epoch，或直接 100% 50 epoch（4090D 约 25h/实验）
- 优先级：SCE(α=0.1, β=1.0) → GCE(q=0.7) → ELR(β=0.9, λ=3.0)
- 评估指标：Val Top-1 + 在已知噪声子类（如 'radio' 类 ~60% 标签错）上的预测分布

## Augmentation Ablation

| Date | Config | Result | Notes |
|------|--------|--------|-------|
| 2026-05-18 | resnet50_mixup_randaug.yaml | 训练中 | Mixup(α=0.2) + CutMix(α=1.0) + RandAug(2,9) + RandomErasing |

预期：突破 baseline 77.53%，目标 ≥ 78.0%（冲刺线）。

## Full Training Runs

| Date | Model | Loss | Epochs | Val Top-1 | Val Top-5 | Checkpoint | Notes |
|------|-------|------|--------|-----------|-----------|------------|-------|
| 2026-05-18 | ResNet-50 | CE+LS | 100 | 77.53% | 93.63% | checkpoints/best.pth | baseline，已提交候选 |

## Key Findings

1. **数据集不是长尾分布**：不平衡比仅 1.78（732~1300 张/类），无需特殊长尾处理
2. **数据质量高**：损坏 0 张，近空白 0 张，主要噪声来自标签歧义（radio ~60%）
3. **val 必须按 ImageFolder 类目录组织**：扁平 val 目录会让训练全程显示 Val ≈ 0%（已踩坑）
4. **历史 loss 对比实验需全部重做**：均跑在 val 损坏期间，无信号

## 待办

- [ ] mixup 训练完成后评估
- [ ] 噪声鲁棒 loss 重做（赛题主观分 30% 的核心支撑）
- [ ] 接入赛题提供的 100K 测试集（当前 `data/imagenet/` 下未见 test 目录）
- [ ] 确认提交 CSV 格式（逗号分隔，是否带空格、是否带 header）

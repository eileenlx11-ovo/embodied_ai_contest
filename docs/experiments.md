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

## Augmentation Ablation

### 单项增强消融（30 epoch quick screen）

| Date | Config | Epochs | Train Acc | Val Top-1 | Val Top-5 | Notes |
|------|--------|--------|-----------|-----------|-----------|-------|
| 2026-05-23 | `configs/imagenet_resnet50_mixup_only.yaml` | 30 | 35.03% | **69.78%** | **89.59%** | Mixup-only，30 epoch 短期筛选结果 |
| 2026-05-23 | `configs/imagenet_resnet50_cutmix_only.yaml` | 30 | 49.53% | **69.12%** | **89.42%** | CutMix-only，30 epoch 短期筛选结果 |

结论：

- 30 epoch 短期消融中，Mixup-only Top-1 比 CutMix-only 高 **0.66 pp**，Top-5 高 **0.17 pp**。
- 这两个 30 epoch 结果不能直接与 100 epoch baseline 的 77.53% 公平比较，只用于增强策略筛选。
- 若后续继续增强方向，优先考虑 **Mixup-only**；CutMix-only 暂不作为优先延长训练对象。

### 完整强增强组合（暂停）

| Date | Config | Result | Notes |
|------|--------|--------|-------|
| 2026-05-22 | `configs/imagenet_resnet50_mixup.yaml` | 已停止 | 原配置同时启用 Mixup + CutMix + RandAug + RandomErasing，训练成本高且不适合作为单项消融 |

## Loss Comparison

⚠️ **2026-05-11~12 的 SCE/GCE/ELR 实验全部无效**：当时 val 目录未修复，所有结果 Val Top-1 ≈ 0.1~4%（与随机猜测同量级，与训练 loss/acc 表现完全脱钩）。不能据此判断任何 loss 函数的优劣，需在 val 修复后重做。

| Loss | Params | 历史"结果"（无效） | 当前状态 |
|------|--------|--------------------|----------|
| CE + LS(0.1) | smoothing=0.1 | — | 30% subset 对照已完成：66.63% Top-1 / 87.16% Top-5 |
| SCE | α=0.1, β=1.0 | 0.16% (invalid) | 待重做 |
| SCE | α=0.5, β=0.5 | 4.07% (invalid) | 待重做 |
| GCE | q=0.7 | 0.14~0.15% (invalid) | 30% subset 已完成：26.01% Top-1，明显低于 CE |
| ELR | β=0.9, λ=3.0 | incomplete | 可选 |

### 30% subset + 40 epoch 重做方案

目标：在更可信的数据规模下比较 CE、SCE、GCE、ELR 的收敛与验证表现，为最终报告中的噪声鲁棒性结论提供依据。

统一设置：

- config: `configs/imagenet_resnet50_noema_full.yaml`
- data root: `/root/autodl-tmp/imagenet_full/ILSVRC/Data/CLS-LOC`
- subset: `0.3`
- epochs: `40`
- warmup_epochs: `3`
- eval_interval: `5`
- EMA: off
- Mixup/CutMix: off

推荐执行顺序：

1. CE + Label Smoothing 对照组
2. GCE q=0.7
3. SCE α=0.1, β=1.0
4. SCE α=0.5, β=0.5
5. ELR β=0.9, λ=3.0（可选）

命令：

```bash
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss ce --run-name rerun30_ce_ls01
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss gce --gce-q 0.7 --run-name rerun30_gce_q07
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss sce --sce-alpha 0.1 --sce-beta 1.0 --run-name rerun30_sce_a01_b10
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss sce --sce-alpha 0.5 --sce-beta 0.5 --run-name rerun30_sce_a05_b05
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss elr --run-name rerun30_elr_b09_l30
```

结果记录表：

| run_name | Loss | Params | Epochs | Best Val Top-1 | Final Val Top-1 | Final Val Top-5 | Final Train Acc | NaN/Inf | Conclusion |
|----------|------|--------|--------|----------------|-----------------|-----------------|-----------------|---------|------------|
| rerun30_ce_ls01 | CE+LS | smoothing=0.1 | 40 | **66.63%** | **66.63%** | **87.16%** | 73.71% | No | 30% subset 对照组，作为 loss rerun 基准线 |
| rerun30_gce_q07 | GCE | q=0.7 | 40 | 26.01% | 26.01% | 34.56% | 26.26% | No | 明显低于 CE+LS，不作为后续 full training / GCE+Mixup 优先候选 |
| rerun30_sce_a01_b10 | SCE | α=0.1, β=1.0 | 40 | 60.66% | 60.66% | 80.99% | 61.92% | No | 低于 CE+LS，不作为最终模型优先候选 |
| rerun30_sce_a05_b05 | SCE | α=0.5, β=0.5 | 40 | **66.63%** | **66.63%** | 86.66% | 72.68% | No | Top-1 与 CE+LS 持平，但 Top-5 略低；可作为接近 CE 的鲁棒 loss 结果 |
| rerun30_elr_b09_l30 | ELR | β=0.9, λ=3.0 | 40 | TBD | TBD | TBD | TBD | TBD | 可选 |

汇总命令：

```bash
python scripts/plot_loss_sweep.py --input logs --output docs --runs \
  rerun30_ce_ls01 rerun30_gce_q07 rerun30_sce_a01_b10 rerun30_sce_a05_b05 rerun30_elr_b09_l30
```

### 30% subset loss rerun 结论

当前已完成 CE+LS、GCE 和两组 SCE 对比：

| Loss | Params | Best Val Top-1 | Final Val Top-5 | 结论 |
|------|--------|----------------|-----------------|------|
| CE+LS | smoothing=0.1 | **66.63%** | **87.16%** | 当前最稳，对照基准 |
| SCE | α=0.5, β=0.5 | **66.63%** | 86.66% | Top-1 追平 CE，但 Top-5 略低 |
| SCE | α=0.1, β=1.0 | 60.66% | 80.99% | 明显低于 CE |
| GCE | q=0.7 | 26.01% | 34.56% | 明显欠学习，不适合当前设置 |

结论：30% subset + 40 epoch 对比中，没有鲁棒 loss 明显优于 CE+LS。SCE α=0.5, β=0.5 能在 Top-1 上追平 CE，可作为“接近 CE 的鲁棒 loss”写入报告；GCE q=0.7 训练准确率和验证准确率均明显偏低，说明该配置在从零训练 ImageNet-1K 时学习信号不足。后续不优先进行 GCE+Mixup 或 SCE+Mixup，最终模型仍以 baseline / Mixup-only 方向为主。

## Full Training Runs

| Date | Model | Loss | Epochs | Val Top-1 | Val Top-5 | Checkpoint | Notes |
|------|-------|------|--------|-----------|-----------|------------|-------|
| TBD | ResNet-50 | Best Strategy | 100 | TBD | TBD | best.pth | 正式提交模型待定 |
| 2026-05-18 | ResNet-50 | CE+LS | 100 | 77.53% | 93.63% | checkpoints/best.pth | baseline，当前保底提交候选 |

## Key Findings

1. **Baseline 已可作为保底**：ResNet-50 + CE + LS + EMA 达到 77.53% Top-1 / 93.63% Top-5。
2. **Mixup-only 在 30 epoch 消融中略优于 CutMix-only**：69.78% vs 69.12%，后续增强方向优先考虑 Mixup。
3. **30% subset loss rerun 中没有鲁棒 loss 明显优于 CE+LS**：CE+LS 达到 66.63% Top-1 / 87.16% Top-5；SCE α=0.5, β=0.5 的 Top-1 同为 66.63%，但 Top-5 为 86.66%，略低于 CE；SCE α=0.1, β=1.0 为 60.66% Top-1；GCE q=0.7 仅 26.01% Top-1，明显欠学习。
4. **完整强增强组合暂停**：Mixup + CutMix + RandAug + RandomErasing 成本较高，且不适合作为单项消融解释。
5. **数据集不是严重长尾分布**：不平衡比仅 1.78（732~1300 张/类），无需特殊长尾处理。
6. **主要噪声来自标签歧义**：radio 等类别存在类内标签噪声；maillot / ear / laptop-notebook 等更多是标签定义或类间混淆问题。
7. **历史 loss 对比实验需全部重做**：均跑在 val 损坏期间，无信号。

## 待办

- [x] Mixup-only 30 epoch 完成并记录
- [x] CutMix-only 30 epoch 完成并记录
- [ ] 噪声鲁棒 loss 重做：30% subset + 40 epoch
- [ ] 接入赛题提供的 100K 测试集
- [ ] 确认提交 CSV 格式（逗号分隔，是否带 header）
- [ ] 最终 checkpoint 确定后生成 `submit/result.csv`

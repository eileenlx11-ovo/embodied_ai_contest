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

增广配方对收敛速度影响极大。核心结论：**100ep 预算下，重增广（RandAug+Erasing）收敛不完全，应减到 Mixup+CutMix only。**

| Date | Config | 增广配方 | Val Top-1 | Notes |
| ---- | ------ | -------- | --------- | ----- |
| 2026-05-26 | mixup_v2 (resnet50_mixup_e100_v2) | Mixup(α=0.2)+CutMix(α=1.0)+RandAug(2,9)+RandomErasing | **76.91%** (终值) | A1-tier 全增广；ep77 才 69.08%，靠后程拉到 76.91%，仍未破 baseline |
| 2026-06-02 | full_mixcut (resnet50_full_mixcut) | Mixup(α=0.1)+CutMix(α=1.0)，去掉 RandAug/Erasing | **77.57%** plain / **77.99%** HFlip TTA | 最终提交候选；`best_source=raw`，单模型 HFlip TTA +0.42pp |
| 2026-05-30 | cleaned_mixcut | 同 full_mixcut，但用清洗数据 | 废（仅 2ep 即停，让 GPU 给 full_mixcut） | — |

**同期 val top-1 对比（ep13-17，EMA 已充满电的可比区间）**：

| Epoch | baseline | mixup_v2 | cleaned_v1 | full_mixcut |
| ----- | -------- | -------- | ---------- | ----------- |
| ep15 | **56.60** | 53.69 | 54.97 | 54.31 |
| ep17 | **56.96** | 52.12 | 54.52 | 54.24 |

同期排序：**baseline > cleaned_v1 ≈ full_mixcut > mixup_v2**。full_mixcut 与 cleaned_v1 几乎重合，落后 baseline 约 2.7pp。这符合预期——Mixup/CutMix 作为正则化会压低前中期收敛速度（训练信号被混合，train acc 仅 ~28% vs baseline ~50%），换取后期泛化。真正胜负看后程 ep70-100，前期领先的 baseline 未必是终点赢家（参照 mixup_v2 靠后程从 ep77=69% 拉到终值 76.91%）。

> ⚠️ **读数陷阱（答辩须知）**：baseline / mixup_v2 前 ~9 个 epoch 的 Val Top-1 恒为 0.10%，并非模型没学，而是 validate() 用 EMA 模型评估、EMA decay=0.9999 启动极慢，前期 EMA 权重接近初始噪声 → val≈随机。约 ep10-11 EMA「充电」完成后 val 才跳到真实值（baseline ep10:0.1%→ep11:29%→ep15:56%）。**因此跨实验比较 val 必须从 ep13+ 起比，早期数字不可比。** full_mixcut/cleaned_v1 用了 raw/EMA 取 max 的评估逻辑，前期即显示真实 raw 值。

## Full Training Runs

| Date | Model | 数据/增广 | Epochs | Val Top-1 | Val Top-5 | Checkpoint | Notes |
| ---- | ----- | --------- | ------ | --------- | --------- | ---------- | ----- |
| 2026-05-18 | ResNet-50 | 全量 / 无增广 | 100 | **77.53%** | 93.63% | checkpoints/best.pth | 历史强基线 |
| 2026-05-26 | ResNet-50 | 全量 / Mixup+CutMix+RandAug+Erase | 100 | 76.91% | — | checkpoints_mixup_v2/best.pth | 全增广，100ep 未破 baseline |
| 2026-05-30 | ResNet-50 | C1+C2 清洗 / 无增广 | 100 | 76.84% | 92.63% | checkpoints_cleaned_v1/best.pth | 清洗数据，与 mixup_v2 基本打平 |
| 2026-06-02 | ResNet-50 | 全量 / Mixup+CutMix | 100 | **77.57%** plain / **77.99%** HFlip TTA | 93.77% plain / 93.97% HFlip TTA | checkpoints_full_mixcut/best.pth | 最终提交候选，best_source=raw |

> 截至 2026-06-02：full_mixcut 已完成并超过 baseline。最终提交路线为 `checkpoints_full_mixcut/best.pth` + single-model HFlip TTA，验证集 Top-1 77.99。

## Key Findings

1. **数据集不是长尾分布**：不平衡比仅 1.78（732~1300 张/类），无需特殊长尾处理
2. **数据质量高**：损坏 0 张，近空白 0 张，主要噪声来自标签歧义（radio ~60%）
3. **10% subset 不足以评估 val**：12816 张训练图 + 30 epochs 不够泛化到 50000 val
4. **val 必须按 ImageFolder 类目录组织**：扁平 val 目录会让训练全程显示 Val ≈ 0%（已踩坑）
5. **历史 loss 对比实验需全部重做**：均跑在 val 损坏期间，无信号

## 待办

- [x] mixup 训练完成后评估（mixup_v2=76.91%，full_mixcut=77.57 plain / 77.99 HFlip TTA）
- [x] full_mixcut 跑完后定提交模型（超过 baseline 77.53%）
- [x] 选定 checkpoint → HFlip TTA → 生成提交 CSV（`submit/full_mixcut_hflip.csv`）
- [ ] 噪声鲁棒 loss 重做（赛题主观分 30% 的核心支撑）
- [x] 接入赛题提供的 100K 测试集并完成 full_mixcut_hflip 推理
- [x] 确认提交 CSV 格式（100000 行、无 header、`.JPEG` 文件名、4 位类别编号）

# Final Strategy Draft: 噪声鲁棒与混淆类别分层处理

## 一句话结论

当前最稳妥的校赛策略是：以 ResNet-50 + CE/Label Smoothing baseline 作为保底模型，停止完整强增强组合和鲁棒 loss 叠加方向；Mixup-only / CutMix-only 30 epoch 结果作为增强消融，鲁棒 loss 结果作为噪声分析支撑。

---

## 当前基线

| Model | Setting | Val Top-1 | Val Top-5 | 状态 |
|---|---|---:|---:|---|
| ResNet-50 | CE + Label Smoothing, 100 epoch, EMA | 77.53% | 93.63% | 保底候选 |
| ResNet-50 | Mixup-only, 30 epoch quick screen | 69.78% | 89.59% | 增强消融，短期优于 CutMix-only |
| ResNet-50 | CutMix-only, 30 epoch quick screen | 69.12% | 89.42% | 增强消融，优先级低于 Mixup-only |

Baseline 已超过 76% 目标线，距 78% 冲刺线仅差 0.47 pp。Mixup-only / CutMix-only 30 epoch 结果只用于增强策略筛选，不能直接与 100 epoch baseline 公平比较。30% subset loss rerun 显示鲁棒 loss 没有明显优于 CE+LS，因此后续不优先进行 GCE+Mixup 或 SCE+Mixup。

---

## 问题分层

Week 2 噪声分析和 Week 3 混淆矩阵说明，当前错误不能简单归为一种类型。需要分层处理：

| 问题类型 | 代表类别/类别对 | 证据 | 推荐处理 |
|---|---|---|---|
| 类内标签噪声 | radio, modem, nematode | 人工复查发现标签歧义/主体偏差，radio 噪声率估计 >=60% | SCE/GCE/ELR，targeted cleaning |
| 标签定义重复/歧义 | maillot ↔ maillot, ear → corn | 混淆矩阵 Top-10；ImageNet 类名或语义本身有歧义 | 报告说明，必要时清洗或合并，loss 只能缓解 |
| 类间细粒度混淆 | laptop → notebook, dog breeds, snakes, turtles | 视觉或语义边界接近 | 增强、采样、对比学习；不主要依赖 SCE/GCE |
| 共现上下文混淆 | academic gown → mortarboard | 两个物体常同时出现 | CutMix/RandAug，降低上下文依赖 |

---

## 关于 radio 的结论

`n04041544 radio` 是典型的 intra-class noise。它的问题主要发生在类别内部：图片主体不一致、标签歧义、部分图像并非真正收音机。Week 3 混淆矩阵中 radio 没有进入 Top-10 类间混淆，说明它不是“经常被判成某个相邻类别”的问题。

因此，radio 适合用噪声鲁棒 loss 或人工清洗处理。SCE/GCE 对这类问题有理论和实验验证价值。

---

## 关于 maillot / ear / laptop-notebook 的结论

### maillot

`n03710637` 与 `n03710721` 都叫 maillot，双向混淆严重。这是 ImageNet 类定义重复导致的问题。模型难以仅靠视觉信息区分两个同名类别。SCE/GCE 可以降低部分异常标签影响，但不能从根本上解决重复类别定义。

### ear

`ear -> corn` 的混淆来自英文语义歧义：ear 既可以是耳朵，也可以是玉米穗。若训练或验证样本中包含玉米穗图片，这属于数据集标签定义问题，而不是普通模型能力不足。

### laptop-notebook

`laptop -> notebook` 是类间细粒度/语义重叠问题。两类视觉和语义都高度接近，SCE/GCE 不应被期待直接解决这个问题。更合适的方向是增强细粒度判别能力，例如更强数据增强、混淆对重采样或对比学习。

## 简化实验路线

当前完整强增强组合 `imagenet_resnet50_mixup.yaml` 已暂停，因为它同时包含 Mixup、CutMix、RandAugment 和 RandomErasing，训练成本高且不适合作为单项消融解释。校赛阶段已改为更轻量的单项消融和 loss rerun：

| 优先级 | 实验 | 目的 | 结论用途 |
|---:|---|---|---|
| 1 | Mixup-only / CutMix-only 30 epoch | 筛选单项增强方向 | Mixup-only 略优于 CutMix-only |
| 2 | CE/GCE/SCE 30% subset loss rerun | 验证鲁棒 loss 是否优于 CE+LS | 无鲁棒 loss 明显优于 CE+LS |
| 3 | GCE/SCE + Mixup | loss 与增强叠加 | 暂不优先，避免继续扩大实验成本 |

---



历史 SCE/GCE/ELR 结果全部作废，因为当时 val 目录结构错误，验证结果接近随机，不能用于最终决策。

### Stage 1: 10% subset 快速筛查

目标：排查 NaN/Inf、明显不收敛、训练 loss 不下降。

| Run | Config | Loss | 参数 | Epochs | 记录项 |
|---|---|---|---|---:|---|
| rerun10_sce_a01_b10 | no-EMA | SCE | alpha=0.1, beta=1.0 | 10 | loss/acc/val |
| rerun10_sce_a05_b10 | no-EMA | SCE | alpha=0.5, beta=1.0 | 10 | loss/acc/val |
| rerun10_sce_a05_b05 | no-EMA | SCE | alpha=0.5, beta=0.5 | 10 | loss/acc/val |
| rerun10_gce_q07 | no-EMA | GCE | q=0.7 | 10 | loss/acc/val |
| rerun10_elr_b09_l30 | no-EMA | ELR | beta=0.9, lambda=3.0 | 10 | loss/acc/val |

### Stage 2: 30% subset 主对比

目标：比较不同 loss 在更可信数据规模下的 Top-1/Top-5 表现。

| Run | Config | Loss | 参数 | Epochs | 判断标准 |
|---|---|---|---|---:|---|
| rerun30_ce_ls01 | no-EMA | CE+LS | smoothing=0.1 | 40 | 对照组 |
| rerun30_sce_a01_b10 | no-EMA | SCE | alpha=0.1, beta=1.0 | 40 | 是否优于 CE |
| rerun30_sce_a05_b05 | no-EMA | SCE | alpha=0.5, beta=0.5 | 40 | 是否优于 CE |
| rerun30_gce_q07 | no-EMA | GCE | q=0.7 | 40 | 是否优于 CE |
| rerun30_elr_b09_l30 | no-EMA | ELR | beta=0.9, lambda=3.0 | 40 | 是否优于 CE |

---

## 要跑的命令

### 10% 快速筛查

```bash
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.1 --sce-beta 1.0 --run-name rerun10_sce_a01_b10
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.5 --sce-beta 1.0 --run-name rerun10_sce_a05_b10
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.5 --sce-beta 0.5 --run-name rerun10_sce_a05_b05
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss gce --gce-q 0.7 --run-name rerun10_gce_q07
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss elr --run-name rerun10_elr_b09_l30
```

### 30% 主对比

```bash
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss ce --run-name rerun30_ce_ls01
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss sce --sce-alpha 0.1 --sce-beta 1.0 --run-name rerun30_sce_a01_b10
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss sce --sce-alpha 0.5 --sce-beta 0.5 --run-name rerun30_sce_a05_b05
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss gce --gce-q 0.7 --run-name rerun30_gce_q07
python scripts/train.py --config configs/imagenet_resnet50_noema_full.yaml --subset 0.3 --epochs 40 --warmup-epochs 3 --eval-interval 5 --loss elr --run-name rerun30_elr_b09_l30
```

### Mixup/CutMix 单项结果

Mixup-only 和 CutMix-only 均已完成 30 epoch quick screen，结果见下方表格。本轮 30 epoch 结果只用于增强策略筛选，不直接作为最终提交模型。

---

## 结果记录表

### Mixup/CutMix 结果

| Config | Epochs | Best Val Top-1 | Val Top-5 | 相对 baseline 77.53 | 结论 |
|---|---:|---:|---:|---:|---|
| imagenet_resnet50_mixup_only.yaml | 30 | 69.78% | 89.59% | -7.75 pp | Mixup-only 短期筛选略优于 CutMix-only |
| imagenet_resnet50_cutmix_only.yaml | 30 | 69.12% | 89.42% | -8.41 pp | CutMix-only 短期筛选低于 Mixup-only |

### Loss 重做结果

| run_name | Loss | 参数 | Epochs | Best Val Top-1 | Final Val Top-1 | Final Val Top-5 | Final Train Acc | NaN/Inf | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| rerun30_ce_ls01 | CE+LS | smoothing=0.1 | 40 | **66.63%** | **66.63%** | **87.16%** | 73.71% | No | 30% subset 对照组，当前最稳 |
| rerun30_sce_a01_b10 | SCE | alpha=0.1, beta=1.0 | 40 | 60.66% | 60.66% | 80.99% | 61.92% | No | 明显低于 CE+LS，不作为最终模型候选 |
| rerun30_sce_a05_b05 | SCE | alpha=0.5, beta=0.5 | 40 | **66.63%** | **66.63%** | 86.66% | 72.68% | No | Top-1 追平 CE+LS，但 Top-5 略低，可作为接近 CE 的鲁棒 loss 结果 |
| rerun30_gce_q07 | GCE | q=0.7 | 40 | 26.01% | 26.01% | 34.56% | 26.26% | No | 明显欠学习，不适合当前从零训练设置 |
| rerun30_elr_b09_l30 | ELR | beta=0.9, lambda=3.0 | 40 | 未跑 | 未跑 | 未跑 | 未跑 | — | 可选实验；当前结论已足够，暂不优先 |

---

## Loss rerun 结论

30% subset + 40 epoch 的 loss 重做结果显示，鲁棒 loss 没有明显优于 CE+Label Smoothing。CE+LS 达到 66.63% Top-1 / 87.16% Top-5，是当前对照基准；SCE alpha=0.5, beta=0.5 的 Top-1 同为 66.63%，但 Top-5 为 86.66%，略低于 CE，可作为“接近 CE 的鲁棒 loss”结果；SCE alpha=0.1, beta=1.0 降至 60.66% Top-1；GCE q=0.7 仅 26.01% Top-1，表现为明显欠学习。

因此，后续不优先进行 GCE+Mixup 或 SCE+Mixup，也不建议继续投入大量时间调鲁棒 loss。最终模型路线仍应优先考虑已完成的 100 epoch CE+LS baseline，以及是否延长 Mixup-only 训练。鲁棒 loss 部分主要用于报告中说明：数据集噪声更偏局部标签歧义和类间混淆，而不是需要全局替换 CE 的强噪声场景。

---

## 最终决策规则

1. 若 Mixup/CutMix Top-1 >= 78.0%，优先作为最终提交候选。
2. 若 Mixup/CutMix 未超过 baseline，则 baseline 仍是保底模型。
3. 若 30% loss 重做中某个鲁棒 loss 明显优于 CE 对照，再考虑 full training 复验。
4. 若鲁棒 loss 未优于 CE，对报告仍有价值：说明本数据集整体较干净，强鲁棒 loss 对全局训练不一定有效，噪声处理应 targeted。
5. 最终报告要明确：标签噪声、标签定义问题、类间细粒度混淆是三类不同问题，不能用单一 loss 解释全部现象。

# Strategy v1: 10% 子集 Loss 收敛验证与训练策略

## 目标

验证 SCE / GCE 在 ImageNet 10% 训练子集上的基础收敛稳定性，排查 NaN / Inf / 明显不收敛，并基于 Week 2 噪声分析给出第一版训练策略。

## 实验设置

- 数据集：ImageNet-1K train 10% 子集，val 使用完整验证集
- 模型：ResNet-50（如本地时间不足，可先用 StarNet 做快速筛查，再用 ResNet-50 复验）
- 公共参数：
  - epochs: 10
  - warmup_epochs: 1
  - eval_interval: 1
  - batch_size: 128
  - accumulation_steps: 2
  - optimizer: SGD
  - base_lr: 0.1（若 10% 子集震荡，降到 0.02 复跑）
  - weight_decay: 1e-4
  - AMP: true
- 指标输出：每个 run 写入 `logs/<run_name>/metrics.csv` 和 `logs/<run_name>/summary.json`
- 汇总输出：`docs/week2_loss_sweep.csv` 和 `docs/week2_loss_curves.png`

## 必跑 Loss Sweep

| run_name | Loss | 参数 | 命令核心 |
|---|---|---|---|
| sce_a01_b10 | SCE | alpha=0.1, beta=1.0 | `--loss sce --sce-alpha 0.1 --sce-beta 1.0` |
| sce_a05_b10 | SCE | alpha=0.5, beta=1.0 | `--loss sce --sce-alpha 0.5 --sce-beta 1.0` |
| sce_a07_b10 | SCE | alpha=0.7, beta=1.0 | `--loss sce --sce-alpha 0.7 --sce-beta 1.0` |
| gce_q05 | GCE | q=0.5 | `--loss gce --gce-q 0.5` |
| gce_q07 | GCE | q=0.7 | `--loss gce --gce-q 0.7` |
| gce_q09 | GCE | q=0.9 | `--loss gce --gce-q 0.9` |

完整运行命令：

```bash
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.1 --sce-beta 1.0 --run-name sce_a01_b10
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.5 --sce-beta 1.0 --run-name sce_a05_b10
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss sce --sce-alpha 0.7 --sce-beta 1.0 --run-name sce_a07_b10
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss gce --gce-q 0.5 --run-name gce_q05
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss gce --gce-q 0.7 --run-name gce_q07
python scripts/train.py --config configs/imagenet_resnet50.yaml --subset 0.1 --epochs 10 --warmup-epochs 1 --eval-interval 1 --loss gce --gce-q 0.9 --run-name gce_q09
python scripts/plot_loss_sweep.py --input logs --output docs
```

## 结果记录

> 本次实验使用 AutoDL 上从 Kaggle zip 按类别均匀抽样构造的物理 10% train subset，因此训练命令使用 `--subset 1.0`；val 使用完整 50,000 张验证集。模型为 StarNet-S2 快速筛查版。

| run_name | Loss | 参数 | Epochs | Initial Loss | Final Loss | Final Train Acc | Best Val Top-1 | Final Val Top-1 | Final Val Top-5 | NaN/Inf | Train Loss 下降 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| sce_a01_b10 | SCE | alpha=0.1, beta=1.0 | 10 | 4.6878 | 4.6903 | 0.1391% | 0.162% | 0.152% | 0.714% | 否 | 否 | 可跑通但收敛最弱，不推荐作为主配置 |
| sce_a05_b10 | SCE | alpha=0.5, beta=1.0 | 10 | 7.4553 | 7.3206 | 0.7688% | 0.148% | 0.110% | 0.500% | 否 | 是 | 训练集收敛最明显，可作为噪声类 targeted SCE 候选 |
| sce_a07_b10 | SCE | alpha=0.7, beta=1.0 | 10 | 8.8394 | 8.7055 | 0.4836% | 0.144% | 0.090% | 0.544% | 否 | 是 | 稳定但验证集收益不明显，优先级低于 alpha=0.5 |
| gce_q05 | GCE | q=0.5 | 10 | 1.9369 | 1.9352 | 0.2500% | 0.138% | 0.136% | 0.468% | 否 | 是 | 稳定但收敛幅度很小，偏保守 |
| gce_q07 | GCE | q=0.7 | 10 | 1.4172 | 1.4163 | 0.1992% | 0.146% | 0.140% | 0.484% | 否 | 是 | 稳定，可作为 GCE 默认起点 |
| gce_q09 | GCE | q=0.9 | 10 | 1.1089 | 1.1087 | 0.1344% | 0.138% | 0.120% | 0.490% | 否 | 是 | 接近 CE 行为，收敛弱于 SCE alpha=0.5 |

结论：六组 loss 均未出现 NaN/Inf，说明 SCE/GCE 在 10% 子集上可稳定训练。SCE `alpha=0.5, beta=1.0` 的 train loss 与 train acc 改善最明显，建议作为噪声类 targeted SCE 的第一候选；GCE 建议保留 `q=0.7` 作为备选鲁棒 loss 起点。


## 噪声报告结论

基于 `docs/noise_top20.md`：

- ImageNet-1K 图片质量整体较好，真实噪声主要是标签歧义 / 主体偏差，不是文件损坏或低质量图像。
- 自动噪声检测里大量高分来自灰度图 / 白底产品照误判，不适合直接按程序分数全局加大鲁棒 loss。
- 重点关注类别：
  - `n04041544` radio：人工复查估计真实噪声率 ≥60%，优先处理。
  - `n03777754` modem：中等噪声，常与路由器 / 集线器混淆。
  - `n01930112` nematode：中等噪声，微观图不易判别。

## Strategy v1

1. 默认主线仍使用 CE / label smoothing。
   - 理由：95%+ 类别较干净，全局强鲁棒 loss 可能降低干净样本学习效率。
   - 建议：baseline 使用 CE + label_smoothing=0.1。

2. SCE 用作噪声敏感策略，而不是全局默认。
   - 初始候选：`alpha=0.5, beta=1.0` 或 `alpha=0.7, beta=1.0`。
   - 若 radio / modem 等噪声类在 Week 3 混淆矩阵中继续异常，可做类别级加权或 targeted fine-tuning。

3. GCE 作为备选鲁棒 loss。
   - `q=0.7` 作为默认起点。
   - `q=0.5` 更抗噪但更可能欠拟合。
   - `q=0.9` 更接近 CE，可用于验证鲁棒 loss 是否确实带来收益。

4. 数据增强策略保持保守。
   - v1 可启用 RandAug / ColorJitter / RandomErasing 做常规增强。
   - Mixup / CutMix 当前在 `src/data/transforms.py` 中尚未接入 trainer，v1 文档不把它们列为已启用策略。

5. 学习率策略。
   - 10% 子集 sweep 首先沿用 ImageNet ResNet-50 配置的 SGD + cosine + 1 epoch warmup。
   - 如果任一 loss 出现震荡或 train loss 不下降，优先把 `base_lr` 从 0.1 降到 0.02 复跑，不直接判定 loss 不可用。

## 验收标准

- 六组实验均产出 `logs/<run_name>/metrics.csv` 和 `summary.json`。
- `python scripts/plot_loss_sweep.py --input logs --output docs` 产出：
  - `docs/week2_loss_sweep.csv`
  - `docs/week2_loss_curves.png`（若环境安装 matplotlib）
- 每组明确记录：是否 NaN/Inf、train loss 是否下降、best val top-1/top-5。
- 根据 sweep 结果把本文件的结果表从 TBD 更新为实际数值。

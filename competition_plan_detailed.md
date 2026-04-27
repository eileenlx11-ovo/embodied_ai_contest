# 上海理工大学具身智能赛事 — 备赛全指南（精细化版）

> 赛题：网络监督的细粒度图像识别（Webly-Supervised Fine-Grained Image Recognition）
> 数据集：ImageNet-1K（1000 类，1,281,167 训练图，50,000 验证图，100,000 测试图）
> 硬件：RTX 4060（8GB VRAM）x 1
> 关键约束：不可使用预训练模型、不可使用额外数据、不可使用多模型集成
> 模型限制：推理时间和模型大小不限

---

## 目录

1. [RTX 4060 适配策略](#1-rtx-4060-适配策略)
2. [评分构成与目标设定](#2-评分构成与目标设定)
3. [项目结构](#3-项目结构)
4. [方向一：数据清洗与分析（详细任务拆解）](#4-方向一数据清洗与分析的详细任务拆解)
5. [方向二：模型设计与实现（详细任务拆解）](#5-方向二模型设计与实现的详细任务拆解)
6. [方向三：噪声鲁棒训练策略（详细任务拆解）](#6-方向三噪声鲁棒训练策略的详细任务拆解)
7. [方向四：实验管理、提交与答辩（详细任务拆解）](#7-方向四实验管理提交与答辩的详细任务拆解)
8. [完整协作时间表（4 人 x 6 周，适配 RTX 4060）](#8-完整协作时间表)
9. [RTX 4060 训练时间速查表](#9-rtx-4060-训练时间速查表)
10. [检查清单](#10-检查清单)

---

## 1. RTX 4060 适配策略

### 1.1 硬件约束

| 项目 | 数值 | 影响 |
|------|------|------|
| VRAM | 8GB GDDR6 | 单卡最大 batch size 受限 |
| FP16 TFLOPS | ~15 TFLOPS | 约为 A100(312 TFLOPS) 的 **1/20** |
| 显存带宽 | 272 GB/s | 数据加载瓶颈更明显 |

### 1.2 核心应对策略

| 策略 | 具体做法 | 预期收益 |
|------|---------|---------|
| **混合精度 AMP** | `torch.cuda.amp`，FP16 训练 | 显存减半，速度 2x |
| **梯度累积** | `accumulation_steps = 4~8` | 模拟大 batch，每步等效 batch = 物理 batch × 累积步数 |
| **小 batch + 大累积** | 物理 batch=64，累积4步 → 等效256 | 单卡训练可行 |
| **Channels Last** | `model.to(memory_format=torch.channels_last)` | 显存节省 ~10%，速度提升 |
| **torch.compile** | `model = torch.compile(model)` | 训练速度提升 20-30%（PyTorch 2.x） |
| **梯度 checkpointing** | 前向不保存中间激活，反向重算 | 显存节省 30-50%，速度降低 20% |
| **DataLoader 优化** | `num_workers=4~8`, `pin_memory=True`, `prefetch_factor=2` | CPU 加载不拖后腿 |
| **TF32 精度** | `torch.backends.cuda.matmul.allow_tf32 = True` | 在不损失精度下加速矩阵乘 |
| **减少验证频率** | 每 1 epoch 验证 → 每 2-4 epoch 验证 | 节省验证时间 |

### 1.3 各 backbone 在 RTX 4060 上的实际资源占用

| 模型 | 参数量 | 物理 batch | 梯度累积 | 等效 batch | 显存占用 | 100 epoch 预估时间 |
|------|--------|-----------|---------|-----------|---------|-------------------|
| **StarNet** | 12M | 256 | 2 | 512 | ~5.5GB | **~3 天** |
| **ResNet-50** | 25M | 128 | 2 | 256 | ~6.5GB | **~5-6 天** |
| **ResNet-RS-50** | 30M | 96 | 3 | 288 | ~7.2GB | **~7 天** |
| **ConvNeXt-V2-T** | 28M | 64 | 4 | 256 | ~7.0GB | **~8 天** |
| **ConvNeXt-V2-S** | 50M | 32 | 8 | 256 | ~7.8GB | **~12 天** ❌ 太慢 |

> ⚠️ **建议主攻方向**：ResNet-50 / ResNet-RS-50（稳妥）→ 有余力再上 ConvNeXt-V2-T（性能好但训练慢）
> 快速实验用 StarNet（12M 参数，3 天跑完 100 epoch，适合调参迭代）
> **注意**：规则明确推理时间和模型大小不限，最终提交不用担心模型太大。
> **待评估**：GhostNet-V2 和 EfficientViT（赛事参考模型），第 2 周在子集上快速测试。

### 1.4 显存优化速查清单

```python
# 写在 train.py 开头的优化（全部要加）
import torch

# 1. TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. cudNN 自动 tuning
torch.backends.cudnn.benchmark = True

# 3. AMP 自动混合精度
scaler = torch.cuda.amp.GradScaler()

# 4. DataLoader 多进程（很重要！4060 CPU 加载容易成瓶颈）
DataLoader(..., num_workers=4, pin_memory=True, prefetch_factor=2)

# 5. Channels Last 格式
model = model.to(memory_format=torch.channels_last)

# 6. torch.compile（PyTorch 2.x）
model = torch.compile(model)

# 7. 梯度累积（需要时启用）
accumulation_steps = 4  # 当物理 batch 不够大时
```

---

## 2. 评分构成与目标设定

### 2.1 评分公式

```
总成绩 = 70% 客观评分 + 30% 主观评分

客观分 = 标准化后的 Accuracy 得分（最高分设定 100 分，其余按比例调整）
主观分 = 标准化后的答辩分数（含答辩表现 + 技术方案 + 代码文档）
```

> ⚠️ **主观分占 30%，权重不低**。答辩准备不能只放最后一周，需从第 5 周开始。
> "问题与思考"部分需要真实的踩坑记录，从第 1 周起就要记录问题日志。

### 2.2 ImageNet-1K 从零训练参考 Acc

| 模型 | 从零训练 Top-1（参考） | 备注 |
|------|----------------------|------|
| ResNet-50 | ~72-74% | 标准 baseline |
| ResNet-50 + 增强 + 调优 | ~76-78% | 数据增强 + 训练技巧 |
| ResNet-RS-50 | ~77-79% | 改进结构 + 训练策略 |
| ConvNeXt-V2-T | ~79-81% | 更先进的结构 |
| ConvNeXt-V2-S | ~81-83% | 更大模型（4060 跑不动完整版） |

### 2.3 预估竞争态势

| 水平 | Top-1 Acc | 客观分（假设最高 78%） | 总成绩预估 |
|------|-----------|----------------------|-----------|
| 基线 | 72% | 92.3 | 64.6 + 主观分 |
| 有竞争力 | 76% | 97.4 | 68.2 + 主观分 |
| 顶尖 | 78%+ | 100 | 70 + 主观分 |

> 🎯 **目标设定**：
> - 保底：Top-1 ≥ 74%，Top-5 ≥ 91%
> - 目标：Top-1 ≥ 76%，Top-5 ≥ 92.5%
> - 冲刺：Top-1 ≥ 78%，Top-5 ≥ 93.5%

---

## 3. 项目结构

```
d:\dev\embodied AI\
├── data/
│   ├── imagenet/               # ImageNet-1K 数据集（符号链接或直接存放）
│   │   ├── train/              # 1000 个子文件夹，1,281,167 张
│   │   ├── val/                # 1000 个子文件夹，50,000 张
│   │   └── test/               # 100,000 张测试图（用于最终预测提交）
│   └── metadata/
│       ├── class_counts.csv     # 每类图片数量统计
│       └── noise_samples/       # 噪声可视化样本
├── configs/                    # 实验配置（YAML）
│   ├── baseline.yaml
│   ├── experiment_001.yaml
│   └── sweeps/                 # WandB sweep 配置
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py           # ResNet / ResNet-RS
│   │   ├── convnext_v2.py      # ConvNeXt-V2
│   │   ├── starnet.py          # StarNet（轻量快速）
│   │   └── build_model.py      # 模型工厂函数
│   ├── data/
│   │   ├── __init__.py
│   │   ├── imagenet_dataset.py # ImageNet 数据集加载
│   │   ├── transforms.py       # 数据增强（RandAug, MixUp, CutMix, AugMix）
│   │   ├── sampler.py          # 采样策略（ClassBalancedSampler, WeightedSampler）
│   │   ├── sample_selection.py # 样本选择（loss-based, confidence-based）
│   │   └── noise_detection.py  # 噪声检测方法
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base_loss.py        # Loss 统一接口
│   │   ├── sce.py              # Symmetric Cross Entropy
│   │   ├── peer_loss.py        # Peer Loss
│   │   ├── gce.py              # Generalized Cross Entropy
│   │   ├── elr.py              # Early Learning Regularization
│   │   ├── label_smoothing.py  # Label Smoothing
│   │   ├── nce.py              # Normalized CE（用于自监督）
│   │   └── composite_loss.py   # 复合 Loss 组合器
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py     # 基础训练器
│   │   └── noisy_trainer.py    # 噪声鲁棒训练器（含样本选择）
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py          # 评估指标
│       ├── visualization.py    # 可视化（混淆矩阵, t-SNE, grid）
│       ├── seed.py             # 随机种子
│       └── ema.py              # EMA 模型权重
├── scripts/
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 验证入口
│   ├── predict.py              # 测试集预测 + CSV（10万张）
│   ├── validate_csv.py         # CSV 格式校验
│   ├── compute_accuracy.py     # 自行计算 Accuracy
│   └── run_experiment.sh       # 批量实验
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_noise_analysis.ipynb
│   └── 03_result_analysis.ipynb
├── submit/
│   ├── result.csv              # 预测结果（文件名,类别名 格式）
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run.sh                  # 一键复现（训练+验证+预测）
│   └── tech_report/
│       ├── main.pdf            # 技术方案（按官方大纲）
│       └── figures/
├── docs/
│   └── README.md               # ★ 代码、环境、操作说明文档（独立于技术报告）
├── problem_log.md              # ★ 问题日志（从第1周起记录，供技术报告"问题与思考"）
├── logs/                       # TensorBoard / WandB 日志
├── checkpoints/                # 模型权重
├── .gitignore
└── README.md
```

---

## 4. 方向一：数据清洗与分析的详细任务拆解

### 4.1 总览

| 人员 | 建议：1 人主力 + 1 人支援 |
|------|--------------------------|
| 依赖 | 无（可独立启动） |
| 核心产出 | 完整数据集、统计分析报告、样本选择工具、可视化图表 |
| 与其它方向接口 | 输出 `sample_selection.py` 给方向三；输出图表给方向四 |

### 4.2 任务包 A1：数据集下载与组织（2 天）

#### A1-1 下载 ImageNet-1K
```bash
# 推荐下载方式
# 1. 从 OpenDataLab / Academic Torrents / 官方源 下载
# 2. 目录结构必须为:
#    data/imagenet/
#    ├── train/
#    │   ├── n01440764/
#    │   │   ├── n01440764_10026.JPEG
#    │   │   └── ...
#    │   ├── n01443537/
#    │   └── ...
#    └── val/
#        ├── n01440764/
#        └── ...
```

#### A1-2 数据完整性校验
```python
# TODO: 实现 verify_dataset.py
# 功能列表:
# 1. 检查每个图片能否用 PIL 正常打开（损坏文件列表）
# 2. 检查 train 是否 1000 个文件夹，约 1.28M 张
# 3. 检查 val 是否 1000 个文件夹，50000 张
# 4. 检查每类至少 1 张图
# 5. 输出验证报告
```

#### A1-3 基础数据集封装
```python
# TODO: src/data/imagenet_dataset.py
# 实现功能:
# 1. torchvision.datasets.ImageFolder 包装
# 2. 返回 (image, label, image_path) 三元组（路径用于后续分析）
# 3. 支持 anno_file 模式（读取 txt/csv 标签文件）
# 4. 支持 subset（随机取 10%/20% 用于快速实验）
```

### 4.3 任务包 A2：统计分析（1.5 天）

#### A2-1 类别分布统计
```python
# TODO: notebooks/01_data_exploration.ipynb Cell 1-6
# 1. 遍历 train 目录，统计每类图片数量 → class_counts.csv
# 2. 绘制：类别数量分布直方图（log scale）
# 3. 绘制：类别数量排序图（降序）
# 4. 计算：
#    - 最多图片类别: ____ 张
#    - 最少图片类别: ____ 张  
#    - 平均: ____ 张/类
#    - 中位数: ____ 张/类
#    - 不平衡比（max/min）: ____
# 5. 标注头部 10 类和尾部 10 类
```

#### A2-2 标签映射与层级分析
```python
# TODO: notebooks/01_data_exploration.ipynb Cell 7-10
# 1. 加载 imagenet_class_index.json（id→synset→class_name 映射）
# 2. 分析 1000 个类别的语义层级（属于哪些上位类：动物、植物、物品...）
# 3. 找出 semantically similar 类别组（如不同品种的狗、不同车型）
# 4. 输出类别层级树（只到 2 层）
```

### 4.4 任务包 A3：噪声可视化与估算（2 天）

#### A3-1 随机抽样可视化
```python
# TODO: scripts/visualize_samples.py
# 输入: data/imagenet/train/
# 输出: figures/sample_grids/
# 逻辑:
# 1. 对每个类别，随机抽 9 张图
# 2. 拼接成 3x3 grid 图，标注类别名
# 3. 保存为 {class_id}_{class_name}.jpg
# 总计 1000 张 grid 图（每个类别一张）
# 建议：先只输出 100 张（抽样），人工标注噪声
```

#### A3-2 人工噪声标注
```python
# TODO: notebooks/02_noise_analysis.ipynb
# 流程:
# 1. 从 A3-1 的 100 张 grid 图中人工标注：
#    - ✓ = 标签正确
#    - ✗ = 标签错误
#    - ? = 不确定
# 2. 统计噪声率估算: 错误数 / 总标注数 = ~____%
# 3. 分类噪声类型:
#    - 对称噪声: 标签随机错到其他类
#    - 非对称噪声: 标签错到相似类（如哈士奇→狼）
#    - 开集噪声: 图片根本不属于任何已知类
# 4. 输出噪声估算报告
```

#### A3-3 Loss 排序噪声检测（依赖 B1 完成）
```python
# TODO: src/data/noise_detection.py
# class LossBasedNoiseDetector:
# 原理: 模型先记忆干净样本（小 loss），后记忆噪声样本（大 loss）
# 方法:
# 1. 在 warm-up 阶段（5-10 epoch）记录每个样本的累计 loss
# 2. warm-up 后按 loss 排序:
#    - 小 loss (bottom 20%) → 高置信度干净样本
#    - 大 loss (top 20%) → 疑似噪声样本
# 3. 可视化 loss 分布直方图（双峰？）
# 4. 输出: clean_indices, noisy_indices
```

### 4.5 任务包 A4：样本选择机制（1.5 天）

#### A4-1 实现样本选择器
```python
# TODO: src/data/sample_selection.py
# class SampleSelector:
#     基于 loss 或 置信度 动态选择样本
#
# 策略一: LossThresholdSelector
#   - warmup_epochs = 5
#   - 每个 epoch 计算每个样本的 loss
#   - 保留 loss < threshold 的样本（如百分位 80%）
#   - threshold 可固定或动态调整
#
# 策略二: ConfidenceSelector  
#   - 用模型预测的 softmax 最大概率作为置信度
#   - 保留置信度 > threshold 的样本
#   - 适合训练中后期
#
# 策略三: Co-teaching（两模型互选）
#   - 两个模型各自选择小 loss 样本
#   - 交换给对方训练
#   - 实现较复杂，先不做
#
# 接口:
#   def select(self, losses: Dict[str, float], epoch: int) -> List[str]:
#       返回选中的样本路径列表
```

#### A4-2 与 Dataset 集成
```python
# TODO: src/data/imagenet_dataset.py 增加 SelectiveDataset
# class SelectiveDataset(Dataset):
#   - 包装原始 ImageNet 数据集
#   - 维护一个 mask: bool[] 表示每个样本是否被选中
#   - 每次 __getitem__ 只返回被选中的样本
#   - update_mask(new_mask) 方法供外部调用
#   - 每个 epoch 开始前可动态更新 mask
```

### 4.6 任务包 A5：长尾分布分析与处理（1 天）

> ⚠️ 赛事规则反复强调"长尾分布"是核心挑战之一，原方案几乎未涉及。
> 第 2 周数据分析后，根据实际类别分布决定是否启用以下策略。

```python
# TODO: src/data/sampler.py — 长尾分布处理
#
# 步骤 1: 在 A2 统计分析中确认 ImageNet-1K 的类别不均衡程度
#   - 计算不平衡比（max_count / min_count）
#   - 如果不平衡比 > 2，启用以下策略
#
# 策略一: ClassBalancedSampler（推荐）
# class ClassBalancedSampler(Sampler):
#   - 对每个类别计算采样权重: weight_c = 1 / count_c
#   - 使用 WeightedRandomSampler 按权重采样
#   - 效果: 尾部类别被更频繁采样，缓解长尾问题
#
# 策略二: Re-weighting Loss
#   - 对每个类别的 loss 乘以权重: weight_c = (1/count_c)^β, β∈[0,1]
#   - β=0 等于不加权，β=1 完全按频率倒数加权
#   - 推荐 β=0.5（平方根倒数加权）
#
# 策略三: 两阶段训练
#   - 阶段一: 正常训练（学习通用特征）
#   - 阶段二: 用 ClassBalancedSampler 微调（平衡尾部类别）
#
# 与噪声鲁棒策略的配合:
#   - 长尾类别本身样本少，噪声影响更大
#   - 建议: 先做噪声检测清洗，再做长尾平衡
#   - 避免对噪声样本过度上采样
```

### 4.7 任务包 A6：混淆矩阵分析（依赖 B2，1 天）

```python
# TODO: notebooks/02_noise_analysis.ipynb
# 1. 加载训练好的 B2 baseline 模型
# 2. 在 validation 集上推理，收集:
#    - true_labels (50000,)
#    - pred_labels (50000,)
#    - pred_probs (50000, 1000)  ← 用于置信度分析
# 3. 生成混淆矩阵 (1000x1000 太大，只关注 top-10 易混淆类)
# 4. 对每对易混淆类:
#    - 统计 misclassify 数量
#    - 抽样可视化错误分类样本
#    - 分析是否为语义相似类别
# 5. 输出: top-10 易混淆类别对 + 示例图
```

### 4.8 任务包 A7：t-SNE 可视化（依赖 B2，1 天）

```python
# TODO: notebooks/02_noise_analysis.ipynb
# 1. 用 B2 模型提取验证集特征（倒数第二层，2048 维）
# 2. PCA 降维到 50 维（加速）
# 3. t-SNE 降维到 2 维（perplexity=30, n_iter=1000）
# 4. 散点图着色:
#    - 图1: 按真实标签着色（看聚类效果）
#    - 图2: 按预测是否正确着色（正确/错误分开）
#    - 图3: 按置信度高低着色（看低置信度分布）
# 5. 分析: 噪声样本是否聚在类间边界？是否形成独立簇？
```

### 4.9 👥 多人协作方案（针对 RTX 4060）

**2人分工：**
- **成员 A1（主力，数据分析）：** A1(2天) → A2(1.5天) → A3(2天，含人工标注) → A5长尾分析(1天) → 产出图表给方向四
- **成员 A2（支援，数据工程）：** A1-3 数据集封装(0.5天) → A4(1.5天，等 B1 的 loss) → A6/A7(2天，等 B2) → 集成样本选择+长尾采样到训练 Pipeline

**时间节点：**
- Day 1-3: 两人一起下载 + 组织数据（网络瓶颈，可以并行验证完整性）
- Day 4-5: A1 做统计分析，A2 做数据集封装
- Day 6-8: A1 做噪声标注，A2 等 B1 输出的 loss 做样本选择
- Day 9: A1 做长尾分布分析，决定是否启用 ClassBalancedSampler
- Day 10+: A1/A2 一起做混淆矩阵 + t-SNE

---

## 5. 方向二：模型设计与实现的详细任务拆解

### 5.1 总览

| 人员 | 建议：1-2 人 |
|------|-------------|
| 依赖 | 方向一提供 DataLoader |
| 核心产出 | 完整训练 Pipeline、Baseline、2-3 个 Backbone、训练技巧模块 |
| 核心约束 | RTX 4060 8GB 显存，模型选择必须考虑资源限制 |
| 模型限制 | 推理时间和模型大小不限（规则明确），最终提交可用较大模型 |

### 5.1.1 模型选型分析（赛事参考模型全覆盖）

> 赛事规则给出 6 个参考模型，技术报告中必须说明为什么选/不选每个模型。

| 参考模型 | 参数量 | 4060 可行性 | 选用决策 | 理由 |
|---------|--------|------------|---------|------|
| **ResNet-RS** | ~30M | ✅ 可行，batch=96 | ✅ 备选 | 改进训练策略的 ResNet，性能好但训练略慢 |
| **ConvNeXt-V2** | 28M(T)/50M(S) | ✅ Tiny 可行 | ✅ 冲刺模型 | 现代 CNN 架构，性能最优但训练最慢 |
| **GhostNet-V2** | ~6M | ✅ 非常轻量 | ⚠️ 需评估 | 轻量模型，训练极快，但 ImageNet 从零训练精度可能不够 |
| **EfficientViT (MIT)** | ~6-13M | ✅ 轻量 | ⚠️ 需评估 | 轻量 ViT，推理快；但 ViT 从零训练通常需要更多 epoch |
| **MobileViT-V2** | ~5-10M | ✅ 轻量 | ❌ 暂不选 | 移动端优化，从零训练 ImageNet 精度偏低 |
| **StarNet** | ~12M | ✅ 最快 | ✅ 快速实验 | 用于快速筛选超参和策略，不作为最终提交模型 |

**选型策略：**
- 主力模型：ResNet-50 / ResNet-RS-50（稳妥，训练效率高）
- 冲刺模型：ConvNeXt-V2-T（时间允许时上）
- 快速实验：StarNet（1.5 天跑完 50 epoch，用于策略筛选）
- 待评估：GhostNet-V2 和 EfficientViT — 第 2 周在 10% 子集上快速测试，如果精度接近 ResNet-50 且训练更快，可替换为主力模型
- 技术报告中需说明：为什么没选 MobileViT-V2（从零训练精度不够）、GhostNet-V2/EfficientViT 的评估结论

### 5.2 任务包 B1：训练 Pipeline 搭建（2 天）

#### B1-1 配置系统
```python
# TODO: configs/baseline.yaml
# 统一配置文件模板:
data:
  root: ./data/imagenet
  batch_size: 128          # 4060 上实际能跑的 batch
  num_workers: 4
  pin_memory: true
  input_size: 224

model:
  name: resnet50
  num_classes: 1000
  drop_path_rate: 0.0

training:
  epochs: 100
  optimizer: sgd           # sgd / adamw
  base_lr: 0.1
  weight_decay: 1e-4
  momentum: 0.9
  scheduler: cosine        # cosine / step / multistep
  warmup_epochs: 5
  label_smoothing: 0.1
  amp: true               # 4060 必须开 AMP
  accumulation_steps: 2   # 模拟更大 batch
  grad_clip: 5.0
  ema_decay: 0.9999

logging:
  use_wandb: true
  log_interval: 50
  eval_interval: 2        # 4060 每 2 epoch 验证一次省时间
  save_interval: 10
```

#### B1-2 训练循环核心
```python
# TODO: src/trainers/base_trainer.py
# class BaseTrainer:
#   关键实现细节（针对 4060）:
#   
#   1. __init__:
#      - 加载配置
#      - 初始化 model, optimizer, scheduler, scaler(AMP)
#      - 设置 torch.compile（4060 上提升明显）
#      - 设置 channels_last
#   
#   2. train_one_epoch:
#      for batch_idx, (images, labels) in enumerate(train_loader):
#          images = images.cuda(non_blocking=True)
#          labels = labels.cuda(non_blocking=True)
#          images = images.to(memory_format=torch.channels_last)
#          
#          with torch.cuda.amp.autocast():
#              outputs = model(images)
#              loss = criterion(outputs, labels)
#              loss = loss / accumulation_steps  # 梯度累积归一化
#          
#          scaler.scale(loss).backward()
#          
#          if (batch_idx + 1) % accumulation_steps == 0:
#              scaler.unscale_(optimizer)
#              torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#              scaler.step(optimizer)
#              scaler.update()
#              optimizer.zero_grad()
#              scheduler.step()  # 每个有效 step 更新一次
#   
#   3. validate:
#      - 不计算梯度
#      - 用 torch.cuda.amp.autocast() 加速
#      - 收集 top-1 / top-5
#   
#   4. save_checkpoint / load_checkpoint
```

#### B1-3 AMP + 梯度累积详细配置
```python
# 4060 上的关键配置组合:

# 情况 A: 小模型（StarNet, batch=256）
batch_size = 256
accumulation_steps = 2   # 等效 batch = 512
# 显存: ~5.5GB, 速度: 最快

# 情况 B: 中等模型（ResNet-50, batch=128）
batch_size = 128
accumulation_steps = 2   # 等效 batch = 256
# 显存: ~6.5GB, 速度: 较快

# 情况 C: 大模型（ConvNeXt-V2-T, batch=64）
batch_size = 64
accumulation_steps = 4   # 等效 batch = 256
# 显存: ~7.0GB, 速度: 慢

# 情况 D: 极限模型（ConvNeXt-V2-S, batch=32）
batch_size = 32
accumulation_steps = 8   # 等效 batch = 256
# 显存: ~7.8GB, 速度: 很慢 ❌ 不推荐
```

### 5.3 任务包 B2：ResNet-50 Baseline（3-5 天持续运行）

#### B2-1 实现 ResNet-50
```python
# TODO: src/models/resnet.py
# 从零实现 ResNet-50（或从 torchvision 取随机初始化版本）
#
# 关键结构:
# - stem: Conv7x7 → BN → ReLU → MaxPool
# - stage1: 3x Bottleneck(256) 
# - stage2: 4x Bottleneck(512)
# - stage3: 6x Bottleneck(1024)
# - stage4: 3x Bottleneck(2048)
# - head: GlobalAvgPool → FC(2048, 1000)
#
# 初始化: Kaiming Normal (Conv), Constant (BN)
# 注意: 不能加载预训练权重！
```

#### B2-2 Baseline 训练运行
```bash
# 在 RTX 4060 上运行 Baseline
# 预计 100 epoch = 5-6 天

# 先用 10% 数据快速验证流程（1 小时）
python scripts/train.py --config configs/baseline.yaml --subset 0.1 --epochs 10

# 再用 50 epoch 确认趋势（2-3 天）
python scripts/train.py --config configs/baseline.yaml --epochs 50

# 最后跑完整 100 epoch（5-6 天）
python scripts/train.py --config configs/baseline.yaml --epochs 100
```

#### B2-3 预期 Baseline 结果
```yaml
# ResNet-50 from scratch on 4060:
# Top-1: ~72-73% (100 epoch, cosine decay, batch=256等效)
# Top-5: ~90-91%
# 训练时间: ~5-6 天
```

### 5.4 任务包 B3：ConvNeXt-V2 实现（3 天，RTX 4060 适配版）

#### B3-1 ConvNeXt Block 实现
```python
# TODO: src/models/convnext_v2.py
#
# 针对 4060 的优化版本（ConvNeXt-V2-Tiny）:
# 
# ConvNeXt-V2-T 配置:
#   depths: [3, 3, 9, 3]
#   dims: [96, 192, 384, 768]
#  参数量: ~28M
#
# 核心模块:
# 1. LayerNorm (代替 BN)
# 2. 7x7 Depthwise Conv (空间特征)
# 3. GELU 激活
# 4. GRN (Global Response Normalization, V2 新加)
# 5. LayerScale (每个残差块输出的可学习缩放)
# 6. DropPath (Stochastic Depth，训练时随机丢弃层)
#
# 4060 注意:
# - drop_path_rate 设 0.1-0.2（正则化节省训练时间）
# - 需要 AMP 支持（FP16 训练）
# - 输入/输出都用 channels_last
```

#### B3-2 ConvNeXt-V2 全模型
```python
# 模型整体结构:
# stem: Conv4x4, stride=4 (代替 ResNet 的 7x7+pool)
# stage1: 3x ConvNeXtBlock(dim=96)
# stage2: 3x ConvNeXtBlock(dim=192), downsampling
# stage3: 9x ConvNeXtBlock(dim=384), downsampling
# stage4: 3x ConvNeXtBlock(dim=768), downsampling
# head: LayerNorm → GlobalAvgPool → Linear(768, 1000)
#
# Downsampling: LayerNorm → Conv2x2, stride=2 (代替 ResNet 的 stride=2 conv)
```

#### B3-3 随机初始化验证
```python
# 验证模型输出形状正确:
model = ConvNeXtV2T(num_classes=1000)
x = torch.randn(4, 3, 224, 224)
y = model(x)  # (4, 1000)
print(y.shape)  # 应输出 torch.Size([4, 1000])

# 验证随机初始化的输出分布:
# softmax 应接近均匀分布（每个类概率 ~1/1000）
probs = torch.softmax(y, dim=1)
print(probs.mean(dim=0).std())  # 应该很小
```

### 5.5 任务包 B4：训练技巧集成（2 天）

#### B4-1 学习率策略
```python
# TODO: src/trainers/base_trainer.py 中 scheduler 部分
# 
# Warmup + Cosine Decay（推荐方案）:
# - warmup_epochs = 5, 从 0 线性增加到 base_lr
# - cosine decay: 从 base_lr 衰减到 0
#
# 实现（PyTorch 自带）:
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[5])
```

#### B4-2 Label Smoothing
```python
# TODO: src/losses/label_smoothing.py
# 公式: loss = (1 - ε) * CE(one_hot, pred) + ε * CE(uniform, pred)
# 推荐: ε = 0.1
#
# 实现要点:
# - 直接用 torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # PyTorch 1.10+
# - 或手动实现（兼容旧版本）
```

#### B4-3 Stochastic Depth (DropPath)
```python
# TODO: src/models/convnext_v2.py 中实现 DropPath
# 
# 公式: training 时以概率 p 丢弃整个残差分支
#       output = x + drop_path(branch(x))  /   (1 - p) 在 inference 时 scale
#
# drop_path_rate: 从 0 线性增加到 0.1-0.2（按层递增）
# 只有深层的 block 才有效果
```

#### B4-4 EMA (Exponential Moving Average)
```python
# TODO: src/utils/ema.py
# class ModelEMA:
#   - 维护模型参数的指数移动平均
#   - 每个 step 更新: ema_param = decay * ema_param + (1 - decay) * model_param
#   - decay = 0.9999 (模型权重给 EMA 很大 inertia)
#   - 验证时用 EMA 参数，训练完保存 EMA 参数
#   - 实现要点: 不额外占用太多显存（4060 上需注意）
```

#### B4-5 混合精度 AMP 详细配置
```python
# TODO: 在 base_trainer.py 中 AMP 完整实现
#
# RTX 4060 支持 FP16 和 BF16
# 推荐: FP16（更稳定）
# 
# 注意点:
# 1. GradScaler 初始值: init_scale=2**16 (默认)
# 2. 每 2000 step 检查一次 overflow
# 3. 配合 gradient clipping: scaler.unscale_() 后再 clip
# 4. loss 在 backward 前已被 scaler scale，所以 loss 打印需要 loss*scaler.get_scale()
```

### 5.6 👥 多人协作方案（针对 RTX 4060）

**2人分工：**
- **成员 C1（框架 + Baseline 主力）：** B1(2天) → B2(运行5天，同步做B4) → 提供 loss 给方向一/三
- **成员 C2（Backbone 开发）：** 先读论文(1天) → B3 ConvNeXt-V2 实现(3天) → 集成测试(1天)

**1人精简方案（4060 限制下）：**
- B1(2天) → 跑 B2(后台持续，同时做 B4) → 不实现 ConvNeXt-V2，专注 ResNet-50 调优
- 时间线：第1周搭建+B2开跑，第2-4周逐步改进

**4060 特化建议：**
- StarNet（12M 参数）作为快速实验模型，3 天跑完 100 epoch
- ResNet-50 作为主模型，5-6 天跑完 100 epoch
- ConvNeXt-V2-T 作为冲刺模型，最后 2 周才跑
- **每个模型先在 10% 子集上验证正确性，再跑全量**

---

## 6. 方向三：噪声鲁棒训练策略的详细任务拆解

### 6.1 总览

| 人员 | 建议：1 人主力 |
|------|--------------|
| 依赖 | 方向一（DataLoader + 样本选择）+ 方向二（Model + Pipeline）|
| 核心产出 | 5-8 种 Loss、对比实验表格、最佳训练策略 |
| 4060 注意 | 对比实验必须控制变量，每次只改一个因子；先在 StarNet 上快速筛选 |

### 6.2 任务包 C1：Loss 函数统一接口（0.5 天）

```python
# TODO: src/losses/base_loss.py
# 统一接口设计:

from abc import ABC, abstractmethod
import torch.nn as nn

class BaseLoss(nn.Module, ABC):
    """所有 Loss 的基类"""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) 模型输出 logits
            targets: (N,) 整数标签 或 (N, C) 软标签
        Returns:
            loss: scalar
        """
        pass

class LossWrapper:
    """包装器：支持将 custom loss 接入标准训练流程"""
    def __init__(self, loss_fn, **kwargs):
        self.loss_fn = loss_fn(**kwargs)
    
    def compute(self, logits, targets, model=None, epoch=None, **extra):
        loss = self.loss_fn(logits, targets)
        return loss
```

### 6.3 任务包 C2：Label Smoothing（0.5 天）

```python
# TODO: src/losses/label_smoothing.py
# 原理: 将 one-hot 标签平滑为 [ε/(C-1), ..., 1-ε, ..., ε/(C-1)]
# 公式: loss = (1-ε) * CE(one_hot, p) + ε * CE(uniform, p)
# 推荐 ε = 0.1

class LabelSmoothingLoss(BaseLoss):
    def __init__(self, num_classes=1000, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            targets = F.one_hot(targets, self.num_classes).float()
            targets = targets * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss
```

### 6.4 任务包 C3：SCE（Symmetric Cross Entropy，1 天）

```python
# TODO: src/losses/sce.py
# 论文: https://arxiv.org/abs/1908.06112
#
# 核心思想: 
# - 标准 CE = -Σ y * log(p)  对噪声标签 overfit
# - RCE (Reverse CE) = -Σ p * log(y)  对噪声标签鲁棒但训练不稳定
# - SCE = α * CE + β * RCE  两者互补
#
# 推荐参数: α = 0.1, β = 1.0（论文推荐值）
# 从头调参: α ∈ [0.01, 1.0], β ∈ [0.1, 10.0]

class SCELoss(BaseLoss):
    def __init__(self, num_classes=1000, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        pred = F.softmax(logits, dim=-1)
        pred = torch.clamp(pred, min=1e-8)  # 数值稳定
        pred_log = torch.log(pred)
        
        one_hot = F.one_hot(targets, self.num_classes).float()
        
        # CE: -Σ y * log(p)
        ce = -(one_hot * pred_log).sum(dim=-1).mean()
        
        # RCE: -Σ p * log(y)  (注意 log(y) 中 0 处取 log 会 -inf)
        rce = -(pred * torch.log(one_hot + 1e-8)).sum(dim=-1).mean()
        # 实际实现中用: rce = -(pred * one_hot).sum(dim=-1).mean() 的变体
        # 参考官方实现: rce = -(pred * torch.log(one_hot.clamp(min=1e-8))).sum(-1).mean()
        
        loss = self.alpha * ce + self.beta * rce
        return loss
```

### 6.5 任务包 C4：Peer Loss（1 天）

```python
# TODO: src/losses/peer_loss.py
# 论文: https://arxiv.org/abs/1908.07229
#
# 核心思想: 
# - 对每个样本，从 batch 中随机取另一个样本的标签作为"peer label"
# - loss = CE(y_pred, y_orig) - CE(y_pred, y_peer)
# - 相当于减去 baseline，降低噪声影响
#
# 推荐 λ = 0.5 ~ 1.0

class PeerLoss(BaseLoss):
    def __init__(self, num_classes=1000, lam=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam
    
    def forward(self, logits, targets):
        # 主 loss
        main_loss = self.ce(logits, targets)
        
        # 随机 shuffle labels 作为 peer
        batch_size = targets.size(0)
        peer_indices = torch.randperm(batch_size, device=targets.device)
        peer_targets = targets[peer_indices]
        
        # peer loss
        peer_loss = self.ce(logits, peer_targets)
        
        return main_loss - self.lam * peer_loss
```

### 6.6 任务包 C5：GCE（Generalized Cross Entropy，1 天）

```python
# TODO: src/losses/gce.py
# 论文: https://arxiv.org/abs/1805.07836
#
# 核心思想: 
# - 将 CE 推广为 Box-Cox 变换: L_q(f(x), y) = (1 - f_y(x)^q) / q
# - q → 0 时退化为 CE, q=1 时变为 MAE
# - MAE 对噪声鲁棒但训练困难，GCE 折中
#
# 推荐: q = 0.7

class GCELoss(BaseLoss):
    def __init__(self, num_classes=1000, q=0.7):
        super().__init__()
        self.q = q
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        pred = F.softmax(logits, dim=-1)
        pred = torch.clamp(pred, min=1e-8)
        
        one_hot = F.one_hot(targets, self.num_classes).float()
        
        # 取目标类别的预测概率
        pred_y = (pred * one_hot).sum(dim=-1)  # (N,)
        
        # GCE = (1 - pred_y^q) / q
        loss = (1 - pred_y ** self.q) / self.q
        return loss.mean()
```

### 6.7 任务包 C6：ELR（Early Learning Regularization，1.5 天）

```python
# TODO: src/losses/elr.py
# 论文: https://arxiv.org/abs/1911.07471
#
# 核心思想:
# - 模型在 early learning 阶段学到的是 clean pattern
# - ELR 鼓励模型维持 early learning 阶段的预测（EMA of logits）
# - 对每个样本维护一个 EMA logits: t = β*t + (1-β)*logits
# - loss = CE + λ * log(1 - <p, t>)   # 让预测接近 EMA
#
# 推荐: β = 0.9, λ = 3.0

class ELRLoss(BaseLoss):
    def __init__(self, num_classes=1000, beta=0.9, lam=3.0):
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.num_classes = num_classes
        self.target_estimates = None  # 每个样本的 EMA
    
    def forward(self, logits, targets, indices=None):
        """
        Args:
            logits: (N, C)
            targets: (N,)
            indices: (N,) 样本索引，用于查找/更新 EMA
        """
        if indices is None:
            indices = targets  # fallback（不准确）
        
        pred = F.softmax(logits, dim=-1)
        
        # 初始化 EMA 估计
        if self.target_estimates is None:
            self.target_estimates = torch.zeros(len(self.target_estimates), self.num_classes, device=logits.device)
        
        # 更新 EMA: t = β*t + (1-β)*p
        self.target_estimates[indices] = (
            self.beta * self.target_estimates[indices] + 
            (1 - self.beta) * pred.detach()
        )
        
        # CE loss
        ce = F.cross_entropy(logits, targets)
        
        # ELR regularization: log(1 - <p, t>)
        t = self.target_estimates[indices].detach()
        elr_reg = torch.log(1 - (pred * t).sum(dim=-1) + 1e-8).mean()
        
        return ce + self.lam * elr_reg
```

### 6.8 任务包 C7：对比实验管理（持续进行）

#### C7-1 实验矩阵设计（4060 上优先做的对比）

```yaml
# 阶段一：快速筛选（在 StarNet 上跑 50 epoch，~1.5 天/组）
# 固定条件: StarNet, batch=256, epoch=50, cosine lr, no extra aug
# 变量: Loss 函数
# 
# 实验组:
#   E001: CE (baseline)
#   E002: LabelSmoothing(0.1)
#   E003: SCE(α=0.1, β=1.0)
#   E004: PeerLoss(λ=0.5)
#   E005: GCE(q=0.7)
#   E006: ELR(β=0.9, λ=3.0)

# 阶段二：组合优化（选 Top-3 Loss，在 ResNet-50 上跑 100 epoch）
# E007: SCE + ELR
# E008: LabelSmoothing + GCE  
# E009: SCE + LabelSmoothing + ELR（三重组合）

# 阶段三：超参搜索（对最优组合，在 StarNet 上调参）
# 搜索空间:
#   lr: [0.01, 0.05, 0.1, 0.2]
#   weight_decay: [1e-5, 5e-5, 1e-4, 5e-4]
#   loss_params: 根据具体 loss 定
```

#### C7-2 实验记录模板
```markdown
## 实验 E001

### 配置
- 模型: ResNet-50
- Loss: CrossEntropy (baseline)
- 优化器: SGD, lr=0.1, wd=1e-4, momentum=0.9
- Scheduler: Warmup 5ep + Cosine 95ep
- Batch: 128, Accum: 2 (等效256)
- Epochs: 100
- AMP: True
- 数据增强: Standard (RandomResizedCrop + RandomHorizontalFlip)
- 额外: 无

### 结果
- Top-1 Acc: 72.3%
- Top-5 Acc: 90.8%
- 训练时间: 5.5 天
- 最佳 epoch: 87

### 备注
- loss 曲线在 epoch 40 后缓慢下降，但 Acc 仍在上升
- 验证 loss 在 epoch 60 后略微上升（过拟合噪声标签）
```

### 6.9 👥 多人协作方案（RTX 4060 优化版）

**1 人主力 + 方向一/二支援：**
- **成员 E（训练策略，1 人）：** C1(0.5天) → C2+C3+C4(2天) → 等 StarNet 训练完成 → C5+C6(2.5天)
- **成员 C（方向二支援）：** 提供已训练好的模型做特征提取
- **成员 A（方向一支援）：** 提供样本选择模块，集成到 noisy_trainer

**4060 上的实验节奏（核心）：**
```
第2周: 用 StarNet 快速跑 CC 1-6 的对比（7组 × 1.5天 = 10.5天，并行不了，所以只选4组）
  优化策略: 1. 每 epoch 只验证 1 次（不是每 epoch 都验证）  
          2. epoch 数减为 50（趋势一致）
          3. 每次只改一个变量

实际执行:
  第2周初: CE baseline + LabelSmoothing + SCE 同时开始（但只有1张卡，串行）
  第2周末: PeerLoss + GCE
  第3周初: 选最好的2个组合跑 ResNet-50 完整 100 epoch
```

> ⚠️ **4060 限制下的核心策略**：只有 1 张卡，实验必须**串行**。因此**先在小模型上快速筛选，再在大模型上验证**是关键。

---

## 7. 方向四：实验管理、提交与答辩的详细任务拆解

### 7.1 总览

| 人员 | 建议：1 人全周期负责 |
|------|-------------------|
| 依赖 | 方向一/二/三的产出 |
| 核心产出 | 实验记录、可复现包、技术方案、答辩 PPT、问题日志 |
| 核心要求 | Docker 可复现 + CSV 格式正确 + 答辩有说服力 + 问题日志真实 |

### 7.1.1 任务包 D0：问题日志（第 1 周起，持续记录）

> ⚠️ 技术报告"问题与思考"章节需要真实的踩坑记录，不能最后编。从第 1 周起就要记录。

```markdown
# problem_log.md 模板

## 问题日志

### Week 1
| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 4/28 | ImageNet 下载速度慢 | 数据准备延迟 | 换用 OpenDataLab 镜像源 | 已解决 |
| 4/30 | 部分图片损坏无法打开 | 训练时报错 | verify_dataset.py 过滤 | 已解决 |

### Week 2
| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| ... | ... | ... | ... | ... |

## 关键决策记录
| 日期 | 决策 | 理由 | 结果 |
|------|------|------|------|
| 5/5 | 选择 ResNet-50 而非 GhostNet-V2 | 子集测试精度差 3% | 待验证 |

## 进一步优化思考
- 如果时间更多，可以尝试 DivideMix
- 自监督预训练 + 噪声标签微调可能效果更好
- ...
```

> 每周至少记录 2-3 条，最终整理进技术报告第四章"问题与思考"。

### 7.2 任务包 D1：实验管理体系（第 1 周，0.5 天）

#### D1-1 WandB 配置
```bash
# 1. 注册 wandb.ai 账号
# 2. 创建项目: embodied-ai
# 3. 初始化:
wandb login
wandb init
```

#### D1-2 实验命名规范
```yaml
# 命名格式: {模型}_{Loss}_{数据增强}_{Epoch}_{备注}
# 示例:
#   rn50_ce_base_100ep          # ResNet-50 + CE + 标准增强
#   rn50_sce_100ep              # ResNet-50 + SCE
#   rn50_sce+elr_100ep          # ResNet-50 + SCE + ELR
#   rn50_sce_randaug_100ep      # ResNet-50 + SCE + RandAug
#   convnext_sce_100ep          # ConvNeXt-V2-T + SCE
#   starnet_ce_50ep             # StarNet + CE + 50epoch（快速筛选）
```

#### D1-3 实验记录表
```
Google Sheet / Excel 表头:
| 实验ID | 日期 | 命名 | Backbone | Loss | 增强 | Epoch | Batch | LR | WD | Top-1 | Top-5 | 最佳epoch | 训练时间 | 备注 |
|--------|------|------|----------|------|------|-------|-------|-----|-----|-------|-------|-----------|---------|------|
```

### 7.3 任务包 D2：Docker 构建（第 2 周搭建基础框架，持续维护）

> ⚠️ **赛事方会复现代码**：规则明确"赛事方将对各参赛队伍提交的代码进行复现，若在 ImageNet-1k 上无法复现性能，将取消成绩"。
> Docker 必须从第 2 周就搭好基础框架，后续每次重大实验后都验证一次可复现性。

#### D2-1 Dockerfile
```dockerfile
# TODO: submit/Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY submit/run.sh .

# 默认入口
CMD ["bash", "run.sh"]
```

#### D2-2 requirements.txt
```txt
# TODO: submit/requirements.txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
pyyaml
wandb
tensorboard
```

#### D2-3 run.sh 复现脚本
```bash
#!/bin/bash
# TODO: submit/run.sh
# 一键复现脚本（包含训练、验证、预测全流程）

# 1. 设置环境
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 2. 数据准备（假设数据挂载到 /data/imagenet）
# ln -s /data/imagenet ./data/imagenet

# 3. 训练
python scripts/train.py --config configs/final_config.yaml

# 4. 在验证集上验证（确认指标可复现）
python scripts/evaluate.py --checkpoint checkpoints/best.pth --data_dir ./data/imagenet/val
# 预期输出: Top-1 Acc: XX.X%, Top-5 Acc: XX.X%

# 5. 在测试集上预测（10 万张）
python scripts/predict.py --checkpoint checkpoints/best.pth --test_dir ./data/imagenet/test --output submit/result.csv

# 6. 验证 CSV 格式
python scripts/validate_csv.py --csv submit/result.csv

# 7. 自行计算并输出 Accuracy
python scripts/compute_accuracy.py --csv submit/result.csv --gt ./data/imagenet/test_labels.csv
```

### 7.4 任务包 D3：预测与 CSV 生成（第 3 周，0.5 天）

```python
# TODO: scripts/predict.py
# 输入: checkpoint 路径, 测试集目录（10 万张独立测试集，非验证集）
# 输出: submit/result.csv

# ⚠️ CSV 格式（严格按赛事规则，写错直接判无效）:
# 第一列: 完整图片文件名（与测试集文件名完全一致，含扩展名）
# 第二列: 类别名（四位数字，不满四位前面补 0）
# 示例:
# xxxxxxxxxxxx.jpg, 0000
# xxxxxxxxxxxy.jpg, 1111
# xxxxxxxxxxxz.jpg, 0812
#
# 注意事项:
# 1. 文件名必须与测试集完全一致（包括大小写和扩展名）
# 2. 类别名必须是四位数字，不足四位前面补 0
# 3. 推理对象是 10 万张测试集，不是 5 万张验证集
# 4. 需自行计算 Accuracy 并确保正确

# 实现:
# 1. 加载 checkpoint
# 2. 创建 DataLoader（测试集 10 万张，batch_size=256）
# 3. 推理 + softmax
# 4. 取 argmax 作为预测类别
# 5. 将类别 ID 格式化为 4 位数字（str(cls).zfill(4)）
# 6. 输出格式: f"{image_filename}, {class_name_4digit}"
# 7. 保存为 CSV + 格式校验
# 8. 自行计算并记录 Accuracy

# 推理时间预估（RTX 4060，10 万张测试集）:
# ResNet-50 batch=256: ~24 min
# ConvNeXt-V2-T batch=128: ~36 min
```

### 7.5 任务包 D5：技术方案文档（第 4 周开始写，第 5 周完成）

> ⚠️ **必须按赛事规则附录的官方大纲撰写**，不能用 IEEE 论文风格。

#### D5-1 文档结构（严格按官方大纲）
```
技术方案（PDF）

一、算法概述（100-300 字）
   - 算法作品目标
   - 算法主要特点
   - 创新性
   - 应用成效和应用价值
   语言简洁明了，重点突出。

二、实现方案（核心部分，详细展开）
   2.1 解题思路
       - 问题分析：网络数据的标签噪声、数据偏差、长尾分布
       - 整体方案框架图
   2.2 数据处理
       - 数据集分析（类别分布、噪声率估算）
       - 数据清洗与样本选择策略
       - 数据增强方案（RandAug, MixUp, CutMix 等）
       - 长尾分布处理（ClassBalancedSampler / re-weighting）
   2.3 算法设计与开发
       - Backbone 选择及理由（为什么选/不选每个参考模型）
       - 噪声鲁棒损失函数设计
       - 训练策略（学习率、EMA、梯度累积等）
   2.4 模型训练与优化
       - RTX 4060 适配策略
       - 超参搜索过程
       - 训练技巧（AMP、torch.compile 等）
   2.5 测试与验证
       - 验证集评估流程
       - 测试集推理流程（10 万张）
       - CSV 生成与格式校验

三、算法创新点
   - 独特模型架构（如有改进）
   - 创新的数据处理方法
   - 算法优化策略
   - 与现有方法的对比优势

四、问题与思考 ★（从第 1 周起记录，不能最后编）
   - 开发过程中遇到的困难
   - 对应的解决方案
   - 进一步优化的思考
   示例：
   - 问题1: 4060 显存不足导致 batch size 受限 → 解决: 梯度累积 + AMP
   - 问题2: 噪声标签导致验证 loss 后期上升 → 解决: ELR 正则化
   - 问题3: 训练时间过长无法充分实验 → 解决: StarNet 快速筛选策略

五、过程进度（表格形式）
   | 阶段 | 开始时间 | 结束时间 | 主要完成内容 |
   |------|---------|---------|-------------|
   | 数据准备 | 4/28 | 5/4 | 数据下载、完整性校验、统计分析 |
   | 算法设计与开发 | 5/5 | 5/18 | Pipeline搭建、Loss实现、模型实现 |
   | 模型训练与优化 | 5/12 | 6/1 | Baseline训练、对比实验、超参搜索 |
   | 测试与验证 | 5/26 | 6/1 | 验证集评估、测试集推理、CSV生成 |
   | 文档撰写与提交 | 5/19 | 6/8 | 技术方案、答辩PPT、Docker打包 |

六、团队分工
   | 姓名 | 角色 | 具体负责内容 |
   |------|------|-------------|
   | 成员A | 数据分析师 | 数据探索、噪声分析、样本选择、可视化 |
   | 成员C | 算法工程师 | 模型实现、训练Pipeline、Backbone开发 |
   | 成员E | 算法工程师 | Loss函数设计、噪声鲁棒训练策略、对比实验 |
   | 成员F | 项目管理/文档 | 实验管理、Docker、技术方案、答辩PPT |

七、解题参考
   - 赛事方参考模型论文（ResNet-RS, ConvNeXt-V2, GhostNet-V2, EfficientViT, MobileViT-V2, StarNet）
   - 噪声标签学习相关论文（SCE, Peer Loss, GCE, ELR）
   - PyTorch 官方文档、timm 库
```

#### D5-2 必配图表清单
```yaml
# 技术方案中必须包含的图表:
# 图表1: 整体方案框架图（算法流程总览）
# 图表2: 数据分布图（类别分布 + 长尾分析）
# 图表3: 噪声样本示例（grid 图 + 噪声类型分类）
# 图表4: 模型结构图（backbone 结构）
# 图表5: 对比实验表格（不同 Loss 的 Acc）
# 图表6: 消融实验表格（每个组件的影响）
# 图表7: 训练曲线（loss、Acc、LR 变化）
# 图表8: 混淆矩阵（易混淆类别）
# 图表9: t-SNE 可视化（特征分布）
# 图表10: 伪代码（核心算法的伪代码，规则要求）
# 图表11: 过程进度表格（各阶段时间线）
```

### 7.6 任务包 D6：答辩 PPT（第 5-6 周，2 天）

#### D6-1 PPT 结构（15-20 页）
```yaml
# 封面: 赛题、队伍名、成员 (1页)
# 目录 (1页)
# 1. 背景与挑战 (2页)
#    - 细粒度识别 + 噪声标签问题
#    - 比赛约束
# 2. 数据探索 (2页)
#    - 数据分布分析
#    - 噪声标签分析
# 3. 方法设计 (3-4页) ★重点
#    - Backbone 选择及理由
#    - 损失函数设计
#    - 训练策略
# 4. 实验与分析 (3-4页) ★重点
#    - 对比实验结果
#    - 消融实验
#    - 可视化分析
# 5. 结果展示 (1-2页)
#    - 最终结果
#    - 成功/失败案例
# 6. 总结与展望 (1页)
# 7. Q&A (1页)
```

#### D6-2 答辩话术要点
```markdown
# 1分钟自我介绍:
"各位评委好，我们是XX队伍。本次比赛我们针对网络监督的细粒度图像识别任务，设计了一套从零训练的解决方案。
核心思路是一方面通过强数据增强提升模型泛化能力，另一方面设计噪声鲁棒的损失函数来缓解标签噪声的影响。
最终我们在ImageNet-1K上达到了XX%的Top-1准确率。"

# 常见 Q&A:
# Q: 为什么选这个 backbone？
# A: 考虑比赛约束（从零训练、单卡 4060），在性能和计算量之间做了权衡。
#    ResNet-50 训练效率高、显存友好，作为主模型；
#    ConvNeXt-V2-T 性能更优但训练更慢，作为冲刺尝试。

# Q: 噪声标签怎么处理的？
# A: 主要从三个角度：(1) 损失函数层面用 SCE/GCE 降低噪声影响；
#    (2) 训练策略层面用 ELR 保持 early learning 阶段的正确知识；
#    (3) 数据层面用样本选择机制筛选干净样本。

# Q: 为什么这个方法有效？
# A: 核心原因是我们在训练过程中没有过度信任标签，
#    而是让模型自己判断哪些样本可信、哪些不可信，
#    从而避免过拟合到噪声标签。

# Q: 如果时间更多会怎么改进？
# A: (1) 尝试 DivideMix 等更先进的噪声标签方法；
#    (2) 先用自监督预训练（对比学习），再用带噪标签微调；
#    (3) 用更大的 backbone（ConvNeXt-V2-S），配合更多训练技巧。
```

### 7.7 👥 多人协作方案

**1 人全周期 + 第 5-6 周全员冲刺：**
- **成员 F（文档主力）：** 第 1 周建立实验体系+问题日志 → 第 2 周搭 Docker 基础框架 → 第 3 周做预测脚本 → 第 4 周开始写技术方案 → 第 5 周完成技术方案+开始 PPT → 第 6 周答辩准备
- **全员：** 每周记录问题日志 → 第 5 周提供实验数据和图表素材 → 第 5-6 周参与模拟答辩

---

## 8. 完整协作时间表

### 8.1 4 人团队完整甘特图

假设 4 人：
- **A** = 数据（方向一）
- **C** = 模型（方向二）
- **E** = 策略（方向三）
- **F** = 文档（方向四）

```
          | 第1周   | 第2周   | 第3周   | 第4周   | 第5周   | 第6周   |
          | 4/28-5/4 | 5/5-11  | 5/12-18 | 5/19-25 | 5/26-6/1| 6/2-6/8 |
----------+---------+---------+---------+---------+---------+---------+
  A(数据)  | A1 A2   | A3+A5   | A4(等C) | A6 A7   | 支援    | 答辩    |
          | 下载+   | 噪声标注| 样本    | 混淆    | 图表给F | 准备    |
          | 统计    | +长尾   | 选择    | t-SNE   |         |         |
----------+---------+---------+---------+---------+---------+---------+
  C(模型)  | B1      | B2*     | B2继续* | B3      | 最终    | 答辩    |
          | Pipeline| ResNet  | B4技巧  | ConvNeXt| 模型    | 准备    |
          | 搭建    | 训练中  | 集成    | 实现    | 训练    |         |
----------+---------+---------+---------+---------+---------+---------+
  E(策略)  | 读论文  | C1-C4   | C5-C6   | C9      | 最终    | 答辩    |
          | 准备    | Loss实现| 扩展Loss| 超参    | 策略    | 准备    |
          |         | 快速实验| 对比    | 搜索    | 确定    |         |
----------+---------+---------+---------+---------+---------+---------+
  F(文档)  | D0+D1   | D2基础  | D3      | D5开始  | D5完成  | D6+D7   |
          | 问题日志| Docker  | 预测    | 技术    | +D6开始 | PPT+    |
          | +实验体系| 框架    | CSV     | 方案    | 答辩PPT | Q&A     |
----------+---------+---------+---------+---------+---------+---------+
            ★问题日志 ★Docker  ★对比实验 ★技术方案  ★提交包    ★答辩
            从此开始  基础搭好  串行进行  开始写     完成         
```

> `*` = 后台持续运行，不影响做其他工作

### 8.2 单张 4060 的串行训练计划

```yaml
# 只有 1 张卡，训练任务必须串行
# 每天检查训练进度，插入新实验

Week 1:
  Mon-Wed: A1数据下载 + B1 Pipeline搭建（无GPU需求，并行）
           F: 建立实验体系 + 开始问题日志
  Thu-Sun: B2 ResNet-50 Baseline开跑（GPU满负荷）
            同时: A2统计分析, E读论文, F搭实验体系
            ★ 在 10% 子集上快速测试 GhostNet-V2 / EfficientViT（半天）

Week 2:
  B2 继续训练中（第2-3天完成50epoch，周末完成100epoch）
  同时: A3噪声标注 + A5长尾分布分析, E实现Loss函数（无GPU需求）
  F: 搭建 Docker 基础框架，验证 Pipeline 可复现
  周末: 在 StarNet 上开始第一组Loss对比（E001, 1.5天）

Week 3:
  Mon: E001 CE baseline on StarNet (1.5天)
  Tue-Wed: E002 LabelSmoothing / E003 SCE on StarNet (3天)
  Thu-Fri: E004 PeerLoss / E005 GCE on StarNet (3天)
  Sat-Sun: 分析结果，选出 Top-2 Loss
  同时: A4样本选择集成, B4训练技巧集成, F做预测脚本

Week 4:
  Mon-Sat: Top-2 Loss 在 ResNet-50 上跑 100 epoch（6天）
  Sun: 超参搜索准备
  同时: B3编码完成但暂不训练（4060没空）, A6A7可视化
  F: 开始写技术方案（按官方大纲），Docker 持续验证

Week 5:
  Mon-Fri: 最优策略在 ResNet-50 上最终训练（5天）
  同时: C9超参搜索, F完成技术方案+开始答辩PPT
  Sat-Sun: 提交包整理 + Docker 复现验证（在另一台机器上测试）

Week 6:
  答辩准备全员冲刺
  Mon-Wed: PPT完善 + 模拟答辩 × 2次
  Thu-Fri: 最终提交包检查 + Q&A准备
```

> ⚠️ **核心矛盾**：只有 1 张 4060，训练任务必须排队。**解决方案**：小模型（StarNet）快速迭代筛选，大模型只在最后训练。

---

## 9. RTX 4060 训练时间速查表

### 9.1 各配置训练时间

| 模型 | Batch | 梯度累积 | Epoch数 | 每Epoch时间 | 总计时间 |
|------|-------|---------|---------|-----------|---------|
| StarNet | 256 | 2 | 50 | ~40min | ~1.5天 |
| StarNet | 256 | 2 | 100 | ~40min | ~3天 |
| ResNet-50 | 128 | 2 | 50 | ~70min | ~2.5天 |
| ResNet-50 | 128 | 2 | 100 | ~70min | ~5天 |
| ResNet-RS-50 | 96 | 3 | 100 | ~90min | ~6.5天 |
| ConvNeXt-V2-T | 64 | 4 | 100 | ~110min | ~8天 |
| ConvNeXt-V2-S | 32 | 8 | 100 | ~170min | ~12天 ❌ |

### 9.2 推理时间（验证集 50K + 测试集 100K）

| 模型 | Batch | 验证集(50K)时间 | 测试集(100K)时间 |
|------|-------|----------------|-----------------|
| StarNet | 256 | ~8min | ~16min |
| ResNet-50 | 256 | ~12min | ~24min |
| ConvNeXt-V2-T | 128 | ~18min | ~36min |

### 9.3 建议的实验调度

```bash
# 训练规划建议

# 阶段一：快速验证（白天工作，晚上训练）
# StarNet 每 1.5 天出一组结果
# 可用这段时间写代码、做分析

# 阶段二：主力训练（连续运行不中断）
# ResNet-50 需要 5 天连续跑
# 建议周一早上开始，周六完成
# 期间可以做不需要 GPU 的工作

# 阶段三：冲刺训练（最后两周）
# ConvNeXt-V2-T 需要 8 天
# 如果时间不够，回到 ResNet-50 + 更多技巧
```

---

## 10. 检查清单

### 10.1 代码完成度检查

- [ ] `src/data/imagenet_dataset.py` — ImageNet 数据集加载
- [ ] `src/data/transforms.py` — 数据增强（至少 RandAug + MixUp）
- [ ] `src/data/sample_selection.py` — 样本选择
- [ ] `src/models/resnet.py` — ResNet-50（随机初始化）
- [ ] `src/models/convnext_v2.py` — ConvNeXt-V2-T（可选）
- [ ] `src/models/build_model.py` — 模型工厂
- [ ] `src/losses/` — 至少 4 种 Loss（CE, LS, SCE, ELR）
- [ ] `src/trainers/base_trainer.py` — 完整训练循环（含 AMP + 梯度累积）
- [ ] `src/trainers/noisy_trainer.py` — 噪声鲁棒训练
- [ ] `src/utils/metrics.py` — 评估指标（Top-1, Top-5）
- [ ] `src/utils/seed.py` — 随机种子固定
- [ ] `src/utils/ema.py` — EMA 模型权重
- [ ] `scripts/train.py` — 训练入口
- [ ] `scripts/evaluate.py` — 验证入口
- [ ] `scripts/predict.py` — 测试集预测 + CSV（10 万张，文件名格式）
- [ ] `scripts/validate_csv.py` — CSV 格式校验
- [ ] `scripts/compute_accuracy.py` — 自行计算 Accuracy
- [ ] `problem_log.md` — 问题日志（每周更新）
- [ ] `docs/README.md` — 代码、环境、操作说明文档

### 10.2 实验完成度检查

- [ ] ResNet-50 Baseline（100 epoch）
- [ ] 至少 3 种 Loss 对比实验
- [ ] 数据增强消融实验
- [ ] 超参搜索记录
- [ ] 最优策略重复验证（2 次）

### 10.3 提交材料检查

- [ ] `submit/result.csv` — 格式正确（文件名,四位数字类别名），10 万行
- [ ] `submit/Dockerfile` — 可构建，包含完整训练+验证+预测环境
- [ ] `submit/requirements.txt` — 版本固定
- [ ] `submit/run.sh` — 一键复现（训练 → 验证 → 预测全流程）
- [ ] 完整的训练代码和验证代码 — 不只是推理，训练过程也要能复现
- [ ] `docs/README.md` — 独立的代码、环境、操作说明文档（不是技术报告的一部分）
- [ ] `submit/tech_report.pdf` — 技术方案（按官方七章大纲）
- [ ] 答辩 PPT（15-20 页）
- [ ] Docker 镜像可在另一台机器复现（赛事方会验证！）
- [ ] CSV 中文件名与测试集完全一致（大小写、扩展名）
- [ ] 自行计算的 Accuracy 与 CSV 一致

### 10.4 答辩准备检查

- [ ] PPT 完整（15-20 页）
- [ ] 讲稿逐字稿完成
- [ ] 至少 2 次模拟答辩
- [ ] Q&A 清单（至少 15 题）
- [ ] 每人准备好 1 分钟自我介绍
- [ ] 每人准备好负责部分的 2 分钟讲解

---

> **最后提醒**：
> 1. **三个致命错误不能犯**：CSV 格式写错（文件名+四位类别名）、Docker 无法复现（赛事方会验证）、技术报告不按官方大纲
> 2. **核心约束**：从零训练、无额外数据、不集成。推理时间和模型大小不限。
> 3. **主观分 30% 不能忽视**：问题日志从第 1 周开始记录，答辩从第 5 周开始准备
> 4. **长尾分布**是赛题明确提到的挑战，必须在数据分析和训练策略中体现
> 5. RTX 4060 虽然慢，但通过 StarNet 快速筛选 + ResNet-50 主力训练的策略，完全可以在 6 周内完成充分的实验迭代。关键在于**尽早跑通完整 Pipeline + Docker**，后期所有改进都是增量式的。

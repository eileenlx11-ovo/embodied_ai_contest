# ClassBalancedSampler 权重分析报告 (Week 3)

> 基于 Week 1 类别分布统计，为 A 提供采样权重，为 C 提供 Loss 调参参考。

## 一、数据集不平衡程度

| 指标 | 数值 |
|------|------|
| 类别数 | 1000 |
| 总图片数 | 1,281,167 |
| 单类最高 | 1300 (tench, goldfish 等) |
| 单类最低 | 732 (black-and-tan coonhound) |
| 均值 | 1281.2 |
| **不平衡比** | **1.78x** |

结论：ImageNet-1K **相对均衡**，不平衡比 < 2x。但这不意味着不需要处理——尾部 20 类的样本数仅为头部类的 56%，对少样本类的 Recall 仍有影响。

## 二、采样权重策略

三种 beta 值的权重对比：

| beta | 含义 | 最高权重 | 最低权重 | 权重比 | 推荐场景 |
|------|------|----------|----------|--------|----------|
| 0.0 | 均匀采样 (baseline) | 1.000 | 1.000 | 1.00x | 对照组 |
| **0.5** | **平方根倒数 (推荐)** | **1.321** | **0.991** | **1.33x** | **主力策略，温和平衡** |
| 1.0 | 全逆频 (激进) | 1.743 | 0.981 | 1.78x | 消融实验 |

### Top-10 尾部类别 (beta=0.5)

| class_id | 类别名 | 图片数 | 采样权重 |
|----------|--------|--------|----------|
| n02089078 | black-and-tan coonhound | 732 | 1.3210 |
| n02091635 | otterhound | 738 | 1.3156 |
| n02089973 | English foxhound | 754 | 1.3016 |
| n02113978 | Mexican hairless | 755 | 1.3007 |
| n02085782 | Japanese spaniel | 772 | 1.2863 |
| n02087046 | toy terrier | 860 | 1.2285 |
| n03197337 | digital watch | 889 | 1.2084 |
| n03498962 | hatchet | 891 | 1.2071 |
| n03995372 | power drill | 908 | 1.1958 |
| n04370456 | sweatshirt | 931 | 1.1810 |

## 三、与噪声分析的交叉验证

对 Top-20 尾部类别与 Week 2 噪声 Top-20 做交叉对比：

| 尾部类别 | 噪声报告状态 | 处理建议 |
|----------|-------------|----------|
| n02110627 affenpinscher (954) | CLEAN (白底误判) | 正常上采样 |
| n02089078 coonhound (732) | 未出现在噪声 Top-20 | 正常上采样 |

**好消息**：尾部类别与噪声类别几乎不重叠。radio（n04041544，高噪声）有 1300 张图在头部，不是长尾问题。这意味着：

- **ClassBalancedSampler 和 SCE Noise Loss 可以独立使用**，不存在"噪声尾类被过度上采样放大"的风险。
- 可以在 ClassBalancedSampler 的同时，仅对 radio/modem 等噪声类加 SCE，两者正交。

## 四、给 A 的交付内容

### 使用方法

```python
from src.data.sampler import ClassBalancedSampler, build_class_weights
from torch.utils.data import DataLoader

# 加载权重（推荐 beta=0.5）
weights, stats = build_class_weights(
    "docs/week1/class_counts.csv", beta=0.5
)

# 构建采样器
sampler = ClassBalancedSampler(
    dataset=train_dataset,
    class_weights=weights,
    replacement=True,  # 尾部类可重复出现
)

# 训练时使用 sampler（不能同时用 shuffle）
train_loader = DataLoader(
    train_dataset, batch_size=128,
    sampler=sampler, num_workers=4, pin_memory=True,
)
```

### 权重文件

- **`docs/week3/sampling_weights.json`** — 包含 beta=0.0 / 0.5 / 1.0 三组权重
- 格式: `{beta_X: {stats: {...}, weights: {class_id: {class_name, num_images, weight}}}}`

## 五、给 C 的交付内容

### Loss Weight 使用方法

```python
from src.data.sampler import build_class_weights, compute_loss_weights

weights, _ = build_class_weights("docs/week1/class_counts.csv", beta=0.5)
loss_weights = compute_loss_weights(weights, scale=1.0)

# nn.CrossEntropyLoss(weight=loss_weights)
criterion = nn.CrossEntropyLoss(weight=loss_weights)
```

### 建议

1. **默认使用 CE + beta=0.5 的 loss weight**，对尾部类损失放大 1.3x
2. 不要对 radio（n04041544）同时加采样权重和 SCE——该类别数据量充足（1300 张），只需 SCE 降噪
3. 消融实验：beta=0.0 (no weight) vs beta=0.5 (mild) vs beta=1.0 (full)，验证最佳配置

## 六、后续

| 任务 | 状态 | 依赖 |
|------|------|------|
| 生成 sampling_weights.json | ✅ 完成 | — |
| 交付 A 采样权重 | ✅ 可交付 | — |
| 交付 C loss 调参建议 | ✅ 可交付 | — |
| 验证 ClassBalancedSampler 集成 | ⏳ 待 A 接入训练管线 | ResNet-50 checkpoint |

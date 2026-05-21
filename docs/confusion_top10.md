# 混淆矩阵分析报告 (Week 3)

> Baseline: ResNet-50 / 100 epoch / Val Top-1 **77.53%** / Top-5 **93.63%**
> Checkpoint: `resnet50_baseline.pth`, EMA weights, epoch 100

---

## Top-10 易混淆类别对

按验证集误分类样本数排序：

| 排名 | 真实类别 | 误判为 | 混淆次数 | 混淆类型 | 诊断 |
|------|----------|--------|----------|----------|------|
| 1 | laptop (n03642806) | notebook (n03832673) | 27 | 细粒度视觉 | 笔记本电脑外观高度相似，ImageNet 将两者分为独立类别但差异极小 |
| 2 | maillot (n03710637) | maillot (n03710721) | 26 | **ImageNet 标签 BUG** | 两个不同 class ID 共享相同类名 "maillot"，是数据集已知问题 |
| 3 | projectile (n04008634) | missile (n03773504) | 24 | 语义重叠 | 投射物 vs 导弹——类别层级重叠，导弹是投射物的子类 |
| 4 | academic gown (n02669723) | mortarboard (n03787032) | 22 | **共现混淆** | 学位袍和学位帽在毕业照中同时出现，模型被上下文误导 |
| 5 | cassette player (n02979186) | tape player (n04392985) | 22 | 近同义词 | 卡带播放器 vs 磁带播放器——功能外观几乎一致 |
| 6 | sidewinder (n01756291) | horned viper (n01753488) | 20 | 细粒度 | 蛇类物种细粒度区分——侧进蛇 vs 角蝰 |
| 7 | Appenzeller (n02107908) | Greater Swiss Mountain dog (n02107574) | 20 | 细粒度 | 犬种细粒度区分——阿彭策尔犬 vs 大瑞士山地犬，均为瑞士山地犬种 |
| 8 | **ear (n13133613)** | corn (n12144580) | 20 | **同形异义 BUG** | "ear" = 耳朵，但 "ear of corn" = 玉米穗。ImageNet 的 ear 类混杂了耳朵和玉米穗图片 |
| 9 | maillot (n03710721) | maillot (n03710637) | 19 | ImageNet 标签 BUG | 同上 #2，两个方向的混淆合计 **45 次** |
| 10 | terrapin (n01667778) | mud turtle (n01667114) | 18 | 细粒度 | 龟类物种——淡水龟 vs 泥龟，外观极难区分 |

> 注：#2 + #9 是同一个问题（maillot 双向混淆），合计 45 次，是实际最严重的混淆。

---

## 混淆模式分析

### 模式 1：ImageNet 数据集本身的标签问题（3 对，合计 65 次）

| 问题 | class_id 对 | 说明 |
|------|------------|------|
| **maillot 重复类** | n03710637 ↔ n03710721 | ImageNet-1K 包含两个类名完全相同的独立类别，模型不可能区分 |
| **ear 同形异义** | n13133613 → n12144580 | "ear" 类混杂了耳朵和玉米穗图片，20 次误判实质是标签正确但类名歧义 |

**建议**：这两类混淆不应归咎于模型能力，报告中需注明为数据集 noise。对 maillot 类可合并训练，对 ear 类建议人工清洗或排除。

### 模式 2：细粒度视觉相似（3 对，合计 58 次）

| 对 | 领域 |
|----|------|
| laptop → notebook | 电子产品 |
| sidewinder → horned viper | 蛇类 |
| terrapin → mud turtle | 龟类 |

这些类别外观高度重合，即使在人类专家眼中区分也需专业知识。

**建议**：对这些特定对启用 **triplet loss 辅助** 或 **对比学习**，增强细粒度判别能力。

### 模式 3：语义重叠 / 近同义词（2 对，合计 46 次）

| 对 | 关系 |
|----|------|
| projectile → missile | 上位词/下位词 |
| cassette player → tape player | 功能近同义 |

**建议**：这些对本质上不是模型错误，而是类别定义本身模糊。不推荐针对它们加重惩罚。

### 模式 4：共现混淆（1 对，22 次）

| 对 | 原因 |
|----|------|
| academic gown → mortarboard | 毕业照中常同时出现 |

**建议**：通过数据增强（CutMix 混合不同场景）减少上下文依赖。

---

## 与 Week 2 噪声分析的交叉验证

Week 2 噪声 Top-20 发现的高噪声类别（radio、modem、nematode）**未出现在混淆 Top-10 中**。这揭示了两类不同的问题：

| 问题类型 | 代表类别 | 表现 | 诊断 |
|----------|----------|------|------|
| **标签噪声**（Week 2 发现） | radio (n04041544, ~60% 噪声率) | 类内标签错误——图片不是收音机但标为收音机 | 类内混乱，不影响其他类 |
| **类别混淆**（Week 3 发现） | laptop/notebook, maillot/maillot | 类间难以区分——模型将 A 类错判为视觉极相似的 B 类 | 类间边界模糊 |

**关键结论**：radio 类噪声率高但类间边界清晰（收音机不会和调制解调器混淆），说明 SCE 应对标**类内噪声**；而混淆矩阵对应的细粒度问题需要**对比学习**。两者的技术方案互补。

---

## 给 C 的 Loss 策略建议

### 分档建议

| 标准 CE | SCE α=0.5 β=0.5 | SCE α=0.7 β=0.3 | 对比学习 (triplet) |
|---------|------------------|------------------|-------------------|
| 95%+ 干净类（默认） | radio、modem 等噪声类 | maillot 重复类（合并处理） | 犬种/蛇类/龟类细粒度对 |

### 优先级

1. **maillot (n03710637 + n03710721)**：建议在数据预处理中合并为一个类，或训练时两个类使用相同的 soft label
2. **ear (n13133613)**：查看 20 次被误判为 corn 的样本——如果确实是"玉米穗"图，则 ear 类的标签本身有问题，需数据清洗
3. **laptop/notebook + cassette/tape**：这两个对占总混淆的 27+22=49 次，激活 triplet loss 微调这两对
4. **dog breeds + snake species**：20+20=40 次，细粒度通用问题，建议样本加权（对错误样本加大权重）

---

## 交付清单

| 产出 | 路径 | 状态 |
|------|------|------|
| 预测结果 (npz) | `docs/week3/predictions.npz` | ✅ |
| 混淆对 JSON | `docs/week3/confusion_pairs.json` | ✅ |
| 混淆对 Grid 图 | `docs/week3/confusion_grids/*.png` | ✅ 10 张 |
| 本报告 | `docs/confusion_top10.md` | ✅ |
| 采样权重表 | `docs/week3/sampling_weights.json` | ✅ (A5) |

### 交付给 C

→ **`docs/confusion_top10.md`** — 易混淆 Top-10 + 分类诊断 + SCE/对比学习分档建议

### 交付给 A

→ **`docs/week3/sampling_weights.json`** — ClassBalancedSampler 三组 beta 权重

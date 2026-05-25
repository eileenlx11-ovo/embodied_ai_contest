# 答辩 PPT 大纲

> 题目：网络监督的细粒度图像识别（Webly-Supervised Fine-Grained Image Recognition）
> 队伍：上海理工大学
> 共 10 页 · 答辩时长建议 8-10 分钟（每页 50-60 秒）

---

## P1 · 封面

**标题**：网络监督的细粒度图像识别 — ResNet-50 全增广 Recipe 与 Bug 驱动的严格消融

**副信息**：
- 队伍名 / 学校
- 成员：陈（项目工程）· 乔（训练）· 姚（数据）· 龚（算法）
- 日期：2026 年 6 月

**视觉**：简洁封面，居中 logo + 一张缩略图（class_distribution_full.png 或 noise_top20 截图，体现"网络数据"主题）

**口播 30 秒**：自我介绍 + 一句话定位 — "我们的方案最终选择了完整增广 recipe + 严格归因的 ablation 实验，而不是文献中流行的鲁棒损失路线；下面会用 4 组负样本实验和 3 个 trainer bug 复盘解释这个判断。"

---

## P2 · 赛题理解与挑战

**赛题**：ImageNet-1K 类别空间下，从含噪 web 训练数据中从零训练分类模型

**三大挑战**（左侧）：
1. **标签噪声** — 整体噪声率 5–15%，部分类别（radio ~60%、modem、spatula）严重歧义
2. **长尾分布** — 1000 类样本数 500–1300 不均衡
3. **数据偏差** — web 图像 vs 标准 ImageNet val 域分布差异

**数据规模**（右侧表）：
| 集合 | 数量 |
|---|---|
| 训练 | 1,281,167 |
| 验证 | 50,000 |
| 测试 | 100,000 |

**我们的切入点**：噪声率不算高 → 优先工程化基线 + 增广 recipe，不押宝复杂噪声鲁棒损失

**视觉**：左 1/3 文字三挑战，中 1/3 数据规模表，右 1/3 一张噪声 Top-20 类别条形图（来自 noise_top20.md）

**口播 60 秒**：拿 radio/modem 当例子，但强调整体噪声仍属"弱噪场景"，这一判断后面会用实验证伪鲁棒损失。

---

## P3 · 整体方案框架

**一张大图横贯左到右**：

```
[数据准备]      [训练]                  [推理]              [提交]
  ↓             ↓                       ↓                   ↓
完整 1000 类  ResNet-50 + 全增广     多模型 softmax 集成   result.csv
val 重建      Mixup α=0.2             + HFlip TTA          格式校验
              CutMix α=1.0
              RandAugment + RE
              CE + LS 0.1
              SGD + Cosine + EMA
              AMP / Grad Accum=2
```

**右下角小框**：硬件 — 单卡 RTX 4090D 24GB，单 epoch ~33min

**配色建议**：四个阶段四种淡背景色（蓝/绿/黄/紫），箭头加粗

**口播 60 秒**：强调"从零训练 / 单卡 / 100 epoch / 全 imagenet-1k"四个关键约束，把后面所有设计决策的硬件预算讲清楚。

---

## P4 · 数据处理与增广 Recipe

**左半**：数据质量
- 损坏文件 0 张，近空白图 0 张
- 极小图：50 余类各 1–3 张，总数百张占比 < 0.05%，忽略
- **结论**：数据本身可用，瓶颈在"标签"不在"图像"

**右半**：增广 Recipe 演进
| 阶段 | 增广组合 | 用于 |
|---|---|---|
| 基础 | RRC(224) + HFlip | baseline_v1 |
| 进阶 | + RandAugment(N=2,M=9) + RandomErasing | mixup_v2 |
| 正则 | + Mixup(α=0.2) + CutMix(α=1.0) prob=0.5 | mixup_v2 |

**底部强调框**：30ep schedule 下 mixup-only vs 纯 CE+LS：**69.78 vs 66.63（+3.15%）** — 单技术最大增益

**视觉**：左侧饼图（数据质量），右侧 recipe 表 + 底部蓝色高亮框写收益数字

**口播 50 秒**：先把"数据本身没事"立住，再讲增广 recipe 收益数字，自然过渡到下一页的鲁棒损失对比。

---

## P5 · 损失函数对比 — 鲁棒损失证伪

**主表**（占页面 70%）：

| Loss | 超参 | 40ep val_top1 | 结论 |
|---|---|---|---|
| **CE + LS 0.1** | smoothing=0.1 | **66.63** | 基线 |
| SCE | α=0.5, β=0.5 | 66.63 | 与 CE 持平，零增益 |
| SCE | α=0.1, β=1.0 | 60.66 | β 过大反而退化 |
| GCE | q=0.7 | 26.01 | 直接崩盘 |

**右下角**："**4 组实验 → 0 个超越 CE+LS**"红色大字

**结论 bullets**：
- 弱噪场景（5–15%）下，鲁棒损失为高噪（40%+）设计，假设不成立
- 不是负面结论 — 是"明确证伪 → 算力转向增广方向"的关键决策依据

**口播 70 秒**：这页是创新点之一。重点讲"我们花了一周 4 组实验，最终把整个技术路线从鲁棒损失切换到增广"，强调实验过程的严谨性。

---

## P6 · 核心消融实验汇总

**全页一张统一表**（重点页）：

| 实验 | 增广 | Loss | Sched | val_top1 | 说明 |
|---|---|---|---|---|---|
| baseline_v1 | 弱 | CE+LS | 100ep | **77.53** | 基线（已收敛） |
| rerun30_ce_ls01 | 弱 | CE+LS | 40ep | 66.63 | 短锚点 |
| mixup_only_e30 | 弱+Mixup | CE+LS¹ | 30ep | **69.78** | 单技术最大增益 |
| cutmix_only_e30 | 弱+CutMix | CE+LS¹ | 30ep | 69.12 | 与 mixup 近似 |
| **mixup_v2** | **全增广** | CE+LS | 100ep | **TBD（训练中）** | 预期 78–80% |

> ¹ Bug #1 未修，LS 实际未生效，反映"+Mixup −LS"复合效应

**右上角小框**：mixup_v2 进度条 — `[████░░░░░] 8/100 ep · 预计 5/28 完成`

**视觉**：表格用三色高亮 — 灰（baseline）/ 黄（短锚点）/ 蓝（mixup 系列）/ 红框（mixup_v2 TBD）

**口播 60 秒**：先讲 100ep baseline 77.53% 是真实硬指标，再讲 30ep 系列做"快速筛选"省算力，最后用 TBD 留悬念引出 mixup_v2 是"完整 recipe 合力"的最终验证。

---

## P7 · 工程严谨性 — 3 个 Trainer Bug 复盘

**核心信息**：发现并修复 trainer 中 3 个隐蔽 bug，让消融实验归因 100% 干净 — 这是我们的方法论创新

**三栏并排**：

| Bug #1 · LS 静默失效 | Bug #2 · cudnn 配置冲突 | Bug #3 · resume scheduler |
|---|---|---|
| **现象**：mixup 路径完全跳过 label_smoothing | **现象**：deterministic=True 时 benchmark 不生效 | **现象**：mixup_e100_resume 从 ep30 续训 → val 72.51→12.64% 崩盘 |
| **影响**：mixup 系列 vs baseline 变量不可控 | **影响**：100ep 慢 5–15%（约 2–7h） | **影响**：浪费 70h 算力 |
| **修复**：mixup 分支显式注入 LS（mixed × (1-ε) + ε/N） | **修复**：deterministic=False + benchmark=True，trade off bit-exact 换速度 | **方案**：增加 resume_mode: finetune，只 load model+ema，不碰 sched |

**底部一句话**：发现 bug 不是失败 — 是把"看不见的实验变量"变成"看得见的方法论"

**视觉**：三个等宽卡片，红色标题 + 灰色现象 + 黑色影响 + 绿色修复（颜色暗示从问题到解决）

**口播 70 秒**：这页最能体现"团队工程能力"。重点讲 Bug #1：发现 baseline 和 mixup 不公平后回头审计代码，找到根因，修完才跑 mixup_v2 — 这种严谨性是答辩亮点。

---

## P8 · 推理端优化 — 零额外训练成本增益

**左半架构图**：

```
test image (10万张)
      ↓
  ┌──hflip TTA──┐
  ↓             ↓
  原图          翻转图
  ↓             ↓
  ResNet-50    ResNet-50
  baseline_v1  baseline_v1
  ↓             ↓
  softmax      softmax
  ↘           ↙
   两次平均
      +
  ┌──hflip TTA──┐ (mixup_v2 同样流程)
      ↓
  CPU logits 累加（400MB）
      ↓
  argmax → result.csv
```

**右半收益估算**：

| 技术 | 时间成本 | 预期增益 |
|---|---|---|
| HFlip TTA | ×2 推理时间 | +0.2–0.5% |
| 双模型集成 | ×N（N=ckpt 数） | +0.5–1.5% |
| **组合（baseline+mixup_v2+TTA）** | ~6 min（10万张） | **+0.7–2.0%** |

**关键工程细节**：CPU logits 累加器避免 GPU OOM（100k × 1000 × 4B ≈ 400MB）

**调用示例**（底部代码框）：
```bash
python scripts/predict.py \
  --checkpoint baseline_v1/best.pth checkpoints_mixup_v2/best.pth \
  --tta hflip --output submit/result.csv
```

**口播 50 秒**：强调"零额外训练成本" — 在比赛后期算力已锁死时，推理端是最后一道增益来源。

---

## P9 · 创新点 + 工程化 + 进度

**左半 · 5 大创新点**：
1. **系统性证伪鲁棒损失**：4 组实验为 webly-supervised 任务选型提供反例参考
2. **Bug #1 修复**：让 baseline 与 mixup_v2 loss 计算路径完全等价，差异 100% 归因于"增广 recipe 合力"
3. **推理端零额外训练成本增益**：TTA + 多模型集成 +0.7–2.0%
4. **快速筛选范式**：StarNet-12M @ 10% 子集快速验证 → ResNet-50 全量训练，迭代速度 ×10
5. **完整 Docker 可复现**：根目录 Dockerfile + .dockerignore + MODE 开关 + 跨机验证指南

**右半 · 时间线**（横向甘特图样式）：
```
4/28 ──5/4─── 数据准备
       5/5───5/11─ 算法 pipeline + 5 种 Loss
              5/12─5/15─ 数据问题修复
                     5/15──5/18 baseline 100ep （77.53）
                              5/19─5/25 Loss 对比 4 组
                                     5/22─5/25 增广对比 2 组
                                            5/26 Bug 复盘 + 修复
                                            5/26──5/28 mixup_v2 100ep
                                                   5/27─5/28 推理端优化
                                                          5/28──6/8 Docker 验证 + 答辩
```

**底部小框**：团队 4 人分工，陈（D · 项目工程）· 乔（A · 训练）· 姚（B · 数据）· 龚（C · 算法）

**口播 60 秒**：5 点创新先念标题、再选 1-2 个展开。时间线讲"我们没有 last-minute scramble，每个阶段都有交付"。

---

## P10 · 总结与展望

**三段式**：

**已交付**：
- ResNet-50 baseline 77.53% / 93.63%（100ep, EMA）
- 完整增广 recipe（mixup_v2 训练中，预期 78–80%）
- 4 组损失消融 + 3 个 Bug 复盘
- 推理端 TTA + 多模型集成 pipeline
- Docker 可复现 + 跨机验证指南

**待完成（5/28 前）**：
- mixup_v2 100ep 收敛 → 补全消融表
- baseline_v1 + mixup_v2 集成推理 → 最终 result.csv

**后续可探索**：
- Mixup/CutMix 与 noise-aware 损失的联合（高噪场景）
- ConvNeXt-V2-T 备选冲刺（如时间允许）
- ClassBalancedSampler / focal loss 应对长尾

**底部居中大字**："**谢谢，欢迎提问**"

**视觉**：简洁三栏 + 大留白，避免信息过载

**口播 30 秒**：30 秒收尾，留时间答辩。

---

## 制作要点

**字体**（per CLAUDE.md）：
- 中文：等线
- 英文：Times New Roman
- 代码：Consolas

**配色**：白底 + 深蓝主色（#1F4E79）+ 强调橙（#C00000，仅用于负向/Bug/红框）+ 灰色辅助文本

**配图来源**：
- `docs/class_distribution_full.png` — P2 噪声/长尾
- `docs/noise_samples/` — P2 配图候选
- `docs/week2_loss_curves.png` — P5/P6 配图候选
- 自己画的方案框架图 — P3
- 自己画的推理 pipeline 图 — P8

**避免**：
- 不要每页堆超过 7 个 bullet
- 不要直接复制技术方案.md 大段文字 — PPT 是讲解载体，详细文字留在报告里
- 不要用 emoji（per 项目风格）
- 表格列数 ≤ 5，超过的拆 2 行或两个表

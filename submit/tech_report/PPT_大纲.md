# 答辩 PPT 大纲

> 题目：网络监督的细粒度图像识别 — 单模型 ResNet-50 的噪声鲁棒训练方案
> 队伍：上海理工大学
> 建议时长：8-10 分钟
> 当前 HTML 版本：`submit/tech_report/answer_deck.html`

---

## P1 · 封面

**标题**：单模型 ResNet-50 的噪声鲁棒训练方案
**副标题**：ImageNet-1K from scratch · Mixup/CutMix · HFlip TTA
**关键信息**：上海理工大学 / 2026 年 6 月 / 团队成员

**口播**：一句话定位：我们最终没有选择复杂清洗或多模型集成，而是用 full data + 温和 Mixup/CutMix + raw/EMA 双评估，把 baseline 从 77.53 推到单模型 HFlip TTA 的 77.99。

---

## P2 · 赛题约束与目标

**必须强调**：
- 训练集：1,281,167 张；验证集：50,000 张；测试集：100,000 张
- 不使用预训练模型
- 不引入额外数据
- 不做多模型集成
- 最终提交为一个 checkpoint 生成的 CSV

**结论句**：方案目标不是堆复杂技巧，而是在规则允许范围内找到最稳的单模型训练 recipe。

---

## P3 · 最终结果

| 实验 | 数据 | 训练增强 | Val Top-1 | Val Top-5 |
|---|---|---|---:|---:|
| baseline_v1 | full | RRC + HFlip | 77.53 | 93.63 |
| cleaned_v1 | C1+C2 cleaned | RRC + HFlip | 76.84 | 92.63 |
| full_mixcut | full | Mixup0.1 + CutMix1.0 | 77.57 | 93.77 |
| full_mixcut + HFlip TTA | full | 同上，单模型 TTA | 77.99 | 93.97 |

**口播重点**：TTA 是同一个模型的原图/翻转图两次前向，不是 ensemble。

---

## P4 · 为什么没有直接用清洗路线

**清洗设计**：
- C1：剔除明确类冲突样本
- C2：teacher-loss top 10%，per-class cap 15%
- exclude_file 加 sanity check，防止清洗列表没生效

**实验结论**：
- cleaned_v1 = 76.84，低于 baseline 77.53
- 高 loss 样本不一定都是错标，也可能是困难但有价值样本

**口播重点**：网络监督并不等于“删得越多越好”，本任务更需要保留覆盖度，用软监督降低噪声影响。

---

## P5 · 最终训练配方

**模型**：ResNet-50 from scratch
**Loss**：Cross Entropy + Label Smoothing 0.1
**优化**：SGD + Momentum 0.9 + WD 1e-4
**Schedule**：5ep warmup + cosine 100ep
**工程**：AMP、grad accumulation=2、channels_last、grad clip=5.0
**模型选择**：raw/EMA dual-eval

**增强**：
- Mixup α=0.1
- CutMix α=1.0
- mixup_prob=0.5：50% batch Mixup，50% batch CutMix
- 不使用 RandAugment / RandomErasing

**口播重点**：强增广在 100ep 下收敛不充分，最终用的是更轻、更贴合预算的增强。

---

## P6 · 噪声鲁棒性解释

**核心逻辑**：
- Web label 噪声会让 one-hot hard label 过度自信
- Label smoothing 把目标分布软化
- Mixup/CutMix 把监督从单样本硬标签变成邻域软标签
- 训练仍使用全部数据，不牺牲覆盖度

**一句话**：最终方案是“保留样本、软化监督”，而不是“强行删样本”。

---

## P7 · 消融与负结果

| 路线 | 结果 | 结论 |
|---|---:|---|
| CE + LS baseline | 77.53 | 强基线 |
| SCE / GCE | 未超过 CE+LS，GCE 退化明显 | 高噪声 loss 不适合当前弱噪场景 |
| Mixup+CutMix+RandAug+Erasing | 76.91 | 100ep 下强增广收敛不足 |
| C1+C2 cleaned | 76.84 | 删除高 loss 样本损害覆盖度 |
| Mixup0.1+CutMix1.0 | 77.57 | 最终训练 recipe |

**口播重点**：负结果不是失败，而是帮我们定位“这个赛题真正有效的复杂度边界”。

---

## P8 · 工程闭环

**关键修复**：
- Mixup 分支 label smoothing 生效
- CutMix 输入 clone，避免 in-place 副作用
- resume scheduler 模式修复，避免 LR 反向爬升
- raw/EMA dual-eval，保存 `best_source`
- `latest.pth` 仅用于恢复训练，最终只用 `best.pth`
- CSV 100000 行格式校验通过

**口播重点**：这些修复保证了实验可比性和最终推理加载的是正确权重。

---

## P9 · 提交与复现

**最终 checkpoint**：
```text
checkpoints_full_mixcut/best.pth
epoch=100
best_source=raw
best_acc=77.57
```

**预测命令**：
```bash
python scripts/predict.py \
  --config configs/imagenet_resnet50_full_mixcut.yaml \
  --checkpoint checkpoints_full_mixcut/best.pth \
  --tta hflip \
  --output submit/result.csv
```

**输出**：`submit/result.csv`，100000 rows，格式校验通过。

---

## P10 · 总结与展望

**总结**：
- 单模型、从零训练、无额外数据、无预训练
- 最终 validation：77.57 plain，77.99 HFlip TTA
- 最有效路线：full data + mild Mixup/CutMix + dual-eval
- 清洗、强增广、复杂鲁棒 loss 均作为负结果支撑决策

**展望**：
- 更长 schedule 下重新评估强增广
- 尝试更现代主干，但仍保持 from-scratch
- 对高噪声类别采用软权重，而非直接删除

**收尾**：谢谢，欢迎提问。

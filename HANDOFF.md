# 项目交接文档

> 交接人：乔（D 组）→ 组长
> 日期：2026-05-18
> 赛题：网络监督的细粒度图像识别（Webly-Supervised Fine-Grained Image Recognition）

---

## 一、当前进度总览

| 项目 | 状态 | 说明 |
|------|------|------|
| Baseline 训练 | ✅ 完成 | ResNet-50, 100ep, **Val Top-1 77.53% / Top-5 93.63%** |
| 数据增强消融 | ⏸ 就绪未启动 | Mixup/CutMix/RandAug config 已写好，未正式跑 |
| 噪声鲁棒 Loss | ❌ 需重做 | 历史 SCE/GCE/ELR 实验全部无效（val 数据当时损坏） |
| 测试集预测 | ❌ 未开始 | 赛题 10 万张测试集尚未获取 |
| Docker 提交包 | ✅ 可用 | Dockerfile + run.sh 端到端流程已验证 |
| 技术报告 | 🔶 草稿 | 框架已有，实验数据待填充 |

**结论：客观分已达"目标线"（76%），距"冲刺线"（78%）差 0.47pp。主观分 30% 的核心支撑——噪声鲁棒实验——尚无有效数据。**

---

## 二、服务器信息

| 项目 | 值 |
|------|-----|
| SSH | `ssh -p <PORT> root@<HOST>`（凭据见团队内部文档，勿提交到仓库） |
| GPU | RTX 4090D, 24GB VRAM |
| 项目路径 | `/root/autodl-tmp/embodied_ai_contest/` |
| 数据路径 | `/root/autodl-tmp/imagenet_full/ILSVRC/Data/CLS-LOC/` (train + val) |
| Python | `/root/miniconda3/bin/python` |
| 训练日志 | `/root/autodl-tmp/embodied_ai_contest/train_full.log`（baseline 完整日志） |
| Checkpoint | `/root/autodl-tmp/embodied_ai_contest/checkpoints/best.pth`（77.53%） |

**注意事项：**
- SSH 连接慢，paramiko 需设 `timeout=60, banner_timeout=120`
- 启动训练必须加 `PYTHONDONTWRITEBYTECODE=1` 防缓存旧代码
- GPU 当前空闲（0% / 0 MiB），可直接使用
- SSH 凭据请联系乔（D 组）获取，不要提交到仓库

---

## 三、代码结构

```
embodied_ai_contest/
├── configs/                    # YAML 配置（切换实验只需换 config）
│   ├── imagenet_resnet50.yaml          # baseline（已跑完）
│   ├── imagenet_resnet50_mixup.yaml    # 增强消融（待跑）
│   ├── imagenet_resnet50_noema.yaml    # 短期实验用（禁 EMA）
│   ├── imagenet_convnext.yaml          # ConvNeXt-V2-T（备选）
│   └── imagenet_starnet.yaml           # StarNet（快速迭代用）
├── src/
│   ├── models/     # ResNet-50, ConvNeXt-V2, StarNet + build_model 工厂
│   ├── losses/     # CE, SCE, GCE, ELR, PeerLoss, LabelSmoothing, Composite
│   ├── trainers/   # base_trainer（AMP+EMA+梯度累积）, noisy_trainer
│   ├── data/       # ImageNet loader, transforms, sampler, noise detection
│   ├── analysis/   # 类别统计, 噪声分析
│   └── utils/      # metrics, EMA, seed, visualization
├── scripts/
│   ├── train.py            # 训练入口（--config 指定 yaml）
│   ├── evaluate.py         # 验证（checkpoint → val Top-1/Top-5）
│   ├── predict.py          # 测试集推理 → CSV
│   └── validate_csv.py     # CSV 格式校验
├── submit/                 # 赛事提交包
│   ├── run.sh              # 一键复现（训练→验证→预测→校验）
│   └── requirements.txt    # 锁版本依赖
├── Dockerfile              # 在仓库根目录构建
├── docs/experiments.md     # 实验记录（已更新）
└── problem_log.md          # 踩坑日志（技术报告素材）
```

**训练命令：**
```bash
cd /root/autodl-tmp/embodied_ai_contest
PYTHONDONTWRITEBYTECODE=1 nohup /root/miniconda3/bin/python -u scripts/train.py \
  --config configs/imagenet_resnet50_mixup.yaml \
  > logs/resnet50_mixup_randaug.out 2>&1 &
```

---

## 四、已完成的工作

### 4.1 Baseline（77.53%）
- 配置：SGD 0.1 / cosine / warmup 5ep / batch 128×accum 2 / AMP / EMA 0.9999
- 耗时：~50h on 4090D
- Checkpoint：`checkpoints/best.pth`（含 model + ema + optimizer + scheduler + scaler）
- 已超过计划"目标线"76%

### 4.2 基础设施
- Docker 提交流程端到端可用
- WandB 集成（可选开启）
- 梯度累积 + AMP + channels_last + EMA
- 多 backbone 支持（ResNet-50 / ConvNeXt-V2 / StarNet）
- 多 Loss 支持（CE / SCE / GCE / ELR / PeerLoss / Composite）
- 数据增强开关（RandAug / Mixup / CutMix / RandomErasing / ColorJitter）

### 4.3 数据分析
- 类别分布分析：不平衡比 1.78，无需长尾处理
- 噪声检测脚本：`scripts/detect_noise.py`
- 已知噪声类：radio 类 ~60% 标签歧义

---

## 五、已知问题与踩坑记录

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | val 目录必须按 ImageFolder 组织（1000 个类子文件夹） | 扁平 val 导致 Val=0%，所有实验结论无效 | ✅ 已修复 |
| 2 | EMA decay=0.9999 在短期实验中坍缩 | 短期实验 val 全程 0% | ✅ 已修复（短期实验用 noema config） |
| 3 | 历史 SCE/GCE/ELR 实验（5/11~12）全部无效 | 跑在 val 损坏期间，结果无信号 | ⚠️ 需重做 |
| 4 | 赛题 10 万张测试集未获取 | predict.py 无法生成提交 CSV | ⚠️ 待解决 |
| 5 | 计划写"4060 训练"但实际用 4090D | 技术报告措辞需据实修改 | ⚠️ 注意 |

详细踩坑记录见 `problem_log.md`（Week 1~3 共 8 条，可直接用于技术报告"问题与思考"章节）。

---

## 六、接下来要做的事（按优先级）

### P0：噪声鲁棒 Loss 实验（赛题核心，主观分 30% 的支撑）

历史实验全废，需重做。建议方案：
- **数据量**：30% subset（~384K 训练图）
- **Epoch**：40
- **对比组**：CE(baseline) / SCE(α=0.1,β=1.0) / SCE(α=0.5,β=0.5) / GCE(q=0.7) / ELR(β=0.9,λ=3.0)
- **预计耗时**：~10h/实验 × 5 = 50h
- **关键**：必须用 `imagenet_resnet50_noema.yaml` 为基础（短期实验禁 EMA）
- **评估**：Val Top-1 + 噪声类（radio 等）预测分布

### P1：Mixup/CutMix 增强消融

Config 已就绪：`configs/imagenet_resnet50_mixup.yaml`
- Mixup α=0.2 + CutMix α=1.0 + RandAug(2,9) + RandomErasing
- 100 epoch, 预计 ~50h
- 目标：突破 78%（冲刺线）

### P2：获取赛题测试集

- 确认赛事方是否已发布 100K 测试集
- 下载后放到 `/root/autodl-tmp/embodied_ai_contest/data/imagenet/test/`
- 用 `scripts/predict.py` 生成 `submit/result.csv`
- 用 `scripts/validate_csv.py` 校验格式

### P3：技术报告完善

- 填入 baseline 实验数据
- 补充噪声鲁棒实验结果（P0 完成后）
- 硬件描述改为 4090D
- "问题与思考"章节直接引用 `problem_log.md`

### P4：提交前检查

- [ ] Docker 构建 + 端到端复现
- [ ] CSV 格式确认（逗号分隔，有无空格，有无 header）
- [ ] 模型大小 / 推理时间（规则说不限，但报告里要写）
- [ ] 确认 `submit/requirements.txt` 版本与服务器一致

---

## 七、赛事关键约束（必须遵守）

1. **从零训练**：不可使用任何预训练权重
2. **无额外数据**：只能用赛事提供的 ImageNet-1K
3. **不可集成**：最终提交只能是单模型
4. **复现性**：Docker 容器内必须能从零训练到推理

---

## 八、评分公式

```
总成绩 = 70% × 客观分（Accuracy 排名标准化）+ 30% × 主观分（答辩 + 技术方案 + 代码文档）
```

当前 77.53% 在"有竞争力"到"顶尖"之间。主观分需要噪声鲁棒实验数据支撑。

---

## 九、Git 分支说明

| 分支 | 状态 | 说明 |
|------|------|------|
| `main` | 稳定 | PR 合并后的主线 |
| `d-week3-tech-report-and-data-fix` | 当前工作分支 | 含最新代码（mixup 支持 + val 修复 + 实验记录） |
| `feature/core-modules` | 已合并 | 核心训练模块 |
| `d-week1-infra-fixes` | 已合并 | Docker/WandB/依赖修复 |

**建议**：将 `d-week3-tech-report-and-data-fix` 合并到 main 后继续开发。

---

## 十、快速上手

```bash
# 1. 连接服务器（凭据见团队内部文档）
ssh -p <PORT> root@<HOST>

# 2. 进入项目
cd /root/autodl-tmp/embodied_ai_contest

# 3. 验证 baseline
/root/miniconda3/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/best.pth \
  --config configs/imagenet_resnet50.yaml

# 4. 启动新实验（以 mixup 为例）
PYTHONDONTWRITEBYTECODE=1 nohup /root/miniconda3/bin/python -u scripts/train.py \
  --config configs/imagenet_resnet50_mixup.yaml \
  > logs/resnet50_mixup_randaug.out 2>&1 &

# 5. 监控
tail -f logs/resnet50_mixup_randaug.out
nvidia-smi -l 30

# 6. Resume 中断的训练
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -u scripts/train.py \
  --config configs/imagenet_resnet50_mixup.yaml \
  --resume checkpoints_mixup/latest.pth
```

---

*如有疑问联系乔（D 组）。*

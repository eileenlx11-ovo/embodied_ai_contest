# 具身智能赛事 — 网络监督细粒度图像识别

> 训练 / 推理 / 复现的最简入口。完整算法方案、消融实验、问题日志见 [submit/tech_report/技术方案.md](../submit/tech_report/技术方案.md)。

## 环境

- Python >= 3.10
- PyTorch >= 2.0, < 2.5
- CUDA >= 12.1
- 主力训练机：RTX 4090D 24GB（AutoDL）

```bash
pip install -r submit/requirements.txt
```

## 数据

ImageNet-1K 下载到 `data/imagenet/`：

```
data/imagenet/
├── train/   # 1000 类子文件夹
├── val/     # 1000 类子文件夹
└── test/    # 10 万张测试图（赛事方提供）
```

## 训练

```bash
# 主力 baseline (ResNet-50, 100ep)
python scripts/train.py --config configs/imagenet_resnet50.yaml

# 全增广版本 (Mixup + CutMix + RandAugment + RE)
python scripts/train.py --config configs/imagenet_resnet50_mixup.yaml --run-name mixup_v2

# 中断续训
python scripts/train.py --config <cfg> --resume checkpoints/<run>/latest.pth --resume-mode full

# 跨 schedule 复用权重（避免 Bug #3：cosine state 反向爬升）
python scripts/train.py --config <cfg> --resume <ckpt> --resume-mode finetune
```

## 推理 + 提交

```bash
# 单模型
python scripts/predict.py --config configs/imagenet_resnet50.yaml \
                          --checkpoint checkpoints/baseline_v1/best.pth \
                          --output submit/result.csv

# 多模型集成 + HFlip TTA
python scripts/predict.py --config configs/imagenet_resnet50.yaml \
                          --checkpoint checkpoints/baseline_v1/best.pth checkpoints/mixup_v2/best.pth \
                          --tta hflip --output submit/result.csv

python scripts/validate_csv.py --csv submit/result.csv
```

## Docker 一键复现

Dockerfile 在仓库根，**在仓库根执行 build**：

```bash
docker build -t embodied-ai .
docker run --gpus all \
  -v /path/to/imagenet:/workspace/data/imagenet \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/submit:/workspace/submit \
  -e CONFIG=configs/imagenet_resnet50.yaml \
  embodied-ai
```

## WandB（可选）

`configs/*.yaml` 中：

```yaml
logging:
  wandb:
    enabled: true
    project: embodied-ai
    run_name: rn50_ce_baseline
    mode: online   # online | offline | disabled
```

离线模式：训练完在有网机器上 `wandb sync wandb/offline-run-*`。

## 文档索引

| 文档 | 内容 |
|---|---|
| [submit/tech_report/技术方案.md](../submit/tech_report/技术方案.md) | 算法方案 + 消融实验 + 问题日志（赛事提交主文档） |
| [submit/tech_report/PPT_大纲.md](../submit/tech_report/PPT_大纲.md) | 答辩 PPT 10 页大纲 |
| [submit/tech_report/answer_deck.html](../submit/tech_report/answer_deck.html) | Swiss-style 答辩 deck |
| [HANDOFF.md](../HANDOFF.md) | 项目交接（D 组 → 组长） |
| [problem_log.md](../problem_log.md) | 周记 Bug 跟踪原始记录 |
| [docs/experiments.md](experiments.md) | 实验日志（4090D 真实结果） |
| [docs/noise_top20.md](noise_top20.md) | Top-20 噪声类别分析 |
| [docs/archive/](archive/) | 早期备赛计划 / strategy_v1（仅历史参考，已不更新） |

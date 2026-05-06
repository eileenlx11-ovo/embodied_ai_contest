# 具身智能赛事 — 网络监督细粒度图像识别

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0, < 2.5.0
- CUDA >= 12.1
- GPU: RTX 4060 (8GB VRAM) 或更高

## 快速开始

### 1. 安装依赖

```bash
pip install -r submit/requirements.txt
```

可选：启用 WandB 实验跟踪：

```bash
pip install wandb
wandb login
```

### 2. 数据准备

CIFAR-10 会自动下载。ImageNet-1K 需手动下载到 `data/imagenet/`：

```
data/imagenet/
├── train/    # 1000 个子文件夹
├── val/      # 1000 个子文件夹
└── test/     # 测试集图片（10 万张）
```

### 3. 训练

```bash
# CIFAR-10 快速烟囱测试（2 epoch，CPU 也能跑）
python scripts/train.py --config configs/baseline.yaml --epochs 2 --device cpu

# ImageNet 子集快速验证（10%，5 epoch）
python scripts/train.py --config configs/baseline.yaml --subset 0.1 --epochs 5

# ImageNet 完整训练（需切 dataset=imagenet 的 config）
python scripts/train.py --config configs/baseline.yaml
```

### 4. 验证

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/baseline.yaml
```

### 5. 预测 + CSV

```bash
python scripts/predict.py --checkpoint checkpoints/best.pth \
                          --config configs/baseline.yaml \
                          --output submit/result.csv
python scripts/validate_csv.py --csv submit/result.csv
```

### 6. Docker 一键复现

Dockerfile 位于仓库根目录，**在仓库根执行** build：

```bash
docker build -t embodied-ai .
docker run --gpus all \
  -v /path/to/imagenet:/workspace/data/imagenet \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/submit:/workspace/submit \
  -e CONFIG=configs/baseline.yaml \
  embodied-ai
```

覆盖 config：`-e CONFIG=configs/other.yaml`

## WandB 实验跟踪

在 yaml 中启用：

```yaml
logging:
  wandb:
    enabled: true
    project: embodied-ai
    run_name: rn50_ce_baseline
    mode: online          # online | offline | disabled
```

无外网时用 `mode: offline`，训练完在有网机器上 `wandb sync wandb/offline-run-*`。

## 项目结构

见 `competition_plan_detailed.md` 第 3 节。

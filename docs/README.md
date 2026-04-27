# 具身智能赛事 — 网络监督细粒度图像识别

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 12.1
- GPU: RTX 4060 (8GB VRAM)

## 快速开始

### 1. 安装依赖

```bash
pip install -r submit/requirements.txt
```

### 2. 数据准备

CIFAR-10 会自动下载。ImageNet-1K 需手动下载到 `data/imagenet/`:

```
data/imagenet/
├── train/    # 1000 个子文件夹
├── val/      # 1000 个子文件夹
└── test/     # 测试集图片
```

### 3. 训练

```bash
# CIFAR-10 快速验证（2 epoch）
python scripts/train.py --config configs/baseline.yaml --epochs 2

# ImageNet 完整训练
python scripts/train.py --config configs/imagenet.yaml

# ImageNet 子集快速实验
python scripts/train.py --config configs/imagenet.yaml --subset 0.1
```

### 4. 验证

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/baseline.yaml
```

### 5. 预测 + CSV

```bash
python scripts/predict.py --checkpoint checkpoints/best.pth --config configs/baseline.yaml --output submit/result.csv
python scripts/validate_csv.py --csv submit/result.csv
```

### 6. Docker 复现

```bash
cd submit
docker build -t embodied-ai .
docker run --gpus all -v /path/to/imagenet:/data/imagenet embodied-ai
```

## 项目结构

见 `competition_plan_detailed.md` 第 3 节。

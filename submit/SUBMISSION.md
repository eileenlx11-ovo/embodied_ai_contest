# ResNet-50 · ImageNet-1K 噪声标签分类 — 复现说明

## 1. 方法概述
- 模型：ResNet-50，从零训练，1000 类输出。
- 数据：ImageNet-1K 全量（约 1.28M 张），不做数据清洗。
- 训练：SGD + cosine 学习率 + 5 epoch warmup，共 100 epoch；label smoothing 0.1；AMP 混合精度；EMA(0.9999)；以 Mixup(α=0.1) + CutMix(α=1.0) 缓解标签噪声。
- 推理：HFlip TTA（原图与水平翻转各推理一次，softmax 取平均）。单模型，**无模型集成**。
- 验证集 top-1：约 77.5%，叠加 HFlip TTA 约 78.0%。

## 2. 目录结构
- Dockerfile          构建文件
- submit/run.sh       复现脚本（训练→验证→预测→CSV 校验）
- submit/requirements.txt  依赖清单
- src/                模型、数据、训练器代码
- scripts/            train.py / evaluate.py / predict.py / validate_csv.py
- configs/imagenet_resnet50_full_mixcut.yaml  提交所用配置

## 3. 环境
- 基础镜像：pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
- 依赖：见 submit/requirements.txt（pip 安装）
- 硬件：单张 GPU，显存建议 ≥ 24GB（batch_size=128）

## 4. 数据准备
挂载 ImageNet-1K 数据集，根目录下需含 train/ val/ test/ 三个子目录（ImageFolder 格式，train/val 按类别建子文件夹，test 为待预测图片）。

## 5. 构建镜像与验证

前置：宿主机已装 Docker；跑训练/预测需 NVIDIA GPU + nvidia-container-toolkit（仅 build 不需要 GPU）。

步骤 1 — 构建（在仓库根目录，与 Dockerfile 同级执行）：

```bash
docker build -t embodied-ai .
```

步骤 2 — 验证镜像构建成功（无需 GPU/数据，几秒）：

```bash
docker images | grep embodied-ai                              # 确认镜像存在
docker run --rm embodied-ai python -c "import torch, torchvision, timm; print(torch.__version__)"   # 确认依赖装好、可导入
docker run --rm embodied-ai bash -c "ls scripts/ configs/imagenet_resnet50_full_mixcut.yaml run.sh"  # 确认代码与配置就位
```

以上三条全部正常 → 镜像可用，可进入完整复现。

## 6. 运行复现

完整流程（训练 → 验证 → 预测 → 校验 CSV，需 GPU + 已挂载数据）：

```bash
docker run --gpus all \
  -e DATA_ROOT=/data/imagenet \
  -v /宿主机/imagenet路径:/data/imagenet \
  embodied-ai
```

`-e DATA_ROOT` 指向容器内数据路径，`-v` 把宿主机数据集挂到该路径（二者一致）。脚本依次执行：训练（ResNet-50，100 epoch）→ 验证（输出 val top-1）→ 对 test 集预测（HFlip TTA）→ 校验 CSV 格式。最终产物：容器内 `submit/result.csv`。

可选 — 跳过 100 epoch 训练、仅用已训练权重复现预测 CSV（验证流水线是否打通，省去数天训练）：

```bash
docker run --gpus all \
  -e DATA_ROOT=/data/imagenet \
  -v /宿主机/imagenet路径:/data/imagenet \
  -v /宿主机/best.pth路径:/workspace/checkpoints_full_mixcut/best.pth \
  embodied-ai \
  bash -c 'sed "s#^  root:.*#  root: ${DATA_ROOT:-/data/imagenet}#" \
    configs/imagenet_resnet50_full_mixcut.yaml > configs/_runtime.yaml && \
    python scripts/predict.py --checkpoint checkpoints_full_mixcut/best.pth \
      --config configs/_runtime.yaml --output submit/result.csv && \
    python scripts/validate_csv.py --csv submit/result.csv'
```

预期：生成 100000 行 CSV，`validate_csv.py` 输出 `PASSED`。

## 7. 提交产物说明
submit/result.csv：test 集预测结果。格式为「图片文件名,类别」，类别 4 位补零，**无表头**，文件名保留原始 .JPEG 后缀。

## 8. 备注
- 随机种子固定为 42；GPU 非确定性算子可能引入约 ±0.1% 的精度波动，属正常范围。
- DATA_ROOT 环境变量用于覆盖配置中的数据路径，run.sh 会据此生成运行配置 configs/_runtime.yaml。
- 若需跳过 100 epoch 训练、直接由已训练权重复现预测 CSV：挂载 checkpoints_full_mixcut/best.pth 后，单独执行
  先按 DATA_ROOT 生成 configs/_runtime.yaml，再执行 `python scripts/predict.py --checkpoint checkpoints_full_mixcut/best.pth --config configs/_runtime.yaml --output submit/result.csv`

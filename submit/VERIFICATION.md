# 跨机 Docker 验证说明

> **目标**：在你自己的机器上验证 Docker 镜像能 build、容器能起、入口脚本能跑通，作为赛事方复现流程的预演。
> **不需要**：完整 ImageNet 数据（130GB）、训练 checkpoint、GPU（CPU 模式即可）。
> **耗时**：build 约 15-30 分钟（首次拉镜像）；验证本身 < 5 分钟。

---

## 前置条件

- Docker（任何近期版本，Linux/macOS/Windows + WSL2 均可）
- 磁盘空间 ≥ 15 GB（base image ~8GB + 依赖 ~3GB + 余量）
- 网络能访问 docker.io（拉 pytorch base image）

GPU 可选——验证 build/entrypoint 不需要，端到端 mock smoke 需要（见 Phase 3）。

---

## Phase 1：Build 验证（必做，~15-30 分钟）

```bash
git clone -b d-week3-predict-tta-and-report-v2 git@github.com:eileenlx11-ovo/embodied_ai_contest.git
cd embodied_ai_contest
docker build -t embodied-ai:test .
```

**预期**：最后一行是 `Successfully tagged embodied-ai:test`。
**失败排查**：
- pip 装包失败 → 网络问题，重试或换 pip 源
- `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` 拉不下来 → 配置 docker daemon mirror

---

## Phase 2：容器入口验证（必做，~1 分钟）

证明镜像里的 Python 环境、PYTHONPATH、scripts 入口都对：

```bash
docker run --rm embodied-ai:test python -c "from src.models.build_model import build_model; print('import OK')"
docker run --rm embodied-ai:test python scripts/predict.py --help
```

**预期**：
- 第一条输出 `import OK`
- 第二条输出 predict.py 的 argparse help（包含 `--checkpoint`、`--tta`、`--ensemble` 相关）

**失败排查**：
- `ModuleNotFoundError` → Dockerfile 漏 COPY 了某个目录
- `ImportError: libGL` → base image 没装 libgl1（Dockerfile 第 3 行已装）

---

## Phase 3：端到端 mock smoke（可选，进阶）

用一个随机初始化的 ResNet-50 + 5 张假图，跑完整 predict 流程，验证 entrypoint 链路没有静默坑。

**Step 1：本地准备 mock 资源（host 机器上跑，不在容器里）**

```bash
python -c "
import torch, os
from torchvision.models import resnet50
from PIL import Image
import numpy as np

os.makedirs('mock/ckpt', exist_ok=True)
os.makedirs('mock/test', exist_ok=True)
os.makedirs('mock/out', exist_ok=True)

# Fake checkpoint
model = resnet50(num_classes=1000)
torch.save({'model': model.state_dict()}, 'mock/ckpt/best.pth')
print('mock ckpt saved')

# 5 fake test images
for i in range(5):
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(f'mock/test/image_{i:05d}.JPEG')
print('mock images saved')
"
```

**Step 2：用容器跑 predict（MODE=predict 跳过 training）**

```bash
docker run --rm \
  -v $(pwd)/mock/ckpt:/workspace/checkpoints \
  -v $(pwd)/mock/test:/workspace/test_data \
  -v $(pwd)/mock/out:/workspace/submit \
  -e MODE=predict \
  -e CKPT=checkpoints/best.pth \
  -e CONFIG=configs/imagenet_resnet50.yaml \
  -e OUTPUT=submit/result.csv \
  embodied-ai:test bash run.sh
```

> 注意：predict.py 默认读 `cfg["data"]["root"]/test`，你需要确保 mount 的路径或 config 里指向 `/workspace/test_data`。如果 config 路径不匹配，加 `--test_dir /workspace/test_data` 到 run.sh 里测试。

**预期**：
- Step 3 输出 `Predicting 5 images from ...`
- Step 4 CSV 格式校验通过（或报"行数不足 10 万"，那是正常的，mock 只有 5 张）
- `mock/out/result.csv` 生成，5 行内容

**Step 3：清理**

```bash
rm -rf mock/
```

---

## 验证完成后请告诉我

任何 Phase 失败的命令完整输出 + 你的机器信息（OS、Docker 版本、是否有 GPU）。
我会基于真实失败 case 修，比我闭门猜要准。

Phase 1+2 全部 OK 就证明镜像 build/entrypoint 没坏，可以提交给赛事方。
Phase 3 OK 就额外证明 predict 链路在镜像内端到端能跑通。

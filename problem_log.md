# 问题日志

> 从第 1 周起记录，供技术报告"问题与思考"章节使用。每周至少 2-3 条。
> 记录格式：日期 / 问题描述 / 影响 / 解决方案 / 状态（已解决/跟进中/待验证）

## Week 1 (4/28 - 5/4)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 5/2 | `submit/Dockerfile` 使用 `COPY ../src/` 越过构建上下文，`docker build` 会直接报错 | 赛事方复现失败风险（赛事规则：无法复现将取消成绩） | 将 Dockerfile 移到仓库根目录，`COPY src/` 和 `COPY submit/requirements.txt` 都相对根路径；构建命令改为 `docker build -t embodied-ai .`（在仓库根执行） | 已解决 |
| 5/3 | `submit/run.sh` 引用不存在的 `configs/final_config.yaml`，容器启动后第一步就挂 | 端到端复现流程跑不通 | 改为从环境变量 `CONFIG` 读取，默认 `configs/imagenet_resnet50.yaml`；用户可通过 `docker run -e CONFIG=xxx` 覆盖 | 已解决 |
| 5/3 | 本机 GitHub HTTPS(443) TLS 层被重置，`git fetch` 报 "Recv failure: Connection was reset" | 无法拉取/推送，阻塞协作 | 将 origin 切到 SSH 并走 `ssh.github.com:443`（git@ssh.github.com:443/owner/repo.git），纯本地配置，不污染仓库 | 已解决 |
| 5/4 | `submit/requirements.txt` 未锁版本，队友环境里装到 torch 2.6/torchvision 0.21 会和 AMP/compile 接口漂移 | 跨机复现可能精度/行为不一致 | 给 torch/torchvision 加上下界+上界（`<2.5.0` / `<0.20.0`），其他包加下界；补 `wandb`、`timm`、`Pillow` | 已解决 |
| 5/4 | `base_trainer.py` 没有 WandB 集成，手写日志无法跨实验对比 | 后续 10+ 组 Loss 对比实验难以追踪 | 加可选 WandB 初始化：cfg.logging.wandb.enabled=true 才启用；未安装 wandb 时降级到 print；step-level 记 lr/running loss/acc，epoch-level 记 val top1/top5/best | 已解决 |

## Week 2 (5/5 - 5/11)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 5/8 | PR1+PR2 合并后需要确认训练入口是否端到端可用 | 若训练通路不通，后续 StarNet/损失函数实验和提交包都会被阻塞 | 已在最新 main 上完成 baseline CIFAR-10 + CE smoke test：CPU、subset=0.01、epochs=2、warmup=1，Best Val Top-1=10.02%；另补测 ELR/index-aware loss 单 epoch，输出 ELR smoke passed | 已解决 |
| 5/8 | 本机缺少 ImageNet 数据，无法直接执行 StarNet-S2 + 10% subset + 5 epoch 验证 | reviewer 建议的正式通路验证暂时不能在本机复现 | 已确认 data/imagenet/train 和 data/imagenet/val 均不存在；先用 CIFAR-10 小子集验证 StarNet-S2 backbone + trainer 通路，输出 StarNet CIFAR smoke passed；正式 ImageNet 验证需在有数据的机器上运行 | 跟进中 |

## Week 3 (5/12 - 5/18)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|

## Week 4 (5/19 - 5/25)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|

## Week 5 (5/26 - 6/1)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|

## Week 6 (6/2 - 6/8)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|

## 关键决策记录

| 日期 | 决策 | 理由 | 结果 |
|------|------|------|------|
| 5/2 | Dockerfile 放仓库根而非 `submit/` 子目录 | Docker 构建上下文不支持 `../`；根目录构建最干净，路径引用直观 | 待验证（需在赛事方机器复现） |
| 5/3 | 训练入口保留单一 `scripts/train.py`，通过 yaml 切换 backbone/loss | 降低 CLI 参数复杂度；对比实验只需换 config 文件 | 待验证 |

## 进一步优化思考

- Docker 构建层缓存：先 `COPY submit/requirements.txt` 再 `pip install`，再 `COPY src/`，源码变动不触发 pip 重装
- WandB 离线模式（`mode: offline`）可以在无外网的赛事机器上用，训练完 `wandb sync` 再上传
- 如果 SSH 也被限，考虑 Cloudflare WARP 或 GitHub CLI HTTPS API（token 认证，走 api.github.com 子域名）


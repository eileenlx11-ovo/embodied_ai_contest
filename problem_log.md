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
| 5/15 | `make_imagenet_10p.py` 的 `prepare_val()` 未成功创建 val 目录：Kaggle 原始 val 是 flat 的（50000 张平铺），脚本检测到非 class-folder 结构后尝试用 CSV 整理但未生效 | autodl-tmp 上 `imagenet_10p/val/` 不存在，训练时 val 评估完全随机（0.1%），所有 Loss 对比实验结论无效 | SSH 登录远程机器，用 `LOC_val_solution.csv` 将 flat val 按 synset 整理成 1000 个类别子文件夹（symlink），验证 train/val classes 完全对齐 | 已解决 |
| 5/15 | autodl-tmp1 上 `imagenet_10p/train/` 只有 123 个类（zip 解压中断） | ResNet-50 config 设 num_classes=1000 但 train 只覆盖 123 类；StarNet config 被手动改为 num_classes=123 掩盖了问题 | 将 autodl-tmp1/imagenet_10p symlink 到 autodl-tmp/imagenet_10p（完整 1000 类），修正 StarNet config num_classes=1000 | 已解决 |
| 5/15 | 之前所有 SCE/GCE 实验（logs/ 下 8 组）Val Top-1 均为 0.1-0.16%，等于随机猜测 | 两周的 Loss 对比实验数据全部作废 | 数据修复后重新启动 5 组 Loss 对比实验（CE/SCE×2/GCE/ELR），10% subset × 30 epochs | 已解决 |
| 5/15 | EMA decay=0.9999 在短期实验（10% subset + 30ep）中导致 EMA 模型坍缩：预测只输出 2 个类，max logit 达 64（正常应 <10） | validate() 使用 EMA 模型评估，Val Top-1 始终 0.1% 即使 train acc 已达 16%；对比实验无法得出有效结论 | 短期实验禁用 EMA（ema_decay=0），全量训练保留 EMA 0.9999；创建 `imagenet_resnet50_noema.yaml` 用于对比实验 | 已解决 |

## Week 4 (5/19 - 5/25)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 5/19 | 代码中混入明文凭据（服务器连接信息硬编码进提交历史） | 凭据泄漏风险，且会进入 git 历史难以彻底清除 | 移除明文凭据，连接信息收敛到本地未追踪的 server.py（已 gitignore）；后续密码于 5/20 轮换 | 已解决 |
| 5/19 | checkpoint resume 只恢复模型权重，未恢复 optimizer/scheduler/scaler 状态 | 中断续训时学习率、动量、AMP scale 全部重置，等于换了套超参，曲线断裂、复现性受损 | resume 逻辑补齐 optimizer/scheduler/GradScaler 三者的 state_dict 恢复，确保断点续训与连续训练等价 | 已解决 |

## Week 5 (5/26 - 6/1)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 5/26 | Bug #1: mixup 路径下 label_smoothing 失效 — mixup/cutmix 分支绕过了 LS loss 构造 | mixup_v2 前 26ep 实际用的是 hard CE，正则效果打折 | 修复 base_trainer.py 中 mixup 分支的 loss 调用，确保 LS 生效 | 已解决 |
| 5/26 | Bug #2: cudnn.deterministic=True 与 benchmark=True 同时设置 — deterministic 覆盖 benchmark 导致训练慢 ~5% | epoch_time 偏高（2184s vs 预期 2050s） | 训练时只开 benchmark=True，关闭 deterministic（seed 已固定初始化） | 已解决 |
| 5/27 | mixup_v2 全增广配方（Mixup+CutMix+RandAug N=2 M=9+RandomErasing）在 100ep 预算下无法收敛到 baseline 水平 | ep77 best=69.08%，远低于 baseline 77.53%；预计终值 71-75% | 诊断：该配方等价 timm A1-tier，需 300-600ep 收敛；100ep 下应使用 A3-tier（弱 aug 或无 aug）。结论：强增广实验作为消融对照保留，不作为提交模型 | 已解决（结论性） |
| 5/28 | 缺少 "mixup+cutmix only（无 randaug/erasing）" 的数据点 — 无法判断 cutmix 在 100ep 下是正收益还是负收益 | 最终配方选择缺乏依据 | 已跑 full_mixcut 100ep：Mixup α=0.1 + CutMix α=1.0，无 RandAug/Erasing，plain 77.57、HFlip TTA 77.99，最终作为提交候选 | 已解决 |
| 5/28 | 77.53% baseline ckpt 定位困难 — train_full.log 不在 metrics.csv 体系内，首次查找浪费 30min | 跨会话交接时容易找错日志文件 | 已在 HANDOFF.md 和 memory 中明确标注：权威日志=train_full.log，metrics.csv 中的 imagenet_baseline_v1 是 5/16 跑废的 10% subset 调试 run | 已解决 |

## Week 6 (6/2 - 6/8)

| 日期 | 问题描述 | 影响 | 解决方案 | 状态 |
|------|---------|------|---------|------|
| 6/2 | `submit/run.sh` 默认 CONFIG=imagenet_resnet50.yaml、checkpoint=checkpoints/best.pth — 指向 baseline 而非提交模型 | 评委按默认值复现会跑错配方、读错权重，复现结果对不上提交的 77.99% | 默认值改为 imagenet_resnet50_full_mixcut.yaml + checkpoints_full_mixcut/best.pth；与提交 ckpt 严格对齐 | 已解决 |
| 6/2 | config 内 data.root 硬编码 `/root/autodl-tmp/...`（我方 AutoDL 路径），与评委挂载路径必然不同 | 评委复现时 ImageFolder 找不到数据，端到端流程第一步即崩 | run.sh 新增 `DATA_ROOT` 环境变量，用 sed 生成 configs/_runtime.yaml 覆盖 root；评委只需 `docker run -e DATA_ROOT=...` 无需改 YAML | 已解决 |
| 6/2 | 对"可复现 docker"理解需校准：是交镜像 tar 还是源码包？ | 若误做成推理镜像（COPY 进 ckpt），违背规则 §九(二)"完整训练+验证代码+复现脚本"，可能被判不合规 | 确认交可复现训练源码包：Dockerfile 不 COPY checkpoint/数据，run.sh 跑完整 train→eval→predict→validate；评委自行 build & run | 已解决（结论性） |
| 6/2 | HFlip TTA（原图+水平翻转 softmax 平均）是否触犯规则 §5"禁止模型集成" | 若被判为集成，提交的 77.99% 作废、回退到 plain 77.57% | 判定：单模型、单组权重、仅推理期两次前向平均，不属于多模型集成 → 合规。技术报告需明确论证此点 | 已解决（结论性） |
| 6/2 | 提交模型选 raw 还是 EMA 权重 | 选错少 ~0.13pp | summary.json：raw top-1=77.57 > EMA 77.438，best_source=raw；提交 best.pth（raw） | 已解决 |
| 6/2 | 镜像无法在本机/服务器构建验证（均未装 docker，AutoDL 容器不能嵌套 docker） | Dockerfile/requirements 的潜在错误无法在提交前暴露 | 静态检查全过（py_compile、依赖、路径自洽）；提交前须在有 docker 的机器上 `docker build` 验证一次 | 待验证 |

## 关键决策记录

| 日期 | 决策 | 理由 | 结果 |
|------|------|------|------|
| 5/2 | Dockerfile 放仓库根而非 `submit/` 子目录 | Docker 构建上下文不支持 `../`；根目录构建最干净，路径引用直观 | 待验证（需在赛事方机器复现） |
| 5/3 | 训练入口保留单一 `scripts/train.py`，通过 yaml 切换 backbone/loss | 降低 CLI 参数复杂度；对比实验只需换 config 文件 | 待验证 |

## 进一步优化思考

- Docker 构建层缓存：先 `COPY submit/requirements.txt` 再 `pip install`，再 `COPY src/`，源码变动不触发 pip 重装
- WandB 离线模式（`mode: offline`）可以在无外网的赛事机器上用，训练完 `wandb sync` 再上传
- 如果 SSH 也被限，考虑 Cloudflare WARP 或 GitHub CLI HTTPS API（token 认证，走 api.github.com 子域名）


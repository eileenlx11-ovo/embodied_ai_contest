# Coreview - docker_submission - 2026-06-02

Scope: embodied_ai_contest Docker 提交包 (`embodied_ai_submission.tar.gz` 解包内容 + 提交 CSV 格式)。
Peer: codex 同时评审 `tech_report` 范围 (互补，无文件重叠)。
Mode: attended 最终检查。

## Review - Claude - 2026-06-02

### Critical
- 无。

### Important
- 无阻塞项。

### Minor
- [ ] **scripts/validate_csv.py:19-30** 只校验「2 列 + 4 位类别」，未校验 `.JPEG` 后缀与总行数=100000。实际提交 CSV 已手工确认合规，故非阻塞；可加 `\.JPEG$` 与行数断言增强。
- [ ] **configs/*.yaml** tar 包含全部历史配置 + 2 个 `.yaml.bak`。无害（run.sh 只用 full_mixcut），保留以防漏依赖。
- [ ] **submit/SUBMISSION.md §6** build 命令用示例 `DATA_ROOT=/data/imagenet`；需用户确认赛方实际挂载路径，否则为通用说明。

### Verified (static + format)
- [x] 全部 src/ + scripts/ `.py` `python -m py_compile` 通过，无断 import。
- [x] src 模块完整 (31 .py，__init__ 齐全，6 子包)。
- [x] **resnet.py:6** `resnet50(weights=None)` → 从零训练，无联网下载预训练权重 (符合规则 + 离线复现安全)。
- [x] **base_trainer.py:129** `wandb.init()` 受 `cfg.logging.wandb.enabled`(默认 False) 门控；full_mixcut 配置无 wandb 段 → 永不触发联网。
- [x] timm 仅 starnet/convnext 用到，非提交路径；resnet50 走 torchvision。
- [x] **run.sh:17** sed `^  root:` 在配置中唯一匹配 (第 6 行)，DATA_ROOT 覆盖不误伤。
- [x] run.sh 流水线自洽：train 写 ./checkpoints_full_mixcut/best.pth → eval/predict 读同路径 (Dockerfile 不 COPY checkpoint，符合可复现训练)。
- [x] CSV 格式合规：`ILSVRC2012_test_00000001.JPEG,0111` — 保 .JPEG、类别 4 位补零、100000 行、无表头。
- [x] tarball md5 本地↔服务器一致 (d44a3daa...)，传输无损。

### Verification commands
- `python -m py_compile $(find src scripts -name '*.py')` → OK
- server `head/tail/wc -l submit/full_mixcut_hflip.csv` → 100000 rows, format OK
- `md5sum` 本地 vs 服务器 → 一致

### Gate
- Status: **Approved for next local batch**
- 静态分析 + 格式校验全过，无 Critical。唯一未验证项：实际 `docker build`（本地/服务器均无 docker）。提交前须在有 docker 的机器上 build 一次。

## Human Audit Checklist
- [ ] 在有 docker 的机器上 `docker build -t embodied-ai .` 确认镜像可构建（纯 build 不需 GPU）。
- [ ] 确认赛方数据集挂载路径，必要时改 SUBMISSION.md §6 的 `DATA_ROOT` 示例。
- [ ] 确认提交内容：`embodied_ai_submission.tar.gz`（源码包）+ `full_mixcut_hflip.csv`（预测结果）。

---

## Co-review - Codex - 2026-06-02

### Accepted
- [x] **submit/run.sh:10-27** Default config/checkpoint now targets final `full_mixcut`; `DATA_ROOT` override generates `configs/_runtime.yaml`, so evaluator-mounted data paths do not require editing the committed YAML.
- [x] **submit/SUBMISSION.md:5-8** Reproduction narrative correctly states single-model HFlip TTA and no model ensemble.
- [x] **reviews/README.md** Docker-submission gate records the remaining build-only caveat clearly.

### Fixed During Co-review
- [x] **submit/run.sh:10-11** Made `CKPT_DIR` overridable together with `CONFIG`, so variant reruns are not trapped on `checkpoints_full_mixcut`.
- [x] **submit/SUBMISSION.md:58-64** Updated the prediction-only example to generate `configs/_runtime.yaml` before prediction; otherwise the command could ignore `DATA_ROOT` and fall back to the AutoDL absolute path in the source config.

### Verification
- Pending rerun after final staging: `py_compile`, `pytest`, YAML/HTML checks, secret scan, and staged-file audit.

### Gate
- Status: **Approved for GitHub commit; docker build still requires a machine with Docker**

# Review: full_codebase — 2026-05-28 [PENDING HUMAN]

Reviewer: Claude Code
Commit: working tree (post dual-review sync)

## Critical

- [ ] **src/data/imagenet_dataset.py:23-25 + ELR interaction** `IndexedDataset` returns `int(idx)` which is the local index into the wrapper (0..len-1). When `Subset` wraps the dataset first (e.g. after exclude_file filtering), then `IndexedDataset` wraps the Subset, the index is 0..len(subset)-1. This is actually correct for ELR (buffer sized to subset length). BUT: `get_indexed_dataloaders()` at L108 wraps AFTER subset, so if both `exclude_file` and `subset_ratio < 1.0` are active, a double-Subset wrapping occurs and ELR buffer size = inner subset size while indices come from outer subset. **Risk**: only triggers if both exclude_file AND subset_ratio are used simultaneously (unlikely in production, possible in debug). → fix: verify index range matches ELR buffer size at init.

## Important

- [ ] **src/losses/elr.py:20-22** `self.target = torch.zeros(num_samples, num_classes)` allocated on CPU by default. When `device="cuda"` is passed, it moves to GPU. For 1.28M × 1000 classes = 5.12 GB float32 — this will OOM on 24GB GPU alongside model+data. The `device="cpu"` default is correct but undocumented; if someone passes `device=cuda` in config it silently OOMs. → fix: add a size check or force CPU for ImageNet-scale.
- [ ] **src/trainers/noisy_trainer.py:30-31** `sample_losses` and `sample_counts` are `torch.zeros(num_samples)` on CPU. With 1.28M samples this is fine (5MB), but `record_batch_losses()` at L50 does `self.sample_losses[indices] += losses.cpu()` — if indices are from a Subset (0..N-1 where N < num_samples), the buffer is oversized. Wastes memory but not a correctness bug.
- [ ] **src/models/starnet.py:78-80** `StarBlock` uses `nn.Conv2d(dim, dim, 1)` for element-wise multiplication (`star_fn`). The forward does `x * self.star_fn(x)` — this is a learned gating mechanism but the 1×1 conv output is not activated before multiplication. If the conv output goes negative, the gate inverts the signal. This may be intentional (paper design) but worth verifying against the StarNet paper.
- [ ] **src/analysis/noise_analysis.py:145-150** `scan_class()` uses `random.sample(files, sample_size)` without seeding. Noise scan results are non-reproducible across runs. → fix: pass a seed or use `np.random.default_rng(seed)`.
- [ ] **scripts/train.py:52-55** CLI `--loss` and `--loss-params` override config values, but if `--loss elr` is passed without `--loss-params`, the default `loss_params={}` means ELR gets no `lambda_` or `beta` — it falls back to ELR's own defaults (lambda=3.0, beta=0.7). This is fine but non-obvious; a user might expect config-file params to be preserved when only `--loss` is overridden.
- [ ] **Dockerfile:1** Base image `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` — PyTorch 2.1 is old (current is 2.4+). The `torch.amp.autocast` API used in code is 2.0+ compatible so no breakage, but `torch.amp.GradScaler(device_type)` signature was added in 2.3. If the container actually uses 2.1, GradScaler init will fail. → verify: does the server's torch version match?

## Minor / Style

- [ ] **src/models/convnext_v2.py:45** `GRN` layer uses `torch.norm(x, p=2, dim=(2,3), keepdim=True)` — this computes L2 norm per channel. For large spatial dims this is fine, but `torch.linalg.vector_norm` is the modern replacement (torch.norm is soft-deprecated).
- [ ] **src/losses/peer_loss.py:13** Comment says "cheaper than original (no second forward pass)" but the permutation approach still has theoretical limitations vs true peer samples. Fine for competition but wouldn't pass peer review (pun intended).
- [ ] **src/analysis/class_stats.py:120** `plot_class_distribution()` imports matplotlib at function level — good for optional dependency, but if called in a headless server without display backend, it may warn. Consider `matplotlib.use('Agg')` before import.
- [ ] **scripts/predict.py:130** `logits_sum` variable name is misleading — it actually accumulates softmax probabilities, not logits. → rename to `probs_sum`.
- [ ] **src/trainers/base_trainer.py:198** Comment "Use original hard targets for accuracy even when mixup is active" — good documentation of intentional design choice.
- [ ] **tests/test_losses.py** — Good coverage for losses but no integration tests for data pipeline (exclude_file, Subset wrapping, IndexedDataset). Consider adding a smoke test.

## Notes

### Architecture Assessment
The codebase is well-structured for a competition project. Clear separation between models/data/losses/trainers, config-driven experiments, and proper checkpoint management.

### Unused/Dead Code
- `src/data/sample_selection.py` — TODO stub (5 lines)
- `src/data/sampler.py` — TODO stub (4 lines)
- `src/utils/visualization.py` — TODO stub (6 lines)
- `src/analysis/` — noise_analysis.py and class_stats.py are analysis tools, not training-critical. They work but won't affect model performance.

### What's Working Well
1. Dual-eval (EMA + raw) with separate checkpoints — solid design
2. Data cleaning pipeline (C1+C2+Merge) with sanity checks — production-ready
3. Loss function library with proper numerical stability
4. Flexible config system with CLI overrides
5. Ensemble + TTA in predict.py — ready for final submission

### Competition-Critical Path
The only files that matter for the final score are:
- `src/trainers/base_trainer.py` (training loop)
- `src/data/imagenet_dataset.py` (data loading + exclude)
- `src/data/transforms.py` (augmentation)
- `scripts/clean_c1_class_bugs.py` + `clean_c2_teacher_loss.py` + `clean_merge.py` (cleaning)
- `configs/imagenet_resnet50_cleaned.yaml` + `_cleaned_mixcut.yaml` (final configs)
- `scripts/predict.py` (submission generation)

Everything else is supporting infrastructure or historical experiments.

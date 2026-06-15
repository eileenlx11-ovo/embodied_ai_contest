# Review: cleaning_dual_eval — 2026-05-28 [CLOSED]

Reviewer: Codex
Commit: working tree

## Critical
- [x] **[src/data/imagenet_dataset.py:45]** `exclude_file` is silently ignored when the path is wrong or missing — a cleaned run can accidentally train on the full noisy dataset while logs still identify it as `resnet50_cleaned_v1` / `resnet50_cleaned_mixcut`, invalidating the final comparison → fix: if `exclude_file` is configured and `not exclude_file.exists()`, raise `FileNotFoundError`; optionally print the absolute resolved path.

## Important
- [x] **[src/trainers/base_trainer.py:385]** `best_ema.pth` and `best_raw.pth` are saved before the global `best_acc` is updated, so their checkpoint metadata can contain a stale `best_acc` even though `acc1` is correct. This can confuse resume/debug scripts that read `best_acc` from those files → fix: either save per-source checkpoints after updating the relevant metadata, or include explicit `source_acc` / `source` fields and avoid relying on `best_acc` there.
- [x] **[src/trainers/base_trainer.py:415]** `latest.pth` is saved with `best_source=None`; `predict.py` will then fall back to EMA if `latest.pth` is ever used for submission or manual evaluation, even if raw was the selected source at the latest validation. The intended path is `best.pth`, but this is an avoidable footgun → fix: persist the latest selected source on the trainer and pass it into `save_checkpoint(..., "latest.pth", best_source=self.latest_selected_source)`.
- [x] **[scripts/clean_c2_teacher_loss.py:148]** The script reports the number excluded after per-class caps, but not the actual global exclusion percentage. Because caps can reduce the target below `top_pct`, operators may assume exactly 10% was removed when the final list is smaller → fix: print `len(high_loss_indices) / num_samples * 100` and optionally write class-level removal stats to CSV.

## Minor / Style
- [x] **[src/trainers/base_trainer.py:217]** `validate()` and `_validate_model()` now duplicate the same evaluation loop, and `fit()` only needs `_validate_model()`. This increases the chance of future metric drift → fix: make `validate()` call `_validate_model(self.ema.ema if self.ema else self.model)` or remove the old method.
- [x] **[scripts/clean_c1_class_bugs.py:33]** The printed class exclusion count uses `len(samples)` from `iterdir()`, while the output only includes files. If a class directory ever contains non-image metadata, the log count will overstate removals → fix: build `files = sorted(s for s in cls_dir.iterdir() if s.is_file())` and use it for both writing and logging.
- [x] **[scripts/clean_merge.py:26]** Merge does not filter blank lines or comments. The dataloader later ignores blank/comment lines, so this is not currently dangerous, but the merged file can contain noise if a source list is manually annotated → fix: use the same `strip()` + `not startswith("#")` filter as `imagenet_dataset.py`.

## Notes
Syntax check passed for the reviewed Python files:

`python -m py_compile src/trainers/base_trainer.py src/data/imagenet_dataset.py src/data/transforms.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py scripts/predict.py scripts/evaluate.py scripts/train.py`

Dual-eval now records `best_source` and `predict.py` / `evaluate.py` load raw vs EMA accordingly. That addresses the main raw-wins-but-inference-loads-EMA risk.

The lack of a validated `mixup+cutmix only` data point is already tracked in `problem_log.md` Week 4; not duplicated here.

---

Reviewer: Codex
Role: Fix implementation
Commit: working tree

## Fixes Applied
- [x] **[src/data/imagenet_dataset.py:47]** Added fail-fast behavior for configured-but-missing `exclude_file`; cleaned configs now cannot silently train on full data.
- [x] **[src/data/imagenet_dataset.py:56]** Changed exclude logging to print the resolved exclude-file path and remaining sample count.
- [x] **[src/trainers/base_trainer.py:218]** De-duplicated validation by making `validate()` delegate to `_validate_model()`.
- [x] **[src/trainers/base_trainer.py:292]** Added `source_acc` checkpoint metadata so `best_raw.pth` / `best_ema.pth` carry their own source-specific accuracy even when global `best_acc` differs.
- [x] **[src/trainers/base_trainer.py:106]** Added `latest_selected_source` tracking and restored it from checkpoint metadata on full resume.
- [x] **[src/trainers/base_trainer.py:404]** Saved `latest.pth` with the current `best_source` instead of `None`.
- [x] **[scripts/clean_c2_teacher_loss.py:148]** Added actual post-cap exclusion percentage to C2 cleaning logs.
- [x] **[scripts/clean_c1_class_bugs.py:33]** Made C1 logging count only files that are actually written to the exclude list.
- [x] **[scripts/clean_merge.py:26]** Made merge filtering match the dataloader: strip lines, skip empty lines, skip `#` comments.

## Verification
- [x] **[local]** `python -m py_compile src/trainers/base_trainer.py src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py scripts/predict.py scripts/evaluate.py scripts/train.py`
- [x] **[server]** `/root/miniconda3/bin/python -m py_compile src/trainers/base_trainer.py src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py scripts/predict.py scripts/evaluate.py scripts/train.py`

## Notes
Synced the fixes to `/root/autodl-tmp/embodied_ai_contest/`.

Server backup for this fix pass:

`/root/autodl-tmp/embodied_ai_contest/.codex_backups/20260528_043423_review_fixes`

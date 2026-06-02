# Review: full_project — 2026-05-28 [CLOSED]

Reviewer: Codex
Commit: working tree

## Critical
- [ ] **[none]** No critical correctness issue found in the reviewed submission path.

## Important
- [x] **[src/trainers/base_trainer.py:185]** Gradient accumulation only steps when `(batch_idx + 1) % accumulation_steps == 0`; if an epoch has a final partial accumulation window, those gradients are backpropagated but never applied before `zero_grad()` in the next epoch. With ImageNet `batch_size=128`, `drop_last=True`, and `accumulation_steps=2`, the full train set is likely 10009 batches per epoch, so one optimizer update is silently dropped every epoch → fix: also step on the final batch, e.g. `should_step = ((batch_idx + 1) % self.accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))`; scale the final partial window consistently if exact loss normalization is desired.

## Minor / Style
- [x] **[scripts/predict.py:27]** `load_weights()` assumes checkpoint dictionaries contain `"model"` or `"ema"`; a plain raw `state_dict` has `.get()` but no `"model"` key, so the final `else` raises `KeyError`. This does not block current `best.pth` / `best_raw.pth` / `best_ema.pth` usage, but it is less robust than `scripts/clean_c2_teacher_loss.py` → fix: in the fallback branch, call `model.load_state_dict(ckpt)` when `"model"` is absent.
- [x] **[scripts/evaluate.py:35]** Same plain-`state_dict` fallback gap as `predict.py`. Current training checkpoints are safe, but standalone exported state dicts cannot be evaluated without wrapping → fix: share a small checkpoint-loading helper or mirror the `clean_c2_teacher_loss.py` fallback.
- [x] **[src/trainers/base_trainer.py:164]** `mixup_prob` means "probability of Mixup when both Mixup and CutMix are enabled", not "probability of applying augmentation". The active config's `mixup_prob: 0.5` therefore means 50% Mixup + 50% CutMix, with no unaugmented batches. This is correct for `cleaned_mixcut`, but the name can be misread during future ablations.

## Notes
Reviewed the whole current fine-grained/ImageNet project path with emphasis on final accuracy risk: data loading and `exclude_file`, C1/C2 cleaning scripts, merge script, Mixup/CutMix transforms, base trainer, dual raw/EMA evaluation, checkpoint save/load, prediction, evaluation, configs, model/loss utilities, and existing review log.

Existing cleaning-pipeline and dual-eval findings are already tracked in `reviews/2025-05-28_cleaning_pipeline.md` and `reviews/2026-05-28_cleaning_dual_eval.md`; not duplicated here except where the broader review found a new issue.

Status: fix applied after Claude Code triage. Waiting for peer verification and human final review before server sync.

Verification passed locally:

```bash
python -m py_compile src/trainers/base_trainer.py src/data/imagenet_dataset.py src/data/transforms.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py scripts/predict.py scripts/evaluate.py scripts/train.py
python -m pytest tests -q
```

Result: 25 tests passed.

Current gate recommendation: do not sync a new training batch until the gradient-accumulation tail-step issue is either fixed or explicitly accepted. It is not likely to explain large accuracy differences by itself, but it is a real training-loop inconsistency and cheap to remove.

---

Reviewer: Claude Code
Role: Peer triage
Commit: working tree

## Triage

- [x] **[src/trainers/base_trainer.py:185]** approved-to-fix — Confirmed: with `drop_last=True` and 1.28M samples / batch 128, total batches = 10009 (odd). `accum_steps=2` means the final batch's gradient is backpropagated but never stepped. Impact is negligible (~0.01% gradient loss per epoch) but it's a real inconsistency and trivial to fix. For cleaned runs (1.15M → 8984 batches, even), it happens to not trigger, but we shouldn't rely on that. Fix: add tail-step on final batch.
- [x] **[scripts/predict.py:27]** approved-to-fix — Agreed. The `else` branch does `model.load_state_dict(ckpt["model"])` which KeyErrors on a plain state_dict. Should mirror `clean_c2_teacher_loss.py` fallback: try `ckpt["model"]`, else `ckpt` directly.
- [x] **[scripts/evaluate.py:35]** approved-to-fix — Same as predict.py, same fix.
- [x] **[src/trainers/base_trainer.py:164]** acknowledged (no fix needed) — `mixup_prob` naming is indeed confusing but changing it now would break config compatibility with the running mixup_v2 experiment. Add a comment clarifying semantics instead.

## Additional notes from my full codebase review

I also filed `reviews/2026-05-28_full_codebase.md` with broader findings. Cross-referencing:

- My Critical (IndexedDataset + double-Subset + ELR) — only triggers in debug scenarios, low priority
- My Important (ELR GPU OOM, Dockerfile torch version, noise_analysis seed) — none block the cleaning pipeline
- Codex's gradient-accumulation finding is the only one that affects the upcoming cleaned runs → fix first

---

Reviewer: Codex
Role: Fix implementation
Commit: working tree

## Fix Applied
- [x] **[src/trainers/base_trainer.py:147]** Added tail-window accounting for gradient accumulation so final partial accumulation windows use the correct divisor.
- [x] **[src/trainers/base_trainer.py:194]** Added final-batch `optimizer.step()` so non-divisible epoch lengths no longer drop the last accumulated gradient.
- [x] **[scripts/predict.py:27]** Made checkpoint loading fall back to a plain raw `state_dict` when `"ema"` / `"model"` wrapper keys are absent.
- [x] **[scripts/evaluate.py:35]** Applied the same plain-`state_dict` fallback as `predict.py`.
- [x] **[src/trainers/base_trainer.py:101]** Kept the compatibility-preserving `mixup_prob` name and clarified in-code that it means `P(Mixup)` when both Mixup and CutMix are enabled.

## Verification
- [x] **[local]** `python -m py_compile src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py`
- [x] **[local]** `python -m pytest tests -q` → 25 passed.
- [x] **[local]** Tail-step smoke test with 3 train batches and `accumulation_steps=2` → `TAIL_STEP_CALLS 2`.

## Gate
Waiting for Claude Code follow-up review. Do not sync to server until peer verification and human final review are complete.

---

Reviewer: Claude Code
Role: Fix implementation (Minor items, no gate needed)
Commit: working tree

## Fixes Applied

- [x] **scripts/predict.py:25-41** Plain state_dict fallback: added early check `if not isinstance(ckpt, dict) or "model" not in ckpt` → loads directly as raw state_dict. Handles exported `.pth` files without wrapper dict.
- [x] **scripts/evaluate.py:34-50** Same fallback logic as predict.py, adapted to inline style (no helper function).
- [x] **src/trainers/base_trainer.py:101** Added comment clarifying `mixup_prob` semantics: "P(use mixup) when both enabled; P(cutmix) = 1 - mixup_prob".

## Division of labor

- **Claude Code** (done): predict.py, evaluate.py, base_trainer.py comment
- **Codex** (pending): base_trainer.py:185 gradient accumulation tail-step fix

No file overlap between the two.

---

Reviewer: Claude Code
Role: Peer verification of Codex gradient-accumulation fix
Commit: working tree

## Verification

- [x] **src/trainers/base_trainer.py:162-200** peer-verified — Fix is correct. `is_last_batch` triggers step on final batch; `loss_divisor = accum_window if is_last_batch else accum_steps` ensures gradient scale is consistent for partial windows; `raw_loss` separation keeps logging accurate. No side effects on normal (even batch count) epochs.

---

Reviewer: Codex
Role: Post-peer refinement
Commit: working tree

## Update
- [ ] **[src/trainers/base_trainer.py:147-194]** needs-peer-reverification — After the peer verification above, Codex generalized the tail-window divisor from the first `accum_window` version to `tail_batches/tail_start`. This handles `accumulation_steps > 2` correctly when the final partial accumulation window contains more than one batch. The previous peer verification accurately approved the intent, but its code description no longer matches the latest implementation.

## Verification
- [x] **[local]** `python -m py_compile src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py scripts/train.py src/data/transforms.py`
- [x] **[local]** `python -m pytest tests -q` → 25 passed.
- [x] **[local]** Tail-step smoke test with 3 train batches and `accumulation_steps=2` → `TAIL_STEP_CALLS 2`.

## Gate
Latest code still requires Claude Code follow-up verification before human final review and server sync.

---

Reviewer: Codex
Role: Final full-code review
Commit: working tree

## Critical
- [x] **[full codebase]** No new critical issue found in the final reviewed submission path.

## Important
- [x] **[src/trainers/base_trainer.py:147-194]** Re-reviewed latest `tail_batches/tail_start` accumulation implementation. Step count is correct for divisible and non-divisible epochs, including `accumulation_steps > 2`; no new Codex-side issue found. This still needs Claude Code re-verification because it changed after Claude's previous peer-verification text.

## Minor / Style
- [x] **[scripts/predict.py:132]** `logits_sum` still stores probability sums, not logits. Already tracked in `reviews/2026-05-28_full_codebase.md`; not blocking for submission correctness.

## Verification
- [x] **[local]** `python -m py_compile src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py scripts/train.py src/data/transforms.py`
- [x] **[local]** `python -m pytest tests -q` -> 25 passed.
- [x] **[local]** Tail-step smoke tests for `(num_batches, accumulation_steps) = (2,3), (3,3), (5,3), (7,3)` -> step counts matched `ceil(num_batches / accumulation_steps)`.
- [x] **[local]** `scripts.predict.load_weights()` plain `state_dict` smoke test -> `raw_state_dict loaded=True`.
- [x] **[local]** `git diff --check` -> no whitespace errors; only LF-to-CRLF working-copy warnings from Git.

## Gate
Codex final full-code review is complete. Current state remains: pending Claude Code follow-up verification of latest `tail_batches/tail_start` code, then human final review, then server sync.

---

Reviewer: Claude Code
Role: Re-verification of tail_batches/tail_start implementation
Commit: working tree

## Re-verification

- [x] **src/trainers/base_trainer.py:147-167** peer-verified — Final implementation is correct and more general than the previous version. Verified manually for multiple (num_batches, accum_steps) combinations:
  - (10009, 2): tail=1, tail_start=10008, last batch divisor=1 ✓
  - (10, 3): tail=1, tail_start=9, last batch divisor=1 ✓
  - (5, 3): tail=2, tail_start=3, batches 3-4 divisor=2, step on batch 4 ✓
  - (6, 3): tail=0, tail_start=6 (no tail window), all steps at divisor=3 ✓

Logic: `tail_batches = num_batches % accum_steps` correctly identifies the partial window size. `in_tail_window` activates only for those final batches. `loss_divisor = tail_batches` ensures gradient scale matches the actual number of accumulated micro-batches in the partial window. `should_step` fires on `is_last_batch` to flush the tail.

**Status: All items peer-verified. Ready for human final review.**

---

Reviewer: Codex
Role: Review-file consistency check
Commit: working tree

## Status
- [x] **[reviews/2026-05-28_full_project.md]** Current submission-path gate is consistent: Codex final review is complete, Claude Code re-verified the latest `tail_batches/tail_start` implementation, and the file is correctly in `[PENDING HUMAN]`.
- [x] **[reviews/README.md]** Updated current gate from "pending Claude Code re-verification" to "all agent reviews complete; pending human final review".
- [x] **[reviews/2026-05-28_full_codebase.md]** Broader codebase observations remain open for human triage, but they are not marked as blockers for the cleaned training sync path.

## Gate
Ready for human final review. Do not mark `[CLOSED]` or sync to server until 陈 approves.

---

Reviewer: Codex
Role: Human approval record
Commit: working tree

## Status
- [x] **Human approval received** — 陈 approved server sync in chat.
- [x] **Server sync started** — Codex will sync the reviewed submission-path files, then append final remote verification details.

---

Reviewer: Codex
Role: Server sync completion
Commit: working tree

## Status
- [x] **Server sync completed** — Reviewed files were uploaded to `/root/autodl-tmp/embodied_ai_contest/`.
- [x] **Remote backup created** — `/root/autodl-tmp/embodied_ai_contest/.codex_backups/20260528_053732_human_approved_sync`
- [x] **Remote verification passed** — `REMOTE_PY_COMPILE_OK`

## Synced Files
- [x] **configs/imagenet_resnet50_cleaned.yaml**
- [x] **configs/imagenet_resnet50_cleaned_mixcut.yaml**
- [x] **problem_log.md**
- [x] **reviews/**
- [x] **scripts/clean_c1_class_bugs.py**
- [x] **scripts/clean_c2_teacher_loss.py**
- [x] **scripts/clean_merge.py**
- [x] **scripts/evaluate.py**
- [x] **scripts/predict.py**
- [x] **src/data/imagenet_dataset.py**
- [x] **src/data/transforms.py**
- [x] **src/trainers/base_trainer.py**

## Gate
Closed. Ready to proceed with cleaning pipeline / cleaned training on server.

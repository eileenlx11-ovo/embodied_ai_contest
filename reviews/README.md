# Code Review Issues

Shared issue tracker for code reviews performed by **Claude Code** and **Codex (GPT)**.

## Current Review Gate

**Scope:** tech_report

**Status:** pending human review; Codex approved for local-to-server sync

**Review file:** `2026-06-02_tech_report.md`

**Reviewed by:** Codex

**Remote sync:** not run

**Server backup:** pending

**Local verification passed for current tech-report scope:**

```bash
python -c 'from html.parser import HTMLParser; from pathlib import Path; import yaml; h=Path("submit/tech_report/answer_deck.html"); s=h.read_text(encoding="utf-8"); HTMLParser().feed(s); imgs=["../../docs/class_distribution_full.png", "../../docs/week2_loss_curves.png"]; print("html_parse_ok=True"); print("slide_count=", s.count("class=\"slide")); print("missing_images=", [i for i in imgs if not (h.parent/i).resolve().exists()]); yaml.safe_load(Path("configs/imagenet_resnet50_full_mixcut.yaml").read_text(encoding="utf-8")); print("yaml_parse_ok=True")'
rg -n "TBD|训练中|预期 78|多模型 softmax|baseline_v1 \+|checkpoints_mixup_v2|全增广版本|完整数据增广 recipe|mixup_v2 训练中|result\.csv" submit\tech_report configs\imagenet_resnet50_full_mixcut.yaml
rg -n "77\.99|77\.57|77\.53|76\.84|checkpoints_full_mixcut/best\.pth|submit/full_mixcut_hflip\.csv|configs/imagenet_resnet50_full_mixcut\.yaml|best_source=raw|不属于多模型集成|no ensemble" submit\tech_report configs\imagenet_resnet50_full_mixcut.yaml
```

**Current tech-report gate note:** technical report, PPT outline, HTML deck, and final full_mixcut config are locally consistent and ready for human review. Server sync should wait until Docker packaging is also confirmed.

---

**Scope:** docker_submission

**Status:** Approved for next local batch (no Critical); pending human `docker build` confirmation

**Review file:** `2026-06-02_docker_submission.md`

**Reviewed by:** Claude + Codex co-review

**Docker-submission gate note:** `embodied_ai_submission.tar.gz` 解包内容静态分析 + 提交 CSV 格式校验全过。已确认：resnet50 从零训练无联网、wandb 受 config 门控不触发、run.sh sed 唯一匹配、流水线路径自洽、CSV 合规（.JPEG/4位补零/100000行/无表头）、tarball md5 本地↔服务器一致。Codex 复审时补齐了 `CKPT_DIR` 覆盖和 prediction-only 示例的 `_runtime.yaml` 生成。唯一未验证：实际 `docker build`（本地与服务器均无 docker），提交前须在有 docker 的机器上构建一次。

**Previous code review gate:** `2026-05-28_full_project.md` was closed and synced to server. Server backup: `/root/autodl-tmp/embodied_ai_contest/.codex_backups/20260528_053732_human_approved_sync`.

**Previous remote verification passed:**

```bash
/root/miniconda3/bin/python -m py_compile src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py scripts/train.py src/data/transforms.py
```

**Previous local verification passed for code:**

```bash
python -m py_compile src/data/imagenet_dataset.py scripts/clean_c1_class_bugs.py scripts/clean_c2_teacher_loss.py scripts/clean_merge.py src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py scripts/train.py src/data/transforms.py
python -m pytest tests -q
python -m py_compile src/trainers/base_trainer.py scripts/predict.py scripts/evaluate.py
python -m pytest tests -q
# Tail-step smoke test: 3 train batches, accumulation_steps=2 -> TAIL_STEP_CALLS 2
# Tail-step smoke tests: (2,3), (3,3), (5,3), (7,3) -> step counts matched ceil(n/accum)
# predict.load_weights plain state_dict smoke test -> raw_state_dict loaded=True
```

**Operational note:** submit/evaluate final models from `best.pth`, `best_raw.pth`, or `best_ema.pth`; `latest.pth` is a resume checkpoint, not the canonical best checkpoint.

**Scope note:** `2026-05-28_full_project.md` is the previous submission-path code gate. `2026-05-28_full_codebase.md` contains broader non-blocking backlog observations and remains for human triage; those items do not block the final documentation sync unless 陈 chooses to promote one.

## File Naming & Status

Each review produces one Markdown file: `YYYY-MM-DD_<scope>.md`

Title line must include status tag:

```text
# Review: <scope> — <date> [STATUS]
```

Status tags:

| Tag | Meaning |
| --- | --- |
| `[NEW]` | Just created, findings recorded, no fixes yet |
| `[IN PROGRESS]` | Peer triage or fixes underway |
| `[PENDING HUMAN]` | Agent work done, waiting for human (陈) final review |
| `[CLOSED]` | Human approved, synced to server |

- `scope` = module or topic reviewed (e.g. `trainer`, `data_pipeline`, `cutmix_logic`)
- One file per review session; append if same scope is reviewed again the same day

## File Format

```markdown
# Review: <scope> — <date>

Reviewer: Claude Code | Codex
Commit: <short SHA or "working tree">

## Critical

- [ ] **[file:line]** Description — why it matters

## Important

- [ ] **[file:line]** Description

## Minor / Style

- [ ] **[file:line]** Description

## Notes

Free-form observations, architecture suggestions, or questions for the team.
```

## Rules

1. Use checkboxes `- [ ]` so issues can be ticked off in any Markdown editor
2. Severity: Critical > Important > Minor
3. Always include file path + line number (or function name if line is unstable)
4. If a fix is obvious, add `→ fix:` inline suggestion
5. Don't duplicate issues already in `problem_log.md` — reference them with `(see problem_log Week N)`
6. Both reviewers append to the same file if reviewing the same scope on the same day

## Suspected Bug Gate

When a reviewer finds a possible behavioral bug, the fix process depends on severity:

### Three-tier gate

| Severity | Before fix | After fix |
|----------|-----------|-----------|
| **Critical** | Peer agent confirms + **human (陈) confirms** before any code change | Human final review |
| **Important** | Both agents confirm (`approved-to-fix`) before code change | Human final review |
| **Minor / Style** | Agent can fix directly, no peer gate needed | Human final review |

**All tiers require human (陈) final review before sync to server.**

### Process

1. Finding reviewer records issue in review file with severity tag
2. Critical: mark `Status: pending human + peer review` → wait for both
3. Important: mark `Status: pending peer review` → peer triages (`approved-to-fix` / `rejected` / `needs-more-evidence`)
4. Minor: mark `Status: fixed` → apply directly
5. After fix: append `Fix Applied` block with changed files and verification
6. Peer performs follow-up review → `peer-verified` or reopens
7. **Human reviews all changes before server sync**

### Scope

Use this gate for: code bugs, training behavior changes, data pipeline changes, checkpoint semantics, submission/evaluation logic.

Skip gate for: documentation-only edits, typo fixes, review-file bookkeeping, comment additions.

### Triage block format

```markdown
---

Reviewer: Claude Code | Codex
Role: Peer triage
Commit: working tree

## Triage
- [ ] **[file:line]** approved-to-fix | rejected | needs-more-evidence — rationale
```

# Review: tech_report — 2026-06-02 [PENDING HUMAN]

Reviewer: Codex
Commit: bf5fe49 + working tree

## Critical

- [x] **[submit/tech_report/技术方案.md:12]** No Critical issues found. Final narrative explicitly states no pretrained model, no extra data, no multi-model ensemble, and single-model HFlip TTA.

## Important

- [x] **[submit/tech_report/技术方案.md:109]** Repro command points to `configs/imagenet_resnet50_full_mixcut.yaml`; local config is now present and matches the server final config.
- [x] **[submit/tech_report/answer_deck.html:329]** Result table matches the verified final metrics: baseline 77.53, cleaned_v1 76.84, full_mixcut 77.57, HFlip TTA 77.99.
- [x] **[submit/tech_report/answer_deck.html:335]** HFlip TTA is framed as same-checkpoint two-view inference, not multi-model ensemble.
- [x] **[submit/tech_report/PPT_大纲.md:130]** Defense outline uses the same final checkpoint, config, TTA, and CSV path as the technical report.

## Minor / Style

- [x] **[submit/tech_report/answer_deck.html:268]** HTML deck has 10 slides and keyboard navigation; image references resolve locally.
- [x] **[submit/tech_report/技术方案.md:189]** References are concise and relevant to ResNet, Mixup, CutMix, label smoothing, and noisy-label loss comparisons.

## Notes

Reviewed scope:

- `submit/tech_report/技术方案.md`
- `submit/tech_report/PPT_大纲.md`
- `submit/tech_report/answer_deck.html`
- `configs/imagenet_resnet50_full_mixcut.yaml`

No blocking documentation or reproducibility issues found in this scope. The local deliverables consistently use the final route:

```text
full data + ResNet-50 from scratch + CE/LS + Mixup α=0.1 + CutMix α=1.0
+ raw/EMA dual-eval + single-model HFlip TTA
```

Final model/CSV references:

```text
checkpoints_full_mixcut/best.pth
configs/imagenet_resnet50_full_mixcut.yaml
submit/full_mixcut_hflip.csv
```

## Verification - Codex - 2026-06-02

- `python -c '... HTMLParser ... yaml.safe_load ...'`: passed
  - `html_parse_ok=True`
  - `slide_count=10`
  - `missing_images=[]`
  - `yaml_parse_ok=True`
- `rg -n "TBD|训练中|预期 78|多模型 softmax|baseline_v1 \+|checkpoints_mixup_v2|全增广版本|完整数据增广 recipe|mixup_v2 训练中|result\.csv" submit\tech_report configs\imagenet_resnet50_full_mixcut.yaml`: passed, no stale terms found.
- `rg -n "77\.99|77\.57|77\.53|76\.84|checkpoints_full_mixcut/best\.pth|submit/full_mixcut_hflip\.csv|configs/imagenet_resnet50_full_mixcut\.yaml|best_source=raw|不属于多模型集成|no ensemble" submit\tech_report configs\imagenet_resnet50_full_mixcut.yaml`: passed, expected final metrics and paths are present.

## Gate - Codex - 2026-06-02

Status: **Approved for local-to-server sync after human review**

Human audit checklist:

- [ ] Confirm Claude Code's Docker packaging changes are complete before bundling final submission materials.
- [ ] Confirm `configs/imagenet_resnet50_full_mixcut.yaml` should be included in the synced batch.
- [ ] Confirm server backup path before overwriting remote files.
- [ ] After sync, verify the report/deck files exist on server and `configs/imagenet_resnet50_full_mixcut.yaml` matches the final training config.

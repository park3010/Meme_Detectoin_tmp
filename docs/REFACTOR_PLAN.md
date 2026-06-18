# Conservative Refactor Plan

## Inspection Summary

- Baseline model files are tiny and tightly coupled:
  - `module/baselines/image_only_clip.py`
  - `module/baselines/text_only_encoder.py`
  - `module/baselines/clip_text_concat.py`
  - `module/baselines/classifier_heads.py`
- Stage implementation files are more numerous, but several contain non-trivial logic and are covered by stage tests. Moving all Stage A-E internals at once is higher risk.
- Experiment runner files are large enough to remain separate by experiment type.
- Utility files are small but conceptually focused; they should remain separate.

## Refactor Actions

1. Consolidate baseline classifiers into `module/baselines/models.py`.
2. Keep old baseline import paths as compatibility wrappers.
3. Add canonical stage alias modules:
   - `module/stage_a/extractor.py`
   - `module/stage_b/acquisition.py`
   - `module/stage_c/verifier.py`
   - `module/stage_d/fusion.py`
   - `module/stage_e/interpretation.py`
4. Keep old stage files intact to minimize behavior risk.
5. Add tqdm progress helpers and pipeline component tracing.
6. Add execution-flow documentation.

## Entry Points To Preserve

- Baselines: `scripts/run_baseline_text_only.py`, `scripts/run_baseline_image_only.py`, `scripts/run_baseline_clip_concat.py`
- Ours Full: `scripts/run_ours_full.py`
- Ablations: `scripts/run_ablation.py`
- Knowledge comparison: `scripts/run_knowledge_comparison.py`
- Runtime/cost: `scripts/run_runtime_cost.py`
- Shell runners: `scripts/run_smoke_experiments.sh`, `scripts/run_exp_phase*.sh`, `scripts/run_core_*.sh`

# Framework Execution Flow

This project keeps a 5-stage harmful meme interpretation pipeline while exposing separate experiment scripts for baselines, Ours Full, ablations, knowledge comparison, analysis, and shell orchestration.

## Main Training Entry Points

### Text Baseline

`scripts/run_baseline_text_only.py`
→ `experiments/train_baseline.py`
→ `module/baselines/models.py`
→ `module/backbones/text_encoder_wrapper.py`
→ `experiments/metrics.py`
→ `result/predictions/{dataset}/text_only_encoder/{seed}/`

`scripts/run_baseline_image_only.py` and `scripts/run_baseline_clip_concat.py` reuse the same CLI helper and training runner.

### Ours Full

`scripts/run_ours_full.py`
→ `experiments/train_ours.py`
→ `module/pipeline/model.py`
→ Stage A: `module/stage_a/extractor.py`
→ Stage B: `module/stage_b/acquisition.py`
→ Stage C: `module/stage_c/verifier.py`
→ Stage D: `module/stage_d/fusion.py`
→ Stage E: `module/stage_e/interpretation.py`
→ `module/losses/structured_losses.py`
→ `experiments/structured_eval.py`
→ `result/predictions/{dataset}/ours_full/{seed}/`

### Ablation

`scripts/run_ablation.py`
→ `experiments/ablation_runner.py`
→ `module/pipeline/model.py`
→ stage-output transformations
→ `experiments/prediction_io.py`
→ `result/predictions/{dataset}/ablation_{name}/{seed}/`

### Knowledge Comparison

`scripts/run_knowledge_comparison.py`
→ `experiments/knowledge_comparison.py`
→ `experiments/ablation_runner.py`
→ Stage B / Stage C mode changes
→ `result/predictions/{dataset}/knowledge_{mode}/{seed}/`
→ `result/analysis/knowledge_comparison_examples.jsonl`

### Runtime/Cost

`scripts/run_runtime_cost.py`
→ `experiments/runtime_cost.py`
→ manual Stage A-E timing
→ `result/metrics/runtime_cost.csv`
→ `result/analysis/runtime_cost_details.json`

## Shell Runner Flow

- `scripts/run_smoke_experiments.sh`: dataset stats, splits, two baselines, Ours Full, one ablation, one knowledge mode, runtime.
- `scripts/run_exp_phase1.sh`: dataset stats, splits, all simple baselines, aggregate main metrics.
- `scripts/run_exp_phase2.sh`: Ours Full, structured eval, ablations, knowledge comparison, fusion comparison, structured aggregation.
- `scripts/run_exp_phase3.sh`: cross-domain, verifier eval, subset analysis, error cases, rationale eval, runtime/cost, significance, paper tables.
- `scripts/run_core_1seed.sh`: draft one-seed package.
- `scripts/run_core_5seed.sh`: five-seed core package.
- `scripts/run_all_experiments.sh`: Phase 1 → Phase 2 → Phase 3.

Shell runners print timestamps and elapsed seconds for each major Python command.

## Outputs

- Predictions: `result/predictions/{dataset}/{model}/{seed}/final_predictions.jsonl`
- Metrics: `result/metrics/`
- Checkpoints/logs: `result/predictions/{dataset}/{model}/{seed}/best_model.pt`, `last_model.pt`, `training_log.json`, `training_log.csv`
- Cross-domain predictions: `result/predictions_cross_domain/`
- Analysis artifacts: `result/analysis/`
- Paper tables: `result/paper_tables/`

## Compatibility Wrappers

Baseline implementations are consolidated in `module/baselines/models.py`.

Old import paths remain valid:

- `module.baselines.image_only_clip`
- `module.baselines.text_only_encoder`
- `module.baselines.clip_text_concat`
- `module.baselines.classifier_heads`

Canonical stage import modules were added for readability:

- `module.stage_a.extractor`
- `module.stage_b.acquisition`
- `module.stage_c.verifier`
- `module.stage_d.fusion`
- `module.stage_e.interpretation`

The original stage implementation files remain in place because they contain non-trivial, separately tested logic.

## Debugging Components

Use `--print-components` with:

```bash
python scripts/run_ours_full.py --dataset harm_c --seed 42 --epochs 1 --limit 20 --print-components
python scripts/run_ablation.py --dataset harm_c --seed 42 --ablation w_o_verifier --limit 20 --print-components
python scripts/run_knowledge_comparison.py --dataset harm_c --seed 42 --mode verified --limit 20 --print-components
python scripts/run_runtime_cost.py --dataset harm_c --limit 20 --print-components
```

Use `--disable-tqdm` when logs should be plain text:

```bash
python scripts/run_ours_full.py --dataset harm_c --seed 42 --epochs 1 --limit 20 --disable-tqdm
```

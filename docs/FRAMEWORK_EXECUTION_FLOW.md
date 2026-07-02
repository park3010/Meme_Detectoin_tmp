# Framework Execution Flow

## Unified CLI

The primary command surface is now `scripts/run.py`.

```bash
python scripts/run.py train --dataset harm_c --experiment ours_full --config configs/config.yaml --seed 42 --device cpu
python scripts/run.py baseline --dataset harm_c --baseline text_only_encoder --config configs/config.yaml --seed 42 --device cpu
python scripts/run.py stage --dataset harm_c --until stage_e --limit 5 --config configs/config.yaml --device cpu
python scripts/run.py audit --run-root result/predictions/harm_c/ours_full/42 --write-report --strict --require-nonempty-metrics
```

## Ours Full Flow

```text
scripts/run.py train
→ experiments/train.py::run_ours_experiment
→ module/runner.py::HarmfulMemePipeline
→ module/internal_evidence_extractor.py      # Stage A
→ module/external_knowledge_acquisition.py   # Stage B
→ module/knowledge_filter_verifier.py        # Stage C
→ module/evidence_fusion_reasoning.py        # Stage D
→ module/structured_interpretation_head.py   # Stage E
→ module/losses.py
→ experiments/evaluation.py
→ experiments/prediction_io.py
```

## Baseline Flow

```text
scripts/run.py baseline
→ experiments/train.py::run_baseline_experiment
→ module/baseline.py
→ module/backbone/vision.py and/or module/backbone/text.py
→ experiments/evaluation.py
```

## Stage-Only Flow

```text
scripts/run.py stage
→ module/runner.py::PipelineRunner
→ module/runner.py::HarmfulMemePipeline
→ selected Stage A-E endpoint
→ result/stage_*/ artifacts
```

## Ablation And Knowledge Flow

```text
scripts/run.py ablation
→ experiments/ablation_runner.py
→ module/runner.py::HarmfulMemePipeline
→ stage output transformations
→ result/predictions/{dataset}/ablation_{mode}/{seed}/
```

Knowledge comparison remains a specialized experiment path because it has its own analysis export:

```text
scripts/run.py suite --suite knowledge_1seed
→ experiments/knowledge_comparison.py
→ experiments/ablation_runner.py::run_framework_variant
```

## Protocol Suite Flow

```text
scripts/run.py suite
→ experiments/experiment_suite.py::resolve_suite_plan
→ persisted split file: result/splits/{dataset}/seed_{seed}.json
→ train/evaluate runner for each planned item
→ run manifest: result/predictions/{dataset}/{run_name}/{seed}/run_manifest.json
→ optional pipeline audit
→ suite manifest: result/experiment_suites/{suite_name}/suite_manifest.json
```

Ablation semantics are declared in `experiments/ablation_configs.py` as `AblationContract` objects. The audit reads the run manifest to distinguish full-model checks from ablation-specific expectations such as `w_o_retrieval` requiring zero verified knowledge or `w_o_structured_auxiliary` requiring exactly four active logits losses.

## Outputs

- Stage artifacts: `result/stage_a/` through `result/stage_e/`
- Predictions: `result/predictions/{dataset}/{model}/{seed}/final_predictions.jsonl`
- Metrics: `result/predictions/{dataset}/{model}/{seed}/metrics.json` and `result/metrics/`
- Audit report: `result/predictions/{dataset}/{model}/{seed}/pipeline_audit_report.md`

# Framework Execution Flow

The primary entry point is:

```bash
python scripts/run.py ...
```

## Command Adapter Flow

```text
scripts/run.py
-> scripts/common.py
-> scripts/commands/experiment.py
-> scripts/commands/data.py
-> scripts/commands/report.py
-> scripts/commands/analysis.py
```

The command files are thin adapters. Implementation logic stays in `experiments/` and `module/`.

## Ours Full

```text
python scripts/run.py train
-> experiments/train.py::run_ours_experiment
-> module/runner.py::HarmfulMemePipeline
-> module/internal_evidence_extractor.py       # Stage A
-> module/external_knowledge_acquisition.py    # Stage B
-> module/knowledge_filter_verifier.py         # Stage C
-> module/evidence_fusion_reasoning.py         # Stage D
-> module/structured_interpretation_head.py    # Stage E
-> module/losses.py
-> experiments/evaluation.py
-> experiments/prediction_io.py
```

Useful diagnostic:

```bash
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --print-components
```

`print_pipeline_components()` is implemented in `module/runner.py`.

## Baselines

```text
python scripts/run.py baseline
-> experiments/train.py::run_baseline_experiment
-> module/baseline.py
-> module/backbone/vision.py and/or module/backbone/text.py
-> experiments/evaluation.py
-> experiments/prediction_io.py
```

## Data Preparation

```text
python scripts/run.py data dataset-stats
python scripts/run.py data make-splits
python scripts/run.py data normalize-labels
python scripts/run.py data audit-annotations
python scripts/run.py data inspect-dataset
python scripts/run.py data inspect-labels
-> experiments/data_preparation.py
-> experiments/splits.py
-> dataset/
```

The dataset path convention remains unchanged:

```text
dataset/source
dataset/annotation
dataset/annotation_normalized
```

## Reporting

```text
python scripts/run.py report aggregate
python scripts/run.py report aggregate-structured
python scripts/run.py report evaluate-structured
python scripts/run.py report export-paper-tables
python scripts/run.py report export-intermediate
-> experiments/reporting.py
-> experiments/evaluation.py
```

## Analysis

```text
python scripts/run.py analysis cross-domain
-> experiments/cross_domain.py

python scripts/run.py analysis verifier
python scripts/run.py analysis rationale
-> experiments/posthoc_quality_evaluation.py

python scripts/run.py analysis subset
python scripts/run.py analysis select-error-cases
-> experiments/posthoc_error_analysis.py

python scripts/run.py analysis runtime
-> experiments/runtime_cost.py

python scripts/run.py analysis significance
-> experiments/statistics.py
```

## Suites And Presets

```text
bash scripts/presets/run_core_smoke.sh
bash scripts/presets/run_core_1seed.sh
bash scripts/presets/run_core_5seed.sh
-> python scripts/run.py suite
-> experiments/experiment_suite.py
```

Each suite writes:

```text
result*/experiment_suites/{suite_name}/suite_manifest.json
result*/predictions/{dataset}/{run_name}/{seed}/run_manifest.json
```

## Output Locations

- Stage artifacts: `result*/stage_a/` through `result*/stage_e/`
- Predictions: `result*/predictions/{dataset}/{model}/{seed}/final_predictions.jsonl`
- Metrics: `result*/predictions/{dataset}/{model}/{seed}/metrics.json` and `result*/metrics/`
- Analysis: `result*/analysis/`
- Paper tables: `result*/paper_tables/`
- Pipeline audit: `result*/predictions/{dataset}/{model}/{seed}/pipeline_audit_report.md`

## Removed Wrapper Replacements

The old standalone wrappers were deleted after their behavior was moved into grouped commands:

- `scripts/build_normalized_labels.py` -> `python scripts/run.py data normalize-labels`
- `scripts/audit_annotations.py` -> `python scripts/run.py data audit-annotations`
- `scripts/inspect_dataset.py` -> `python scripts/run.py data inspect-dataset`
- `scripts/inspect_normalized_labels.py` -> `python scripts/run.py data inspect-labels`
- `scripts/export_intermediate_results.py` -> `python scripts/run.py report export-intermediate`

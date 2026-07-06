# Code Organization

This repository now has one primary Python command surface:

```bash
python scripts/run.py ...
```

The cleanup pass removed duplicate single-purpose Python wrappers and old phase shell runners. The model architecture, dataset paths, normalized-label behavior, split policy, metrics, manifest schemas, and artifact paths were not changed.

## Active Layout

```text
experiments/
├── train.py                         # Ours Full and baseline training/evaluation
├── data_preparation.py              # stats, annotation audit, normalized labels
├── reporting.py                     # metric aggregation, tables, intermediate manifest
├── statistics.py                    # significance tests
├── posthoc_error_analysis.py        # subsets and TP/TN/FP/FN case selection
├── posthoc_quality_evaluation.py    # rationale and verifier analysis
├── evaluation.py                    # harmfulness and structured metrics
├── experiment_suite.py              # reproducible suite runner
├── ablation_runner.py               # ablation, fusion, and framework variants
├── knowledge_comparison.py          # knowledge-source comparison
├── runtime_cost.py                  # runtime and cost profiling
├── preflight.py                     # Experiment 0 readiness gate
├── pretrained_assets.py             # local pretrained asset verification
├── prediction_io.py                 # JSONL serialization
├── run_manifest.py                  # run provenance manifest
├── pipeline_audit.py                # artifact audit contract
└── progress.py                      # shared tqdm/fallback progress API
```

```text
scripts/
├── run.py
├── common.py
├── commands/
│   ├── experiment.py                # train, baseline, stage, assets, evaluate, ablation, audit, suite, preflight
│   ├── data.py                      # dataset-stats, make-splits, normalize-labels, audit-annotations, inspect-*
│   ├── report.py                    # aggregate, structured eval/table exports
│   └── analysis.py                  # cross-domain, verifier, subset, rationale, runtime, significance
└── presets/
    ├── run_preflight.sh
    ├── run_core_smoke.sh
    ├── run_core_1seed.sh
    └── run_core_5seed.sh
```

```text
tests/
├── unit/
│   ├── data/
│   ├── experiment/
│   └── stage/
├── integration/
└── cli/
```

## Canonical Commands

```bash
python scripts/run.py data dataset-stats --dataset all
python scripts/run.py data make-splits --dataset all --all-seeds
python scripts/run.py data normalize-labels --dataset all --disable-tqdm
python scripts/run.py data audit-annotations --dataset all --disable-tqdm
python scripts/run.py data inspect-dataset --dataset harm_c --limit 5
python scripts/run.py data inspect-labels --dataset harm_c --label-set clean --limit 5

python scripts/run.py baseline --dataset harm_c --baseline text_only_encoder --seed 42 --epochs 1 --device cpu
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --print-components
python scripts/run.py stage --dataset harm_c --until stage_e --limit 5 --device cpu

python scripts/run.py report aggregate
python scripts/run.py report aggregate-structured
python scripts/run.py report evaluate-structured --dataset harm_c --model ours_full --seed 42
python scripts/run.py report export-paper-tables
python scripts/run.py report export-intermediate

python scripts/run.py analysis knowledge-comparison --dataset harm_c --mode verified --seed 42
python scripts/run.py analysis runtime --dataset harm_c --limit 20 --device cpu
python scripts/run.py analysis significance
```

## Shell Presets

Shell scripts are now lightweight presets under `scripts/presets/`. They do not implement experiment logic themselves; they call `scripts/run.py` and respect externally supplied `CUDA_VISIBLE_DEVICES`.

```bash
bash scripts/presets/run_preflight.sh
DEVICE=cpu LIMIT=20 bash scripts/presets/run_core_smoke.sh
DEVICE=cuda EPOCHS=5 bash scripts/presets/run_core_1seed.sh
DEVICE=cuda bash scripts/presets/run_core_5seed.sh
```

Each preset has a dedicated default output root:

- `result_preflight`
- `result_core_smoke`
- `result_core_1seed`
- `result_core_5seed`

## Migration Table

| Removed item | Replacement |
| --- | --- |
| `experiments/annotation_normalization.py` | `experiments/data_preparation.py` |
| `experiments/annotation_audit.py` | `experiments/data_preparation.py` |
| `experiments/dataset_stats.py` | `experiments/data_preparation.py` |
| `experiments/aggregate_results.py` | `experiments/reporting.py` |
| `experiments/paper_tables.py` | `experiments/reporting.py` |
| `experiments/significance.py` | `experiments/statistics.py` |
| `experiments/error_case_analysis.py` | `experiments/posthoc_error_analysis.py` |
| `experiments/subset_analysis.py` | `experiments/posthoc_error_analysis.py` |
| `experiments/rationale_eval.py` | `experiments/posthoc_quality_evaluation.py` |
| `experiments/verifier_eval.py` | `experiments/posthoc_quality_evaluation.py` |
| `experiments/components.py` | `module/runner.py::print_pipeline_components` |
| `scripts/build_normalized_labels.py` | `python scripts/run.py data normalize-labels ...` |
| `scripts/audit_annotations.py` | `python scripts/run.py data audit-annotations ...` |
| `scripts/inspect_dataset.py` | `python scripts/run.py data inspect-dataset ...` |
| `scripts/inspect_normalized_labels.py` | `python scripts/run.py data inspect-labels ...` |
| `scripts/export_intermediate_results.py` | `python scripts/run.py report export-intermediate ...` |
| old `scripts/run_exp_phase*.sh` runners | `scripts/presets/run_core_*.sh` and `python scripts/run.py suite ...` |

## Progress And Components

Progress bars are centralized in `experiments/progress.py` and controlled by:

```bash
--disable-tqdm
--tqdm-mininterval 0.5
--tqdm-leave
```

Pipeline component tracing is available through:

```bash
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --print-components
```

The helper lives in `module/runner.py` so it stays close to the canonical `HarmfulMemePipeline`.

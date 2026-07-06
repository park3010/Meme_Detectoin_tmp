# Code Organization And Experiment UX

This repository now uses `python scripts/run.py ...` as the main Python CLI for experiment execution. The older single-purpose Python scripts are retained only where they still provide compatibility or a distinct audit/debug entry point; shell presets route through `scripts/run.py`.

## Scripts Layout

```text
scripts/
├── run.py                         # unified CLI entry point
├── common.py                      # repo-root import bootstrap
├── commands/
│   ├── experiment.py              # core train/baseline/stage/suite commands
│   ├── data.py                    # dataset stats and splits
│   ├── report.py                  # aggregation and exports
│   └── analysis.py                # paper-support analysis runners
├── run_smoke_experiments.sh       # quick CPU/debug preset
├── run_core_1seed.sh              # draft one-seed paper package
├── run_core_5seed.sh              # five-seed core package
├── run_exp_phase1.sh              # stats/splits/baselines
├── run_exp_phase2.sh              # ours, ablations, knowledge, fusion
├── run_exp_phase3.sh              # robustness/analysis/table helpers
├── run_paper_tables_only.sh       # regenerate tables from existing results
└── run_pipeline_audit_smoke.sh    # train plus artifact audit smoke
```

## Experiments Layout

Canonical experiment modules remain separate because they encode different contracts:

```text
experiments/
├── train.py                  # ours and baseline training/evaluation
├── evaluation.py             # harmfulness/structured metrics
├── tactic_decoding.py        # locked formal tactic-rhetorical protocol
├── experiment_suite.py       # suite planning/execution/manifests
├── splits.py                 # split reproducibility
├── prediction_io.py          # prediction serialization
├── run_manifest.py           # provenance contract
├── pipeline_audit.py         # output audit contract
├── preflight.py              # Experiment 0 readiness checks
├── pretrained_assets.py      # local asset readiness
├── progress.py               # canonical tqdm/fallback API
├── ablation_configs.py       # ablation semantic contract
├── ablation_runner.py        # evaluation-time diagnostic ablations/fusion
└── knowledge_comparison.py   # evaluation-time knowledge diagnostics
```

`knowledge_comparison.py` intentionally stays separate from `ablation_runner.py`. Its current manifest status remains an evaluation-time diagnostic unless a future task implements train-time knowledge variants.

## Tests Layout

Most historical tests remain in place to avoid churn in the Stage A-E contracts. New CLI/progress tests are grouped as:

```text
tests/
├── cli/test_cli_contract.py
└── unit/experiment/test_progress.py
```

## Progress Bars

The single progress API is `experiments/progress.py`.

CLI options:

```bash
--disable-tqdm
--tqdm-mininterval 0.5
--tqdm-leave
```

Default behavior uses tqdm auto-detection, so bars are visible in interactive terminals and auto-disabled on non-TTY output. `--disable-tqdm` disables all nested bars. `--tqdm-leave` preserves completed epoch/batch bars for debugging.

Progress hierarchy:

```text
suite bar:  position=0, leave=True
epoch bar:  position=1, leave=False by default
sample/batch bar: position=2, leave=False by default
```

Training bars show lightweight postfix values such as current loss, running loss, learning rate, validation Macro-F1, best metric, and best epoch.

## Old To New Commands

| Old script | Unified command |
| --- | --- |
| `scripts/run_dataset_stats.py` | `python scripts/run.py data dataset-stats ...` |
| `scripts/make_splits.py` | `python scripts/run.py data make-splits ...` |
| `scripts/aggregate_results.py` | `python scripts/run.py report aggregate ...` |
| `scripts/aggregate_structured_results.py` | `python scripts/run.py report aggregate-structured ...` |
| `scripts/evaluate_structured_outputs.py` | `python scripts/run.py report evaluate-structured ...` or `python scripts/run.py evaluate ...` |
| `scripts/export_case_visualization_data.py` | `python scripts/run.py report export-case-data ...` |
| `scripts/export_paper_tables.py` | `python scripts/run.py report export-paper-tables ...` |
| `scripts/run_knowledge_comparison.py` | `python scripts/run.py analysis knowledge-comparison ...` |
| `scripts/run_cross_domain.py` | `python scripts/run.py analysis cross-domain ...` |
| `scripts/run_verifier_eval.py` | `python scripts/run.py analysis verifier ...` |
| `scripts/run_subset_analysis.py` | `python scripts/run.py analysis subset ...` |
| `scripts/select_error_cases.py` | `python scripts/run.py analysis select-error-cases ...` |
| `scripts/run_rationale_eval.py` | `python scripts/run.py analysis rationale ...` |
| `scripts/run_runtime_cost.py` | `python scripts/run.py analysis runtime ...` |
| `scripts/run_significance_tests.py` | `python scripts/run.py analysis significance ...` |
| `scripts/audit_full_pipeline_artifacts.py` | `python scripts/run.py audit ...` |

The duplicate Python wrappers in the left column were physically removed in this pass after the shell presets and docs were migrated. Unique utilities such as annotation audit, normalized-label building, dataset inspection, normalized-label inspection, and intermediate export remain as standalone scripts until they receive full grouped command coverage.

## Example Commands

```bash
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --print-components
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --disable-tqdm
python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --tqdm-mininterval 0.5
python scripts/run.py suite --suite core_1seed --config configs/config.yaml --device cuda --dry-run --output-root result_core_1seed_clean
```

Recommended clean rerun after this refactor:

```bash
DEVICE=cuda OUTPUT_ROOT=result_core_1seed_clean bash scripts/run_core_1seed.sh
```

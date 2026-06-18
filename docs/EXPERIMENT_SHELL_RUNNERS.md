# Experiment Shell Runners

These shell scripts orchestrate the existing Python experiment scripts without changing the model code, dataset paths, or result formats. They are designed for staged execution: start with a smoke run, then run the paper phases or a smaller core package.

## Scripts

- `scripts/run_smoke_experiments.sh`: fast CPU sanity check with `LIMIT=20`, seed 42, and one small pass through stats, splits, baselines, Ours Full, one ablation, one knowledge mode, and runtime profiling.
- `scripts/run_exp_phase1.sh`: dataset statistics, split generation, simple baselines, and main metric aggregation.
- `scripts/run_exp_phase2.sh`: Ours Full, structured evaluation, core ablations, knowledge comparison, fusion comparison, and structured aggregation.
- `scripts/run_exp_phase3.sh`: cross-domain robustness, verifier evaluation, subset analysis, error cases, rationale evaluation, runtime/cost, significance tests, and paper table export.
- `scripts/run_all_experiments.sh`: runs Phase 1, Phase 2, and Phase 3 sequentially.
- `scripts/run_core_1seed.sh`: minimum useful seed-42 package for draft tables.
- `scripts/run_core_5seed.sh`: five-seed core baselines, Ours Full, key ablations, key knowledge comparisons, significance, and table export.
- `scripts/run_paper_tables_only.sh`: regenerates aggregate tables from existing result files.

## Recommended Order

1. Smoke test.
2. Core one-seed run for draft tables.
3. Core five-seed run.
4. Full phase scripts or `run_all_experiments.sh` for complete analysis.

## Example Commands

CPU smoke test:

```bash
bash scripts/run_smoke_experiments.sh
```

Limited debug run:

```bash
LIMIT=50 DEVICE=cpu EPOCHS=1 bash scripts/run_exp_phase2.sh
```

Single GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda EPOCHS=50 PATIENCE=5 MIN_DELTA=0.001 bash scripts/run_exp_phase2.sh
```

One-seed draft package:

```bash
DEVICE=cuda EPOCHS=3 bash scripts/run_core_1seed.sh
```

Five-seed core package:

```bash
DEVICE=cuda EPOCHS=5 bash scripts/run_core_5seed.sh
```

Full run:

```bash
DEVICE=cuda bash scripts/run_all_experiments.sh
```

Regenerate paper tables only:

```bash
bash scripts/run_paper_tables_only.sh
```

## Useful Environment Variables

- `DATASETS`: defaults to `harm_c harm_p facebook memotion`.
- `SEEDS`: defaults to `42 52 123 777 2026`.
- `DEVICE`: defaults to `cuda` for phase/core scripts and `cpu` for smoke.
- `EPOCHS`: default varies by script: smoke `1`, Phase 1 baselines `10`, Phase 2 Ours Full `5`, Phase 3 cross-domain `5`.
- `BATCH_SIZE`: used by baseline scripts. Ours Full currently trains sample-wise and does not accept a batch-size CLI flag.
- `LR`: defaults to `3e-4`.
- `PATIENCE`: defaults to `3`, used by baseline and Ours Full training.
- `MIN_DELTA`: defaults to `0.0`, minimum validation improvement needed to reset patience.
- `EARLY_STOP_METRIC`: defaults to `val_macro_f1`; Ours Full also supports `val_structured_score`.
- `LIMIT`: when non-empty, appended only to Python scripts that support `--limit`.
- `RUNTIME_LIMIT`: defaults to `200`, used only for runtime/cost profiling.
- `ANALYSIS_SEED`: defaults to `42`, used by Phase 3 analysis scripts.
- `RUN_BASELINES`, `RUN_OURS`, `RUN_ABLATIONS`, `RUN_KNOWLEDGE`, `RUN_FUSION`, `RUN_CROSS_DOMAIN`: set to `0` to skip the corresponding block.
- `OUTPUT_ROOT`: defaults to `result`.
- `CONFIG`: defaults to `configs/default.yaml`.
- `PYTHON`: defaults to `python`.

## Output Locations

- `result/dataset_stats/`
- `result/splits/`
- `result/predictions/`
- `result/predictions_cross_domain/`
- `result/metrics/`
- `result/analysis/`
- `result/paper_tables/`

## Notes and Warnings

- Ablation runners are experiment-level ablations over stage outputs unless a Python runner explicitly implements retraining for that ablation.
- Training scripts save `best_model.pt`, `last_model.pt`, and `training_log.json/csv` under `result/predictions/{dataset}/{model}/{seed}/`.
- Cross-domain `--epochs 0` means a frozen/no-training robustness pass. Set `EPOCHS` above zero for lightweight training.
- `LIMIT` is not passed to scripts that do not support `--limit`, such as structured evaluation, aggregation, significance, and paper table export.
- `CUDA_VISIBLE_DEVICES` is intentionally not set inside these scripts. Set it externally when selecting a GPU.
- Full five-seed workflows can be expensive. Prefer `run_smoke_experiments.sh` and `run_core_1seed.sh` before launching a full run.

# Experiment Shell Presets

Shell runners are now small presets in `scripts/presets/`. They exist for convenience only; all Python work flows through the unified CLI.

## Presets

| Preset | Purpose | Default output root |
| --- | --- | --- |
| `scripts/presets/run_preflight.sh` | Run Experiment 0 preflight. | `result_preflight` |
| `scripts/presets/run_core_smoke.sh` | Fast suite smoke check. | `result_core_smoke` |
| `scripts/presets/run_core_1seed.sh` | One-seed draft paper package. | `result_core_1seed` |
| `scripts/presets/run_core_5seed.sh` | Five-seed core package. | `result_core_5seed` |

The removed phase runners (`run_exp_phase1.sh`, `run_exp_phase2.sh`, `run_exp_phase3.sh`, `run_all_experiments.sh`, and `run_paper_tables_only.sh`) are replaced by these presets plus direct `python scripts/run.py ...` commands.

## Examples

```bash
bash scripts/presets/run_preflight.sh
DEVICE=cpu LIMIT=20 bash scripts/presets/run_core_smoke.sh
DEVICE=cuda EPOCHS=5 bash scripts/presets/run_core_1seed.sh
DEVICE=cuda bash scripts/presets/run_core_5seed.sh
```

For a dry run that does not train:

```bash
python scripts/run.py suite --suite core_1seed --config configs/config.yaml --device cuda --dry-run --output-root result_core_1seed_clean
```

## Environment Variables

- `PYTHON`, default `python`
- `CONFIG`, default `configs/config.yaml`
- `DEVICE`, default `cpu` for preflight/smoke and `cuda` for core suites
- `OUTPUT_ROOT`, preset-specific default
- `EPOCHS`, optional override for core suites
- `LIMIT`, optional debug limit for suite presets
- `PROFILE`, default `smoke` for `run_preflight.sh`
- `LABEL_SET`, default `clean` for `run_preflight.sh`

`CUDA_VISIBLE_DEVICES` is respected if set externally. The presets do not set it.

## Direct CLI Replacements

```bash
python scripts/run.py data dataset-stats --dataset all
python scripts/run.py data make-splits --dataset all --all-seeds
python scripts/run.py baseline --dataset harm_c --baseline text_only_encoder --seed 42 --epochs 1 --device cpu
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 20 --device cpu
python scripts/run.py report aggregate
python scripts/run.py report aggregate-structured
python scripts/run.py analysis runtime --dataset harm_c --limit 20 --device cpu
python scripts/run.py report export-paper-tables
```

## Output Locations

- Dataset stats: `result*/dataset_stats/`
- Splits: `result*/splits/`
- Predictions and checkpoints: `result*/predictions/{dataset}/{model}/{seed}/`
- Metrics: `result*/metrics/`
- Analysis exports: `result*/analysis/`
- Paper tables: `result*/paper_tables/`
- Suite manifests: `result*/experiment_suites/{suite_name}/suite_manifest.json`
- Preflight artifacts: `result*/preflight/{profile}/`

Ablation and knowledge-comparison runs remain diagnostic/evaluation-time variants unless a suite or future task explicitly retrains that variant. Verify the intended meaning of `--epochs 0` before treating a run as a trained model result.

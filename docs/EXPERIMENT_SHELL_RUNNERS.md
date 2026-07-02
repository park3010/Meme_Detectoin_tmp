# Experiment Shell Runners

The shell runners orchestrate longer experiment batches and now call the unified CLI where practical.

## Recommended Order

1. Protocol dry run: `python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --dry-run`
2. Protocol smoke run: `python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --disable-tqdm --audit-after-run --strict --require-nonempty-metrics`
3. Legacy shell smoke test: `bash scripts/run_smoke_experiments.sh`
4. Phase 1 baselines: `EPOCHS=10 DEVICE=cuda bash scripts/run_exp_phase1.sh`
5. Phase 2 framework experiments: `EPOCHS=5 DEVICE=cuda bash scripts/run_exp_phase2.sh`
6. Phase 3 analysis: `DEVICE=cuda bash scripts/run_exp_phase3.sh`
7. All phases: `DEVICE=cuda bash scripts/run_all_experiments.sh`

## Useful Environment Variables

- `CONFIG`, default `configs/config.yaml`
- `DATASETS`, default `harm_c harm_p facebook memotion`
- `SEEDS`, default `42 52 123 777 2026`
- `DEVICE`, default depends on script
- `EPOCHS`, `BATCH_SIZE`, `LR`, `PATIENCE`, `MIN_DELTA`
- `LIMIT` for debug-sized runs
- `OUTPUT_ROOT`, default `result`

## Examples

```bash
DEVICE=cpu LIMIT=20 bash scripts/run_smoke_experiments.sh
EPOCHS=5 DEVICE=cuda bash scripts/run_exp_phase2.sh
DEVICE=cuda bash scripts/run_all_experiments.sh
python scripts/run.py train --dataset harm_c --seed 42 --epochs 1 --limit 100 --label-set clean --disable-tqdm --device cpu
python scripts/run.py baseline --dataset harm_c --baseline text_only_encoder --seed 42 --epochs 1 --limit 20 --device cpu
python scripts/run.py stage --dataset harm_c --until stage_e --limit 5 --device cpu
python scripts/run.py suite --suite core_1seed --config configs/config.yaml --device cuda --resume
```

## Output Locations

- Dataset stats: `result/dataset_stats/`
- Splits: `result/splits/`
- Predictions and checkpoints: `result/predictions/{dataset}/{model}/{seed}/`
- Metrics: `result/metrics/`
- Analysis exports: `result/analysis/`
- Paper tables: `result/paper_tables/`
- Suite manifests: `result/experiment_suites/{suite_name}/suite_manifest.json`
- Run manifests: `result/predictions/{dataset}/{model}/{seed}/run_manifest.json`

Ablation runners may apply evaluation-time transformations unless retraining is explicitly implemented for a variant. Verify the intended meaning of `--epochs 0` before using it as a no-training robustness pass.
The protocol-locked suite runner treats `w_o_structured_auxiliary` as a train-time loss ablation, while most other stage ablations remain evaluation-time transformations.

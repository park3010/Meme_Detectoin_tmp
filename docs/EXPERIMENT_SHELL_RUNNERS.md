# Experiment Shell Runners

The shell runners orchestrate longer experiment batches and now call the unified CLI where practical.

## Recommended Order

1. Experiment 0 smoke preflight: `python scripts/run.py preflight --profile smoke --config configs/config.yaml --dataset harm_c harm_p facebook memotion --seed 42 --label-set clean --device cpu --write-report`
2. Initialize local asset layout: `python scripts/run.py assets init-layout --config configs/config.yaml`
3. Place compatible local pretrained assets under `assets/pretrained/`
4. Verify assets: `python scripts/run.py assets verify --config configs/config.yaml --profile main_experiment --write-manifests --strict`
5. Experiment 0 main preflight: `python scripts/run.py preflight --profile main_experiment --config configs/config.yaml --dataset harm_c harm_p facebook memotion --seed 42 --label-set clean --device cpu --write-report --strict`
6. Protocol dry run: `python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --dry-run`
7. Protocol smoke run: `python scripts/run.py suite --suite core_smoke --config configs/config.yaml --device cpu --disable-tqdm --audit-after-run --strict --require-nonempty-metrics`
8. Legacy shell smoke test: `bash scripts/run_smoke_experiments.sh`
9. Phase 1 baselines: `EPOCHS=10 DEVICE=cuda bash scripts/run_exp_phase1.sh`
10. Phase 2 framework experiments: `EPOCHS=5 DEVICE=cuda bash scripts/run_exp_phase2.sh`
11. Phase 3 analysis: `DEVICE=cuda bash scripts/run_exp_phase3.sh`
12. All phases: `DEVICE=cuda bash scripts/run_all_experiments.sh`

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
python scripts/run.py preflight --profile smoke --config configs/config.yaml --dataset harm_c --seed 42 --label-set clean --device cpu --write-report
```

## Output Locations

- Dataset stats: `result/dataset_stats/`
- Splits: `result/splits/`
- Predictions and checkpoints: `result/predictions/{dataset}/{model}/{seed}/`
- Formal tactic decoding artifacts: `result/predictions/{dataset}/{model}/{seed}/tactic_rhetorical_decoding.json`
- Metrics: `result/metrics/`
- Analysis exports: `result/analysis/`
- Paper tables: `result/paper_tables/`
- Suite manifests: `result/experiment_suites/{suite_name}/suite_manifest.json`
- Run manifests: `result/predictions/{dataset}/{model}/{seed}/run_manifest.json`
- Preflight artifacts: `result/preflight/{profile}/`
- Pretrained asset audit: `result/preflight/{profile}/pretrained_asset_audit.json`

Ablation runners may apply evaluation-time transformations unless retraining is explicitly implemented for a variant. Verify the intended meaning of `--epochs 0` before using it as a no-training robustness pass.
The protocol-locked suite runner treats `w_o_structured_auxiliary` as a train-time loss ablation, while most other stage ablations remain evaluation-time transformations.

Main experiment suites should be launched only after `main_experiment` strict preflight passes. In the default offline configuration, fallback encoders are acceptable for smoke checks but block paper-quality main experiments until local pretrained assets are configured.
Model weights under `assets/pretrained/` must not be committed. Track only `.gitkeep` placeholders and `asset_manifest.json`.
Strict vision verification checks that the local checkpoint is compatible with the configured OpenCLIP architecture. File existence and SHA-256 alone are not enough; random initialization, broad missing keys, shape mismatches, and fallback embeddings are never paper-valid pretrained states.

Formal `tactic_rhetorical` metrics are computed from trainable tactic logits only. The validation split selects a sigmoid threshold once, the fixed threshold is applied to the test split, and rendered top-1/heuristic tactic labels remain explanation-only legacy diagnostics.
# Update: Unified CLI

Shell presets now route Python work through `python scripts/run.py ...`. The grouped replacements are:

- `data dataset-stats`, `data make-splits`
- `report aggregate`, `report aggregate-structured`, `report export-paper-tables`
- `analysis knowledge-comparison`, `analysis runtime`, `analysis verifier`, `analysis cross-domain`, `analysis subset`, `analysis rationale`, `analysis significance`

Progress bars use the shared options `--disable-tqdm`, `--tqdm-mininterval`, and `--tqdm-leave`. See `docs/CODE_ORGANIZATION.md` for the full old-to-new command table.

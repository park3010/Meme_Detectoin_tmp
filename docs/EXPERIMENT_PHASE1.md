# Experiment Phase 1: Baseline Infrastructure

This phase adds paper-experiment infrastructure without changing the core
five-stage interpretation framework.

## Added Components

- `experiments/dataset_stats.py`: dataset statistics and annotation coverage.
- `experiments/splits.py`: official split loading and deterministic stratified
  split generation.
- `experiments/metrics.py`: harmfulness metrics for Table 2 drafts.
- `experiments/train_baseline.py`: reusable baseline training/evaluation runner.
- `experiments/aggregate_results.py`: per-seed and mean/std CSV aggregation.
- `module/baselines/`: image-only, text-only, and image+text concat baseline
  classifiers.
- CLI scripts under `scripts/` for stats, splits, baseline runs, and aggregation.

## Table 1: Dataset Statistics

```bash
python scripts/run_dataset_stats.py --dataset all
```

Outputs:

```text
result/dataset_stats/dataset_statistics.json
result/dataset_stats/dataset_statistics.csv
```

## Splits

Official `txt/train.jsonl`, `txt/val.jsonl`, and `txt/test.jsonl` files are used
when available. Otherwise deterministic stratified splits are generated.

```bash
python scripts/make_splits.py --dataset all --seed 42
python scripts/make_splits.py --dataset all --all-seeds
```

Outputs:

```text
result/splits/{dataset}/seed_{seed}.json
```

## Table 2 Draft: Simple Baselines

```bash
python scripts/run_baseline_image_only.py --dataset harm_c --seed 42 --epochs 10
python scripts/run_baseline_text_only.py --dataset harm_c --seed 42 --epochs 10
python scripts/run_baseline_clip_concat.py --dataset harm_c --seed 42 --epochs 10
```

Use all five paper seeds:

```bash
python scripts/run_baseline_text_only.py --dataset harm_c --all-seeds --epochs 10
```

Outputs:

```text
result/predictions/{dataset}/{model}/{seed}/final_predictions.jsonl
result/predictions/{dataset}/{model}/{seed}/metrics.json
result/predictions/{dataset}/{model}/{seed}/metrics.csv
result/predictions/{dataset}/{model}/{seed}/best_model.pt
```

## Aggregate Metrics

```bash
python scripts/aggregate_results.py
```

Outputs:

```text
result/metrics/main_performance.csv
result/metrics/main_performance_mean_std.csv
```

## Fallback Behavior

The baselines reuse the framework backbones:

- image-only uses CLIP/OpenCLIP if configured and available, otherwise fallback
  image features
- text-only uses HuggingFace if configured and available, otherwise hashed text
  embeddings
- concat uses both fallback-safe encoders

This keeps experiments runnable on CPU-only/offline machines.

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
DATASETS="${DATASETS:-harm_c harm_p facebook memotion}"
SEEDS="${SEEDS:-42 52 123 777 2026}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_macro_f1}"
RUN_BASELINES="${RUN_BASELINES:-1}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"

log_section() {
  echo "============================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
  echo "============================================================"
}

run_cmd() {
  local start_ts
  start_ts="$(date +%s)"
  echo "+ $*"
  "$@"
  local end_ts
  end_ts="$(date +%s)"
  echo "Finished in $((end_ts - start_ts))s"
}

run_with_optional_limit() {
  local -a cmd=("$@")
  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  run_cmd "${cmd[@]}"
}

log_section "Phase 1: dataset statistics"
run_cmd "${PYTHON}" scripts/run_dataset_stats.py --config "${CONFIG}" --dataset all --output-root "${OUTPUT_ROOT}/dataset_stats"

log_section "Phase 1: split generation for all paper seeds"
run_with_optional_limit "${PYTHON}" scripts/make_splits.py \
  --config "${CONFIG}" --dataset all --all-seeds --output-root "${OUTPUT_ROOT}/splits"

if [[ "${RUN_BASELINES}" == "1" ]]; then
  for seed in "${SEED_ARRAY[@]}"; do
    for dataset in "${DATASET_ARRAY[@]}"; do
      log_section "Phase 1: image-only baseline dataset=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run_baseline_image_only.py \
        --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
        --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
        --early-stop-metric "${EARLY_STOP_METRIC}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

      log_section "Phase 1: text-only baseline dataset=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run_baseline_text_only.py \
        --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
        --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
        --early-stop-metric "${EARLY_STOP_METRIC}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

      log_section "Phase 1: CLIP+text concat baseline dataset=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run_baseline_clip_concat.py \
        --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
        --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
        --early-stop-metric "${EARLY_STOP_METRIC}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
    done
  done
else
  log_section "Phase 1: RUN_BASELINES=0, skipping baselines"
fi

log_section "Phase 1: aggregate main metrics"
run_cmd "${PYTHON}" scripts/aggregate_results.py --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"

log_section "Experiment Phase 1 complete"

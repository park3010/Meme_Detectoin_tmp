#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
DATASETS="${DATASETS:-harm_c harm_p facebook memotion}"
SEEDS="${SEEDS:-42 52 123 777 2026}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-3e-4}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_macro_f1}"
LIMIT="${LIMIT:-20}"
RUNTIME_LIMIT="${RUNTIME_LIMIT:-20}"

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

log_section "Smoke: dataset statistics"
run_cmd "${PYTHON}" scripts/run_dataset_stats.py --config "${CONFIG}" --dataset all --output-root "${OUTPUT_ROOT}/dataset_stats"

log_section "Smoke: split generation seed 42"
run_with_optional_limit "${PYTHON}" scripts/make_splits.py --config "${CONFIG}" --dataset all --seed 42 --output-root "${OUTPUT_ROOT}/splits"

log_section "Smoke: text-only baseline on harm_c"
run_with_optional_limit "${PYTHON}" scripts/run.py baseline --baseline text_only_encoder \
  --config "${CONFIG}" --dataset harm_c --seed 42 --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --lr "${LR}" --patience "${PATIENCE}" \
  --min-delta "${MIN_DELTA}" --early-stop-metric "${EARLY_STOP_METRIC}" \
  --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

log_section "Smoke: CLIP+text concat baseline on harm_c"
run_with_optional_limit "${PYTHON}" scripts/run.py baseline --baseline clip_text_concat \
  --config "${CONFIG}" --dataset harm_c --seed 42 --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" --lr "${LR}" --patience "${PATIENCE}" \
  --min-delta "${MIN_DELTA}" --early-stop-metric "${EARLY_STOP_METRIC}" \
  --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

log_section "Smoke: Ours Full on harm_c"
run_with_optional_limit "${PYTHON}" scripts/run.py train --experiment ours_full \
  --config "${CONFIG}" --dataset harm_c --seed 42 --epochs "${EPOCHS}" \
  --lr "${LR}" --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
  --early-stop-metric "${EARLY_STOP_METRIC}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

log_section "Smoke: structured evaluation for Ours Full"
run_cmd "${PYTHON}" scripts/run.py evaluate \
  --dataset harm_c --model ours_full --seed 42 --result-root "${OUTPUT_ROOT}" --output-root "${OUTPUT_ROOT}/metrics"

log_section "Smoke: one ablation"
run_with_optional_limit "${PYTHON}" scripts/run.py train \
  --config "${CONFIG}" --dataset harm_c --seed 42 --epochs "${EPOCHS}" \
  --ablation-name w_o_verifier --device "${DEVICE}" --output-root "${OUTPUT_ROOT}" \
  --disable-tqdm

log_section "Smoke: one knowledge comparison"
run_with_optional_limit "${PYTHON}" scripts/run_knowledge_comparison.py \
  --config "${CONFIG}" --dataset harm_c --seed 42 --mode verified --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

log_section "Smoke: runtime/cost"
run_cmd "${PYTHON}" scripts/run_runtime_cost.py \
  --config "${CONFIG}" --dataset harm_c --limit "${RUNTIME_LIMIT}" --device "${DEVICE}" --warmup 0 --output-root "${OUTPUT_ROOT}"

log_section "Smoke experiments complete"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
DATASETS="${DATASETS:-harm_c harm_p facebook memotion}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-5}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-${EPOCHS}}"
OURS_EPOCHS="${OURS_EPOCHS:-${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-3e-4}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_macro_f1}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"

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

log_section "Core 1-seed: dataset statistics"
run_cmd "${PYTHON}" scripts/run_dataset_stats.py --config "${CONFIG}" --dataset all --output-root "${OUTPUT_ROOT}/dataset_stats"

log_section "Core 1-seed: split generation seed=${SEED}"
run_with_optional_limit "${PYTHON}" scripts/make_splits.py \
  --config "${CONFIG}" --dataset all --seed "${SEED}" --output-root "${OUTPUT_ROOT}/splits"

for dataset in "${DATASET_ARRAY[@]}"; do
  for baseline in image_only text_only clip_concat; do
    case "${baseline}" in
      image_only) script="scripts/run.py baseline --baseline image_only_clip" ;;
      text_only) script="scripts/run.py baseline --baseline text_only_encoder" ;;
      clip_concat) script="scripts/run.py baseline --baseline clip_text_concat" ;;
    esac
    log_section "Core 1-seed: baseline=${baseline} dataset=${dataset}"
    run_with_optional_limit "${PYTHON}" "${script}" \
      --config "${CONFIG}" --dataset "${dataset}" --seed "${SEED}" \
      --epochs "${BASELINE_EPOCHS}" --batch-size "${BATCH_SIZE}" --lr "${LR}" \
      --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" \
      --early-stop-metric "${EARLY_STOP_METRIC}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
  done

  log_section "Core 1-seed: Ours Full dataset=${dataset}"
  run_with_optional_limit "${PYTHON}" scripts/run.py train --experiment ours_full \
    --config "${CONFIG}" --dataset "${dataset}" --seed "${SEED}" \
    --epochs "${OURS_EPOCHS}" --lr "${LR}" --patience "${PATIENCE}" \
    --min-delta "${MIN_DELTA}" --early-stop-metric "${EARLY_STOP_METRIC}" \
    --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

  log_section "Core 1-seed: structured eval dataset=${dataset}"
  run_cmd "${PYTHON}" scripts/run.py evaluate \
    --dataset "${dataset}" --model ours_full --seed "${SEED}" \
    --result-root "${OUTPUT_ROOT}" --output-root "${OUTPUT_ROOT}/metrics"

  for ablation in w_o_retrieval w_o_verifier w_o_task_aware_gate; do
    log_section "Core 1-seed: ablation=${ablation} dataset=${dataset}"
    run_with_optional_limit "${PYTHON}" scripts/run.py ablation \
      --config "${CONFIG}" --dataset "${dataset}" --seed "${SEED}" \
      --ablation "${ablation}" --output-root "${OUTPUT_ROOT}"
  done

  for mode in no_knowledge retrieved_only verified; do
    log_section "Core 1-seed: knowledge=${mode} dataset=${dataset}"
    run_with_optional_limit "${PYTHON}" scripts/run_knowledge_comparison.py \
      --config "${CONFIG}" --dataset "${dataset}" --seed "${SEED}" \
      --mode "${mode}" --output-root "${OUTPUT_ROOT}"
  done
done

log_section "Core 1-seed: aggregation and paper tables"
run_cmd "${PYTHON}" scripts/aggregate_results.py --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"
run_cmd "${PYTHON}" scripts/aggregate_structured_results.py --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"
run_cmd "${PYTHON}" scripts/export_paper_tables.py --result-root "${OUTPUT_ROOT}"

log_section "Core 1-seed complete"

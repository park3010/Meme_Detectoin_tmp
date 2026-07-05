#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
DATASETS="${DATASETS:-harm_c harm_p facebook memotion}"
SEEDS="${SEEDS:-42 52 123 777 2026}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
PATIENCE="${PATIENCE:-3}"
MIN_DELTA="${MIN_DELTA:-0.0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_macro_f1}"
RUN_OURS="${RUN_OURS:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_KNOWLEDGE="${RUN_KNOWLEDGE:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
ABLATIONS="${ABLATIONS:-w_o_roi w_o_incongruity w_o_retrieval w_o_verifier w_o_task_aware_gate}"
KNOWLEDGE_MODES="${KNOWLEDGE_MODES:-no_knowledge generated_only retrieved_only generated_retrieved verified}"
FUSION_MODES="${FUSION_MODES:-concat_mlp mean_pooling cross_attention shared_gate task_aware_gate task_aware_gate_verified}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a ABLATION_ARRAY <<< "${ABLATIONS}"
read -r -a KNOWLEDGE_ARRAY <<< "${KNOWLEDGE_MODES}"
read -r -a FUSION_ARRAY <<< "${FUSION_MODES}"

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

log_section "Phase 2: using BATCH_SIZE=${BATCH_SIZE} for compatibility; Ours Full currently trains sample-wise and does not accept --batch-size"

for seed in "${SEED_ARRAY[@]}"; do
  for dataset in "${DATASET_ARRAY[@]}"; do
    if [[ "${RUN_OURS}" == "1" ]]; then
      log_section "Phase 2: Ours Full dataset=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run.py train --experiment ours_full \
        --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
        --epochs "${EPOCHS}" --lr "${LR}" --patience "${PATIENCE}" \
        --min-delta "${MIN_DELTA}" --early-stop-metric "${EARLY_STOP_METRIC}" \
        --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
    else
      log_section "Phase 2: RUN_OURS=0, skipping Ours Full dataset=${dataset} seed=${seed}"
    fi

    log_section "Phase 2: structured evaluation dataset=${dataset} seed=${seed}"
    run_cmd "${PYTHON}" scripts/run.py evaluate \
      --dataset "${dataset}" --model ours_full --seed "${seed}" \
      --result-root "${OUTPUT_ROOT}" --output-root "${OUTPUT_ROOT}/metrics"

    if [[ "${RUN_ABLATIONS}" == "1" ]]; then
      for ablation in "${ABLATION_ARRAY[@]}"; do
        log_section "Phase 2: ablation=${ablation} dataset=${dataset} seed=${seed}"
        case "${ablation}" in
          w_o_retrieval|w_o_verifier|w_o_support_verifier|w_o_task_aware_gate|w_o_structured_auxiliary)
            run_with_optional_limit "${PYTHON}" scripts/run.py train \
              --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
              --epochs "${EPOCHS}" --lr "${LR}" --patience "${PATIENCE}" \
              --min-delta "${MIN_DELTA}" --early-stop-metric "${EARLY_STOP_METRIC}" \
              --ablation-name "${ablation}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
            ;;
          *)
            run_with_optional_limit "${PYTHON}" scripts/run.py ablation \
              --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
              --ablation "${ablation}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
            ;;
        esac
      done
    fi

    if [[ "${RUN_KNOWLEDGE}" == "1" ]]; then
      for mode in "${KNOWLEDGE_ARRAY[@]}"; do
        log_section "Phase 2: knowledge=${mode} dataset=${dataset} seed=${seed}"
        run_with_optional_limit "${PYTHON}" scripts/run_knowledge_comparison.py \
          --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
          --mode "${mode}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
      done
    fi

    if [[ "${RUN_FUSION}" == "1" ]]; then
      log_section "Phase 2: fusion comparison dataset=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run.py ablation \
        --config "${CONFIG}" --dataset "${dataset}" --seed "${seed}" \
        --ablation full --fusion-mode "${FUSION_ARRAY[@]}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
    fi
  done
done

log_section "Phase 2: aggregate structured metrics"
run_cmd "${PYTHON}" scripts/aggregate_structured_results.py \
  --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"

log_section "Experiment Phase 2 complete"

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
LR="${LR:-3e-4}"
RUNTIME_LIMIT="${RUNTIME_LIMIT:-200}"
ANALYSIS_SEED="${ANALYSIS_SEED:-42}"
RUN_CROSS_DOMAIN="${RUN_CROSS_DOMAIN:-1}"

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

if [[ "${RUN_CROSS_DOMAIN}" == "1" ]]; then
  for seed in "${SEED_ARRAY[@]}"; do
    log_section "Phase 3: cross-domain mixed_train seed=${seed}"
    run_with_optional_limit "${PYTHON}" scripts/run.py analysis cross-domain \
      --config "${CONFIG}" --setting mixed_train --model ours_full --seed "${seed}" \
      --epochs "${EPOCHS}" --lr "${LR}" --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"

    for dataset in "${DATASET_ARRAY[@]}"; do
      log_section "Phase 3: leave-one-domain-out heldout=${dataset} seed=${seed}"
      run_with_optional_limit "${PYTHON}" scripts/run.py analysis cross-domain \
        --config "${CONFIG}" --setting leave_one_domain_out --heldout "${dataset}" \
        --model ours_full --seed "${seed}" --epochs "${EPOCHS}" --lr "${LR}" \
        --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
    done
  done
else
  log_section "Phase 3: RUN_CROSS_DOMAIN=0, skipping cross-domain robustness"
fi

for dataset in "${DATASET_ARRAY[@]}"; do
  log_section "Phase 3: verifier evaluation dataset=${dataset}"
  run_with_optional_limit "${PYTHON}" scripts/run.py analysis verifier \
    --config "${CONFIG}" --dataset "${dataset}" --seed "${ANALYSIS_SEED}" --output-root "${OUTPUT_ROOT}"

  log_section "Phase 3: runtime/cost dataset=${dataset}"
  run_cmd "${PYTHON}" scripts/run.py analysis runtime \
    --config "${CONFIG}" --dataset "${dataset}" --limit "${RUNTIME_LIMIT}" \
    --device "${DEVICE}" --output-root "${OUTPUT_ROOT}"
done

log_section "Phase 3: difficult subset analysis"
run_cmd "${PYTHON}" scripts/run.py analysis subset \
  --dataset all --model ours_full --seed "${ANALYSIS_SEED}" --result-root "${OUTPUT_ROOT}"

log_section "Phase 3: error case selection"
run_cmd "${PYTHON}" scripts/run.py analysis select-error-cases \
  --dataset all --model ours_full --seed "${ANALYSIS_SEED}" --result-root "${OUTPUT_ROOT}"

log_section "Phase 3: case visualization export"
run_cmd "${PYTHON}" scripts/run.py report export-case-data \
  --dataset all --model ours_full --seed "${ANALYSIS_SEED}" --result-root "${OUTPUT_ROOT}"

log_section "Phase 3: rationale evaluation"
run_cmd "${PYTHON}" scripts/run.py analysis rationale \
  --dataset all --model ours_full --seed "${ANALYSIS_SEED}" --result-root "${OUTPUT_ROOT}"

log_section "Phase 3: significance tests"
run_cmd "${PYTHON}" scripts/run.py analysis significance \
  --result-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/metrics/significance_tests.csv"

log_section "Phase 3: paper table export"
run_cmd "${PYTHON}" scripts/run.py report export-paper-tables --result-root "${OUTPUT_ROOT}"

log_section "Experiment Phase 3 complete"

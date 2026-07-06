#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SEEDS="${SEEDS:-42 52 123 777 2026}"
export DATASETS="${DATASETS:-harm_c harm_p facebook memotion}"
export DEVICE="${DEVICE:-cuda}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LR="${LR:-3e-4}"
export PATIENCE="${PATIENCE:-3}"
export MIN_DELTA="${MIN_DELTA:-0.0}"
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_macro_f1}"
export ABLATIONS="${ABLATIONS:-w_o_retrieval w_o_verifier w_o_task_aware_gate}"
export KNOWLEDGE_MODES="${KNOWLEDGE_MODES:-no_knowledge retrieved_only verified}"
export RUN_FUSION="${RUN_FUSION:-0}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"

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

log_section "Core 5-seed: Phase 1 baselines"
bash scripts/run_exp_phase1.sh

log_section "Core 5-seed: Phase 2 ours/key ablations/key knowledge"
bash scripts/run_exp_phase2.sh

log_section "Core 5-seed: significance and paper tables"
run_cmd "${PYTHON}" scripts/run.py analysis significance \
  --result-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/metrics/significance_tests.csv"
run_cmd "${PYTHON}" scripts/run.py report export-paper-tables --result-root "${OUTPUT_ROOT}"

log_section "Core 5-seed complete"

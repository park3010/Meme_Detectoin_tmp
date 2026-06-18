#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

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

log_section "Regenerating main performance tables"
run_cmd "${PYTHON}" scripts/aggregate_results.py \
  --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"

log_section "Regenerating structured interpretation tables"
run_cmd "${PYTHON}" scripts/aggregate_structured_results.py \
  --predictions-root "${OUTPUT_ROOT}/predictions" --output-root "${OUTPUT_ROOT}/metrics"

log_section "Regenerating significance tests"
run_cmd "${PYTHON}" scripts/run_significance_tests.py \
  --result-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/metrics/significance_tests.csv"

log_section "Exporting paper tables"
run_cmd "${PYTHON}" scripts/export_paper_tables.py --result-root "${OUTPUT_ROOT}"

log_section "Paper table refresh complete"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-harm_c}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-100}"
EPOCHS="${EPOCHS:-1}"
DEVICE="${DEVICE:-cpu}"
LABEL_SET="${LABEL_SET:-clean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result}"
RUN_ROOT="${OUTPUT_ROOT}/predictions/${DATASET}/ours_full/${SEED}"

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

log_section "Pipeline audit smoke: tests"
run_cmd "${PYTHON}" -m pytest tests -q

log_section "Pipeline audit smoke: train ${DATASET} seed=${SEED}"
run_cmd "${PYTHON}" scripts/run.py train --experiment ours_full \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --limit "${LIMIT}" \
  --label-set "${LABEL_SET}" \
  --disable-tqdm \
  --device "${DEVICE}" \
  --output-root "${OUTPUT_ROOT}"

log_section "Pipeline audit smoke: audit artifacts"
run_cmd "${PYTHON}" scripts/run.py audit \
  --run-root "${RUN_ROOT}" \
  --write-report \
  --allow-empty-split \
  --strict

log_section "Pipeline audit smoke complete"

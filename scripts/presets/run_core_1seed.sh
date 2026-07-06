#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_core_1seed}"
EPOCHS="${EPOCHS:-}"
LIMIT="${LIMIT:-}"

echo "CUDA_VISIBLE_DEVICES is respected if set externally; this preset does not set it."

cmd=(
  "${PYTHON}" scripts/run.py suite
  --suite core_1seed
  --config "${CONFIG}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --audit-after-run
  --strict
  --require-nonempty-metrics
)
if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"

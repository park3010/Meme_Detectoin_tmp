#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_core_smoke}"
LIMIT="${LIMIT:-}"

echo "CUDA_VISIBLE_DEVICES is respected if set externally; this preset does not set it."

cmd=(
  "${PYTHON}" scripts/run.py suite
  --suite core_smoke
  --config "${CONFIG}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --audit-after-run
  --strict
  --require-nonempty-metrics
)
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"

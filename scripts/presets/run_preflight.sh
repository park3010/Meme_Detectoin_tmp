#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/config.yaml}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-result_preflight}"
PROFILE="${PROFILE:-smoke}"
LABEL_SET="${LABEL_SET:-clean}"

echo "CUDA_VISIBLE_DEVICES is respected if set externally; this preset does not set it."

cmd=(
  "${PYTHON}" scripts/run.py preflight
  --profile "${PROFILE}"
  --config "${CONFIG}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --label-set "${LABEL_SET}"
  --write-report
)

echo "+ ${cmd[*]}"
"${cmd[@]}"

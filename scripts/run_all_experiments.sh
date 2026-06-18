#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

log_section() {
  echo "============================================================"
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
  echo "============================================================"
}

start_ts="$(date +%s)"
log_section "All experiments started at $(date)"

log_section "Running Experiment Phase 1"
bash scripts/run_exp_phase1.sh

log_section "Running Experiment Phase 2"
bash scripts/run_exp_phase2.sh

log_section "Running Experiment Phase 3"
bash scripts/run_exp_phase3.sh

end_ts="$(date +%s)"
elapsed="$((end_ts - start_ts))"
log_section "All experiments complete at $(date)"
echo "Elapsed seconds: ${elapsed}"

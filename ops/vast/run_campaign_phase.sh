#!/usr/bin/env bash
set -euo pipefail

phase="$1"
remaining_hours="${WAVEFORGE_REMAINING_HOURS:-8.0}"

cd /workspace/waveforge-thermal
if [[ "$phase" == "recovery" ]]; then
  exec .venv/bin/python -m waveforge.experiments.run_multitask_nca \
    --phase recovery \
    --output /workspace/waveforge-results/multitask_nca_recovery \
    --source-output /workspace/waveforge-results/multitask_nca \
    --remaining-hours "$remaining_hours"
else
  exec .venv/bin/python -m waveforge.experiments.run_multitask_nca \
    --phase "$phase" \
    --output /workspace/waveforge-results/multitask_nca \
    --remaining-hours "$remaining_hours"
fi

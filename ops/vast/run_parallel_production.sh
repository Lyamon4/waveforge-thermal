#!/usr/bin/env bash
set -euo pipefail

cd /workspace/waveforge-thermal
output=/workspace/waveforge-results/multitask_nca
python=.venv/bin/python
module=waveforge.experiments.run_multitask_nca

"$python" -m "$module" --phase production-lock --output "$output"

"$python" -m "$module" --phase production-seed --output "$output" \
  --seed 2026083102 > /var/log/portal/waveforge-seed-2026083102.log 2>&1 &
pid_1=$!
"$python" -m "$module" --phase production-seed --output "$output" \
  --seed 2026083103 > /var/log/portal/waveforge-seed-2026083103.log 2>&1 &
pid_2=$!
"$python" -m "$module" --phase production-seed --output "$output" \
  --seed 2026083104 > /var/log/portal/waveforge-seed-2026083104.log 2>&1 &
pid_3=$!

status=0
wait "$pid_1" || status=1
wait "$pid_2" || status=1
wait "$pid_3" || status=1
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

exec "$python" -m "$module" --phase production-finalize --output "$output" \
  --worker-count 3

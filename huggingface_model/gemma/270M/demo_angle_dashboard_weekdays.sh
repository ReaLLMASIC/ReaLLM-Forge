#!/usr/bin/env bash
set -euo pipefail

python ./vocab_angle_token_dashboard.py \
  --model google/gemma-3-270m \
  --tokens 'Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday' \
  --angle-threshold-deg 70 \
  --chunk-size 512 \
  --device cpu \
  --output-dir ./gemma_angle_dashboard_weekdays

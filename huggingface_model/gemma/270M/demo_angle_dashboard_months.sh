#!/usr/bin/env bash
set -euo pipefail

python ./vocab_angle_token_dashboard.py \
  --model google/gemma-3-270m \
  --tokens 'January,February,March,April,May,June,July,August,September,October,November,December' \
  --angle-threshold-deg 70 \
  --chunk-size 512 \
  --device cpu \
  --output-dir ./gemma_angle_dashboard_months

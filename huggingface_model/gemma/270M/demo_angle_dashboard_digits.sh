#!/usr/bin/env bash
set -euo pipefail

python ./vocab_angle_token_dashboard.py \
  --model google/gemma-3-270m \
  --tokens '0,1,2,3,4,5,6,7,8,9' \
  --angle-threshold-deg 70 \
  --chunk-size 512 \
  --device cpu \
  --output-dir ./gemma_angle_dashboard_digits

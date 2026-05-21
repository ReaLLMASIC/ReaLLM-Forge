#!/usr/bin/env bash
set -euo pipefail

python ./month_quant_angle_comparison.py \
  --model google/gemma-3-270m \
  --embedding-source input \
  --device cpu \
  --output-dir ./gemma_month_quant_angles

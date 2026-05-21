#!/usr/bin/env bash
set -euo pipefail

python ./weekday_quant_angle_comparison.py \
  --model google/gemma-3-270m \
  --embedding-source input \
  --device cpu \
  --output-dir ./gemma_weekday_quant_angles

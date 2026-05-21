#!/usr/bin/env bash
set -euo pipefail

python ./digit_quant_angle_comparison.py \
  --model google/gemma-3-270m \
  --embedding-source input \
  --device cpu \
  --output-dir ./gemma_digit_quant_angles

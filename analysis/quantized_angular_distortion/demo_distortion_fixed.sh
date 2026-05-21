#!/bin/bash

python quantized_angular_distortion.py \
  --dim 4096 \
  --trials 500 \
  --bits 3 4 5 \
  --angles-start 0 \
  --angles-stop 85 \
  --angles-step 1 \
  --scale-mode fixed \
  --clip-sigma 8.0 \
  --output distortion.png \
  --csv distortion.csv


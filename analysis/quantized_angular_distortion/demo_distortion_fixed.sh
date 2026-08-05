#!/bin/bash

python quantized_angular_distortion.py \
  --dim 4096 \
  --trials 500 \
  --bits 3 4 5 6 7 8 \
  --angles-start 0 \
  --angles-stop 90 \
  --angles-step 1 \
  --scale-mode fixed \
  --clip-sigma 8.0 \
  --output distortion.png \
  --csv distortion.csv


#!/bin/bash

# FP8
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats e6m1 e5m2 e4m3 e3m4 e2m5 finite:e1m6 \
  --fp-output fp8_angular_distortion.pdf \
  --fp-csv fp8_angular_distortion.csv

# FP7
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats e5m1 e4m2 e3m3 e2m4 finite:e1m5 \
  --fp-output fp7_angular_distortion.pdf \
  --fp-csv fp7_angular_distortion.csv

# FP6
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats e4m1 e3m2 e2m3 finite:e1m4 \
  --fp-output fp6_angular_distortion.pdf \
  --fp-csv fp6_angular_distortion.csv

# FP5
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats e3m1 e2m2 finite:e1m3 \
  --fp-output fp5_angular_distortion.pdf \
  --fp-csv fp5_angular_distortion.csv

# FP4
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats e2m1 finite:e1m2 \
  --fp-output fp4_angular_distortion.pdf \
  --fp-csv fp4_angular_distortion.csv

# FP3
python quantized_angular_distortion.py --fp-only --dim 4096 --trials 500 \
  --angles-start 0 --angles-stop 90 --angles-step 1 \
  --plot-fp --fp-theory-samples 200000 \
  --fp-formats finite:e1m1 \\
  --fp-output fp3_angular_distortion.pdf \
  --fp-csv fp3_angular_distortion.csv

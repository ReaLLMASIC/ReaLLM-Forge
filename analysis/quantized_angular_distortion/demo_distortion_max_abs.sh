
19:30❯ python quantized_angular_distortion.py \
  --angles-start 0 \
  --dim 4096 \
  --trials 500 \
  --bits 3 4 5 \
  --scale-mode maxabs \
  --output distortion_maxabs.png \
  --csv distortion_maxabs.csv


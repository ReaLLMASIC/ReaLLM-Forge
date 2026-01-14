#!/bin/bash
# data/sinewave/get_dataset.sh

for (( i = 0; i < 16; i++ )); do

period=$((i+15))
python prepare.py \
  --method sinewave \
  --train_input dummy.txt \
  --train_output s"$i"/train.bin \
  --val_output s"$i"/val.bin \
  --percentage_train 0.9 \
  --sine_period "$period" \
  --sine_points_per_period 15 \
  --sine_num_periods 2000 \
  --sine_amplitude 50

cp meta.pkl ./s"$i"
done

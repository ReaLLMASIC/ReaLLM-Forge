#!/bin/bash

for out_dir in $(ls out/); do

output_name="$1"

python analyze_with_dataset.py \
  --out_dir "$output_name" \
  --dataset minipile \
  --display topk \
  --activation_heatmap \
  --activation_heatmap_file activation_heatmap_"$output_name".html \
  --activation_hist_file hist_"$output_name".html \
  --attention_head_heatmap_file attn_"$output_name"_heatmap.html \
  --attention_head_hist_file attn_"$output_name"_hist.html

done

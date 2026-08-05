#!/bin/bash
# demos/dataset_analysis.sh

set -euo pipefail

root_out_dir="${1:?Usage: $0 <out_dir_root> [dataset]}"
dataset="${2:-minipile}"
results_dir="results"
mkdir -p "$results_dir"

if [[ ! -d "$root_out_dir" ]]; then
  echo "Error: '$root_out_dir' is not a directory" >&2
  exit 1
fi

# Analyze every run directory recursively: any directory containing ckpt.pt.
# Typical use: out/ contains subdirs for runs; each run has its own checkpoint.
mapfile -t run_dirs < <(find "$root_out_dir" -type f -name ckpt.pt -printf '%h\n' | sort -u)

if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "No run directories containing ckpt.pt found under '$root_out_dir'" >&2
  exit 1
fi

for output_dir in "${run_dirs[@]}"; do
  run_name="$(basename "$output_dir")"
  echo "Analyzing: $output_dir (dataset=$dataset)"

  python analyze_with_dataset.py \
    --out_dir "$output_dir" \
    --dataset "$dataset" \
    --display topk \
    --analysis_layers all \
    --analysis_heads all \
    --activation_heatmap \
    --activation_heatmap_file "$results_dir/${run_name}_activation_heatmap.html" \
    --activation_hist_file "$results_dir/${run_name}_activation_hist.html" \
    --attention_head_heatmap_file "$results_dir/${run_name}_attn_head_heatmap.html" \
    --attention_head_hist_file "$results_dir/${run_name}_attn_head_hist.html" \
    --plot_residual_magnitude \
    --residual_magnitude_file "$results_dir/${run_name}_residual_magnitude.html"
done

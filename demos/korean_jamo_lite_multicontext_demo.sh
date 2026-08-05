#!/usr/bin/env bash
set -euo pipefail

bash data/korean_jamo_mc/get_dataset.sh

lanes=(korean_jamo_mc/char korean_jamo_mc/first_jamo korean_jamo_mc/last_jamo korean_jamo_mc/eun_neun)

python3 train.py \
  --dataset korean_jamo_mc/char \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${lanes[@]}" \
  --max_iters 1000 \
  --dropout 0.2 \
  --top_k 1 \
  --sample_each_eval \
  --out_dir ./out_mc_korean_jamo_lite \
  --compile

prompt_dir="data/korean_jamo_mc/prompt_start"
prompt_dtype="$(python3 data/template/utils/korean/make_jamo_lite_prompt.py "English: Hello Korean: " "$prompt_dir" --dataset-root data/korean_jamo_mc | tail -n 1)"
mapfile -t prompt_files < "$prompt_dir/start_files.txt"

python3 sample.py \
  --out_dir ./out_mc_korean_jamo_lite \
  --multicontext \
  --multicontext_datasets "${lanes[@]}" \
  --multicontext_start_files "${prompt_files[@]}" \
  --multicontext_start_file_dtype "$prompt_dtype" \
  --max_new_tokens 512 \
  --top_k 1 \
  --num_samples 1

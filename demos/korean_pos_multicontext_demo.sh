#!/usr/bin/env bash
set -euo pipefail

bash data/korean_pos_mc/get_dataset.sh

lanes=(korean_pos_mc/script korean_pos_mc/choseong korean_pos_mc/jungseong korean_pos_mc/jongseong korean_pos_mc/jung_base1 korean_pos_mc/jung_base2 korean_pos_mc/jung_has_w korean_pos_mc/jung_has_y korean_pos_mc/jung_has_i korean_pos_mc/jong_base1 korean_pos_mc/jong_base2 korean_pos_mc/jong_base3 korean_pos_mc/choseong_tense korean_pos_mc/choseong_aspirated korean_pos_mc/choseong_nasal_liquid korean_pos_mc/choseong_place korean_pos_mc/jung_height korean_pos_mc/jung_backness korean_pos_mc/jung_round korean_pos_mc/jong_complex korean_pos_mc/has_batchim korean_pos_mc/syllable_index_mod korean_pos_mc/codepoint_mod korean_pos_mc/pos korean_pos_mc/char)

python3 train.py \
  --dataset korean_pos_mc/char \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets "${lanes[@]}" \
  --max_iters 2000 \
  --dropout 0.2 \
  --top_k 1 \
  --sample_each_eval \
  --out_dir ./out_mc_korean_pos \
  --compile

prompt_dir="data/korean_pos_mc/prompt_start"
prompt_dtype="$(python3 data/template/utils/korean/make_multicontext_prompt.py "English: Hello Korean: " "$prompt_dir" --dataset-root data/korean_pos_mc --use-pos | tail -n 1)"
mapfile -t prompt_files < "$prompt_dir/start_files.txt"

python3 sample.py \
  --out_dir ./out_mc_korean_pos \
  --multicontext \
  --multicontext_datasets "${lanes[@]}" \
  --multicontext_start_files "${prompt_files[@]}" \
  --multicontext_start_file_dtype "$prompt_dtype" \
  --max_new_tokens 512 \
  --top_k 1 \
  --num_samples 1

python3 data/template/utils/korean/recompose_multicontext.py out_mc_korean_pos data/korean_pos_mc/rendered_sample.txt --filename sample.txt --char-file data/korean_pos_mc/char/input.txt --use-pos || true

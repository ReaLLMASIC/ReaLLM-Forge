#!/bin/bash

# head to repo root
cd ../

DATASET="shakespeare_char"
bash "data/${DATASET}/get_dataset.sh"

N_LAYER=2
N_HEAD=2
N_KV_GROUP=2
N_EMBD=16
MAX_ITERS=3
BLOCK_SIZE=2
EVAL_ITERS=2
EVAL_INTERVAL=2

ln_opts=("" "--use_post_ln" "--use_peri_ln" "--use_post_ln --use_peri_ln")
parallel_opts=("" "--use_parallel_mlp")

for ln_flags in "${ln_opts[@]}"; do
  for par_flag in "${parallel_opts[@]}"; do
    tag="pre"
    if [[ "$ln_flags" == "--use_post_ln" ]]; then
      tag="post"
    elif [[ "$ln_flags" == "--use_peri_ln" ]]; then
      tag="peri"
    elif [[ "$ln_flags" == "--use_post_ln --use_peri_ln" ]]; then
      tag="peri_post"
    fi
    if [[ "$par_flag" != "" ]]; then
      tag="${tag}_parallel"
    fi

    OUT_DIR="out_${tag}"

    python3 train.py \
      --out_dir "$OUT_DIR" \
      --device cpu \
      --eval_interval $EVAL_INTERVAL \
      --log_interval 1 \
      --block_size $BLOCK_SIZE \
      --batch_size 2 \
      --n_layer $N_LAYER \
      --n_head $N_HEAD \
      --n_kv_group $N_KV_GROUP \
      --n_embd $N_EMBD \
      --max_iters $MAX_ITERS \
      --lr_decay_iters 2 \
      --dropout 0.0 \
      --dataset "$DATASET" \
      $ln_flags $par_flag

    python3 sample.py --device cpu --out_dir "$OUT_DIR" --num_samples 1 --max_new_tokens 1
  done
done


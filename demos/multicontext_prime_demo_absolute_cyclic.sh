#!/bin/bash
# Compare conventional learned absolute embeddings vs cyclic absolute embeddings
# on the same multicontext prime setup used in multicontext_prime_demo.sh.

set -euo pipefail

pushd data/shakespeare_char_prime
bash get_dataset.sh
popd

COMMON_ARGS=(
  --training_mode multicontext
  --multicontext
  --multicontext_datasets
    shakespeare_char
    data/shakespeare_char_prime/char_mod2
    data/shakespeare_char_prime/char_mod3
    data/shakespeare_char_prime/char_mod7
    data/shakespeare_char_prime/char_mod11
    data/shakespeare_char_prime/char_mod13
    data/shakespeare_char_prime/char_mod17
  --max_iters 2000
  --dropout 0.2
  --top_k 1
  --sample_each_eval
  --use_qk_norm
  --use_qk_norm_scale
  --use_abs_pos_embeddings
  --compile
)

# 1) Conventional learned absolute position embeddings (backwards-compatible default).
python3 train.py \
  "${COMMON_ARGS[@]}" \
  --absolute_pos_embedding_variant learned \
  --out_dir ./out_mc_shakespeare_abs_learned

# 2) Cyclic absolute position embeddings with cycle sizes 2,3,5.
python3 train.py \
  "${COMMON_ARGS[@]}" \
  --absolute_pos_embedding_variant cyclic \
  --cyclic_abs_pos_cycle_lengths 2 3 5 \
  --no-cyclic_abs_pos_randomize_starts \
  --out_dir ./out_mc_shakespeare_abs_cyclic

# Optional random-start training variant (per-cycle random phase each step).
# python3 train.py \
#   "${COMMON_ARGS[@]}" \
#   --absolute_pos_embedding_variant cyclic \
#   --cyclic_abs_pos_cycle_lengths 2 3 5 \
#   --cyclic_abs_pos_randomize_starts \
#   --out_dir ./out_mc_shakespeare_abs_cyclic_random

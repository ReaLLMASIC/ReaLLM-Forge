#!/bin/bash
# hp_searches/efficiency_targets_demo.sh

for target in params vram iter; do
  python3 hyperparam_search.py \
    --orig_settings ./hp_searches/efficiency_targets_demo.yaml \
    --param_names \
    n_layer \
    n_head \
    n_embd \
    mlp_size \
    n_qk_head_dim \
    n_v_head_dim \
    --increments \
    1 \
    1 \
    16 \
    16 \
    16 \
    16 \
    --random_iterations 1 \
    --iterations 1 \
    --num_iterations 10 \
    --efficiency_target "${target}" \
    --results_file "efficiency_target_${target}_demo.yaml"
  done


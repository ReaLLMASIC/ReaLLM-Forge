#!/bin/bash
# hp_searches/test_efficiency_targets.sh
# Run hyperparameter search using each efficiency target option.

set -e

for target in params vram iter; do
  python hyperparam_search.py \
    --orig_settings ./hp_searches/efficiency_targets_demo.yaml \
    --param_names n_layer n_head n_embd mlp_size \
    --increments 1 1 16 16 \
    --random_iterations 1 \
    --iterations 1 \
    --num_iterations 1 \
    --efficiency_target "$target" \
    --results_file "results_${target}.yaml"
done

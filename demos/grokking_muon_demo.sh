#!/bin/bash
# Demonstration of Muon optimizer on a modular addition task
set -e

# generate dataset if not already present
pushd data/modular_addition
bash create_examples.sh
python3 prepare.py --input_file data/base_16.txt --token_file tokens.txt
popd

common_args="--dataset modular_addition --max_iters 2000 --batch_size 64 --block_size 64 --n_layer 2 --n_head 2 --n_embd 64 --eval_interval 100 --eval_iters 100 --init_from scratch"

# Train with AdamW
python3 train.py $common_args --optimizer adamw --out_dir out_grok_adamw

# Train with Muon
python3 train.py $common_args --optimizer muon --muon_momentum 0.95 --muon_ns_iters 5 --muon_scale 0.2 --out_dir out_grok_muon

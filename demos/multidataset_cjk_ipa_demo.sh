#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# tokenize on commonvoice_zh
pushd "data/commonvoice_zh"
bash get_dataset.sh
popd

# tokenize on commonvoice_ko
pushd "data/commonvoice_ko"
bash get_dataset.sh
popd

# tokenize on commonvoice_ja
pushd "data/commonvoice_ja"
bash get_dataset.sh
popd

# running the training
# python3 optimization_and_search/run_experiments.py --config explorations/multidataset.json --output_dir out_multi_zh
python3 train.py \
    --dataset commonvoice_zh \
    --training_mode multidataset \
    --dataset_list commonvoice_zh commonvoice_ko commonvoice_ja \
    --dataset_sampling_probs 1 1 1 \
    --use_lsv \
    --max_iters 10000 \
    --batch_size 16 \
    --apply_lsv_at_layer_idx 0 \
    --eval_interval 500 \
    --eval_iters 50 \
    --dataset_interleaving \
    --dataset_interleaving_shuffle \
    --lsv_variant one_hot \
    --out_dir "out_multi_zh" \
    --init_from "scratch" \
    --gns_type exact

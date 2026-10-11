#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

#!/bin/bash
# demos/multitokenization_ipa.sh

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# obtain and tokenize commonvoice_ja
pushd data/commonvoice_ja
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_ja"
fi
popd

# obtain and tokenize commonvoice_zh
pushd data/commonvoice_zh
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_zh"
fi
popd

# obtain and tokenize commonvoice_ko
pushd data/commonvoice_ko
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_ko"
fi
popd

python3 optimization_and_search/run_experiments.py -c explorations/multidataset.yaml -o out_multidataset_ipa
# python3 train.py \
#     --dataset commonvoice_zh \
#     --training_mode multidataset \
#     --dataset_list commonvoice_zh commonvoice_ko commonvoice_ja \
#     --dataset_sampling_probs 1 1 1 \
#     --use_lsv \
#     --max_iters 10000 \
#     --batch_size 16 \
#     --apply_lsv_at_layer_idx 0 \
#     --eval_interval 500 \
#     --eval_iters 50 \
#     --dataset_interleaving \
#     --dataset_interleaving_shuffle \
#     --lsv_variant one_hot \
#     --out_dir "out_multi_zh" \
#     --init_from "scratch" \
#     --gns_type exact

#!/bin/bash
# demos/multitokenization_ipa.sh

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# obtain and tokenize commonvoice_en
pushd data/commonvoice_ja
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_ja"
fi
popd

pushd data/commonvoice_ko
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_ko"
fi
popd

pushd data/commonvoice_zh
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_ipa.sh
else
  echo "train.bin val.bin and meta.pkl already found for commonvoice_zh"
fi
popd

# obtain and tokenize snac_cvko
pushd data/snac_cvja
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_text.sh
else
  echo "train.bin val.bin and meta.pkl already found for snac_cvja"
fi
popd

pushd data/snac_cvko
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_text.sh
else
  echo "train.bin val.bin and meta.pkl already found for snac_cvko"
fi
popd

pushd data/snac_cvzh
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_text.sh
else
  echo "train.bin val.bin and meta.pkl already found for snac_cvzh"
fi
popd

python3 optimization_and_search/run_experiments.py -c explorations/multidata_base.yaml -o out_multidata_base_one_hot

python3 optimization_and_search/run_experiments.py -c explorations/multidataset_single.yaml -o out_multidata_single

# python3 train.py \
#   --training_mode multidataset \
#   --multidataset_wte \
#   --dataset commonvoice_en \
#   --dataset_list  commonvoice_en byte_en snac_cven \
#   --dataset_sampling_probs 10 1 1 \
#   --dataset_sampling_probs_final 1 1 1 \
#   --dataset_sampling_probs_transition_method cosine \
#   --dataset_interleaving \
#   --batch_size 8 \
#   --learning_rate "6e-4" \
#   --decay_lr \
#   --min_lr "6e-5" \
#   --n_layer 12 \
#   --n_head 12 \
#   --n_embd 768 \
#   --block_size 256 \
#   --use_rotary_embeddings \
#   --no-use_abs_pos_embeddings \
#   --max_sample_tokens 100 \
#   --max_iters 10000 \
#   --warmup_iters 1000 \
#   --eval_interval 1000 \
#   --sample_each_eval \
#   --init_from "scratch" \
#   --use_lsv \
#   --apply_lsv_at_layer_idx 0 \
#   --lsv_variant "one_hot" \
#   --out_dir "out_multoken_one_hot" \
#   --tensorboard_run_name "out_one_hot" \
#   --compile


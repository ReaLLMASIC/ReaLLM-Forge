#!/bin/bash
# demos/multitokenization_ipa.sh

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# obtain and tokenize commonvoice_en
pushd data/commonvoice_zh
bash get_ipa.sh
popd

# obtain and tokenize minipile
pushd data/snac_cvzh
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_text.sh
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

python3 train.py \
  --multidataset_wte \
  --training_mode multidataset \
  --dataset_list commonvoice_zh snac_cvzh \
  --use_qk_norm \
  --use_qk_norm_scale \
  --no-use_abs_pos_embeddings \
  --use_rotary_embeddings \
  --compile \
  --max_sample_tokens 256 \
  --top_k 1 10 \
  --colorize_mode minmax

popd

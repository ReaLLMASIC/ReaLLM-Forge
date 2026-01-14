#!/bin/bash
# demos/multidataset_multitokenization.sh

# obtain and tokenize shakespeare char
bash data/shakespeare_char/get_dataset.sh

# obtain and tokenize minipile
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

python3 train.py \
  --multidataset_wte \
  --training_mode multidataset \
  --dataset_list shakespeare_char minipile \
  --use_qk_norm \
  --use_qk_norm_scale \
  --no-use_abs_pos_embeddings \
  --use_rotary_embeddings \
  --compile \
  --max_sample_tokens 256 \
  --top_k 1 10 \
  --colorize_mode minmax



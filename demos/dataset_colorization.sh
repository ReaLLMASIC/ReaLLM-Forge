#!/bin/bash
# demos/dataset_colorization.sh

pushd data/filipino/tagalog_filipino_eng_translation
bash get_dataset.sh
popd

python train.py \
  --dataset filipino/tagalog_filipino_eng_translation \
  --compile \
  --colorize_output \
  --colorize_mode all \
  --max_iters 10000

python token_prediction_colorizer.py \
  --out_dir        out \
  --dataset        filipino/tagalog_filipino_eng_translation  \
  --split          val \
  --num_tokens     2048 \
  --device         cuda:0 \
  --block_size     256 \
  --mode           minmax  \
  --output_file    kulay_ng_dataset_minmax.txt

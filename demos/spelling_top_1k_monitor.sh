#!/bin/bash


cd data/billsum
python3 get_dataset.py
python3 prepare.py -t input.txt --method char
cd -

python3 train.py \
  --dataset billsum \
  --n_layer 3 \
  --n_embd 120 \
  --colorize_mode softmax \
  --colorize_output \
  --sample_metrics \
  --max_sample_tokens 256 \
  --top_k 1 10


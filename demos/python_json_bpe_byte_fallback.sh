#!/bin/bash

pushd data/python-codes-25k/
bash get_dataset.sh
python3 prepare.py -t input.txt --method json_bpe_byte_fallback --json_tokens_file ../template/premade_vocab_sets/python_programming_tokens.json
popd

python3 train.py --dataset python-codes-25k


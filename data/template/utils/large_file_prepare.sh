#!/bin/bash

input_file=${1:-input.txt}
tokenization=${2:-tiktoken}
tokenizer_config_path=${3:-}
max_parallel=${4:-1}
chunk_size_mb=${5:-50}

python3 ./utils/partition_file.py --input_file "${input_file}" --chunk_size_mb "${chunk_size_mb}"

batch_prepare_args=(
  --input_dir partitioned_file
  --prepare_script prepare.py
  --tokenizer "$tokenization"
  --max_parallel "$max_parallel"
)

if [[ "$tokenization" == "char_bpe" && -n "$tokenizer_config_path" ]]; then
  batch_prepare_args+=(--char_bpe_vocab_path "$tokenizer_config_path")
fi

if [[ "$tokenization" == "json_byte_fallback" && -n "$tokenizer_config_path" ]]; then
  batch_prepare_args+=(--json_tokens_file "$tokenizer_config_path")
fi

python3 ./utils/batch_prepare.py "${batch_prepare_args[@]}"

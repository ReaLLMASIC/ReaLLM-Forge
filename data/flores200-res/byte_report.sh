#!/bin/bash

lang_array=(
  "text_eng_Latn"
  "text_jpn_Jpan"
  "text_kor_Hang"
  "text_zho_Hans"
)

# for lang in "${lang_array[@]}"; do
#   # Tiktoken
#   python3 prepare.py \
#     -t "${lang}.txt" \
#     -s -S "$lang" \
#     -p 1.0 \
#     --method tiktoken \
#     --report_byte_tokenization
# done

# for lang in "${lang_array[@]}"; do
#   # Deepseek
#   python3 prepare.py \
#     -t "${lang}.txt" \
#     -s -S "$lang" \
#     -p 1.0 \
#     --method json_byte_fallback \
#     --json_tokens_file ../template/premade_vocab_sets/deepseek_tokens.json \
#     --report_byte_tokenization
# done

# for lang in "${lang_array[@]}"; do
#   # Gemma
#   python3 prepare.py \
#     -t "${lang}.txt" \
#     -s -S "$lang" \
#     -p 1.0 \
#     --method json_byte_fallback \
#     --json_tokens_file ../template/premade_vocab_sets/gemma_tokens.json \
#     --report_byte_tokenization
# done

for lang in "${lang_array[@]}"; do
  python3 prepare.py -t ipa_"$lang".txt -p 1.0 --method json_byte_fallback --json_tokens_file ../template/premade_vocab_sets/qwen_tokens.json -s -S "$lang" --report_byte_tokenization
done

# for lang in "${lang_array[@]}"; do
#   python3 prepare.py -t ipa_"$lang".txt -p 1.0 --method json_byte_fallback --json_tokens_file ../template/premade_vocab_sets/ipa_extended.json -s -S "$lang" --report_byte_tokenization
# done

#!/usr/bin/env bash
set -euo pipefail

# Prepare one FLORES-200 language three ways for bits-per-byte comparisons:
#   1. raw text with tiktoken
#   2. raw text as UTF-8 bytes
#   3. IPA text with custom IPA tokens plus byte fallback
#
# Usage:
#   cd data/flores200-res
#   ./get_dataset.sh                 # if text_*.txt files do not already exist
#   ./phoneticize.sh                 # if ipa_text_*.txt files do not already exist
#   ./demo_bits_per_byte_tokenizers.sh eng_Latn
#   cd ../..
#   python3 optimization_and_search/run_experiments.py \
#     --config_format yaml \
#     --config explorations/flores200_bits_per_byte_tokenizer_comparison.yaml \
#     --output_dir out/flores200_bpb

LANG_CODE="${1:-eng_Latn}"
TEXT_INPUT="text_${LANG_CODE}.txt"
IPA_INPUT="ipa_text_${LANG_CODE}.txt"
CUSTOM_CHARS_FILE="../template/phoneme_list.txt"

if [[ ! -f "${TEXT_INPUT}" ]]; then
  echo "Missing ${TEXT_INPUT}. Run ./get_dataset.sh first or pass a language code with an existing text_<lang>.txt file." >&2
  exit 1
fi

if [[ ! -f "${IPA_INPUT}" ]]; then
  echo "Missing ${IPA_INPUT}. Run ./phoneticize.sh first, or create an IPA text file at ${IPA_INPUT}." >&2
  exit 1
fi

if [[ ! -f "${CUSTOM_CHARS_FILE}" ]]; then
  echo "Missing IPA/custom character list: ${CUSTOM_CHARS_FILE}" >&2
  exit 1
fi

export PYTHONPATH="../template:${PYTHONPATH:-}"

python3 prepare.py \
  -t "${TEXT_INPUT}" \
  --method tiktoken \
  --tiktoken_encoding gpt2 \
  --output_tokenization_subdir \
  --output_subdir_suffix "${LANG_CODE}"

python3 prepare.py \
  -t "${TEXT_INPUT}" \
  --method byte \
  --output_tokenization_subdir \
  --output_subdir_suffix "${LANG_CODE}"

python3 prepare.py \
  -t "${IPA_INPUT}" \
  --method custom_char_byte_fallback \
  --custom_chars_file "${CUSTOM_CHARS_FILE}" \
  --output_tokenization_subdir \
  --output_subdir_suffix "ipa_${LANG_CODE}"

cat <<MSG
Prepared datasets:
  data/flores200-res/tiktoken_${LANG_CODE}
  data/flores200-res/byte_${LANG_CODE}
  data/flores200-res/custom_char_byte_fallback_ipa_${LANG_CODE}

Next, from the repo root run:
  python3 optimization_and_search/run_experiments.py --config_format yaml --config explorations/flores200_bits_per_byte_tokenizer_comparison.yaml --output_dir out/flores200_bpb
MSG

#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Running text to speech training..."

url="https://huggingface.co/datasets/xinyixuu/tiny_sherlock_whisper_snac_combined"
out_dir="tiny_sherlock_whisper_snac_combined"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

for i in $(seq -f "%02g" 1 24); do
  for j in $(seq -f "%02g" 0 3); do
    wget -nc -O "${out_dir}/tiny_sherlock_audio_${i}_part0${j}.json" "${url}/resolve/main/tiny_sherlock_whisper_snac_combined/tiny_sherlock_audio_${i}_part0${j}.json?download=true" || {
      echo "File tiny_sherlock_audio_${i}_part0${j}.json not found, skipping."
      rm -rf ${out_dir}/tiny_sherlock_audio_${i}_part0${j}.json
      continue
    }
  done
done

# Get the input file for running the dataset
python3 extract_data.py "tiny_sherlock_whisper_snac_combined" "input.txt" --directory

# prepare the dateset for training
python3 prepare.py -t input.txt --tokens_file tokens.txt --method custom

# running the training
pushd ../../
python3 run_experiments.py --config explorations/snac_text.json --output_dir out_test
popd
:

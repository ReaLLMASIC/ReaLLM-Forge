# !/bin/bash

# Set strict error handling
set -xe

# Install python dependencies for Hugging face
pip install -U "huggingface_hub[cli]"

# Authentication with Hugging Face
# Replace with your hugging face tokens
##### You can find and create your own tokens here: https://huggingface.co/settings/tokens ######
##### "Token Type" of "Read" is recommended. ########
if [[ -f ~/.cache/huggingface/token && -s ~/.cache/huggingface/token ]]; then
  export HF_TOKEN=$(cat ~/.cache/huggingface/token)
else
  echo "Consider running 'python3 ./utils/save_hf_token.py' to automate finding HF_TOKEN"
  read -s -p "To continue, please enter your Hugging Face token: " HF_TOKEN
  echo "" # Add a newline for better readability
fi

# Authenticate with hugging face
echo "Authenticating with Hugging Face..."
huggingface-cli login --token "${HF_TOKEN}"

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

url="https://huggingface.co/datasets/xinyixuu/ja_snac"
out_dir="json_outs"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

# Download transcription files under "transcription" directory.
pushd "$script_dir/${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.json" "${url}/resolve/main/json_outs_ja/dev.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated_part_1.json" "${url}/resolve/main/json_outs_ja/validated_part_1.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.json" "${url}/resolve/main/json_outs_ja/test.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.json" "${url}/resolve/main/json_outs_ja/invalidated.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.json" "${url}/resolve/main/json_outs_ja/train.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated2_part1.json" "${url}/resolve/main/json_outs_ja/validated2_part1.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "tiny.json" "${url}/resolve/main/json_outs_ja/tiny.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "final.json" "${url}/resolve/main/json_outs_ja/final.json?download=true" || true

echo "snac conversion files downloaded and saved to ${out_dir}."
popd

output_snac_txt="snac_text.txt"
python3 "$script_dir"/utils/extract_json_snactext.py "$out_dir" "sentence" "$output_snac_txt" --directory
echo "snac-text extraction finished."

output_snac_ipa="snac_ipa.txt"
python3 "$script_dir"/utils/extract_json_snactext.py "$out_dir" "spaced_ipa" "$output_snac_ipa" --directory
echo "snac-ipa extraction finished."

url="https://huggingface.co/datasets/xinyixuu/ko_snac"
out_dir_ko="json_outs_ko"

if [[ ! -d "${out_dir_ko}" ]]; then
  mkdir -p "${out_dir_ko}"
fi

# Download transcription files under "transcription" directory.
pushd "$script_dir/${out_dir_ko}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.json" "${url}/resolve/main/json_dir/dev.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.json" "${url}/resolve/main/json_dir/other.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.json" "${url}/resolve/main/json_dir/test.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.json" "${url}/resolve/main/json_dir/invalidated.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.json" "${url}/resolve/main/json_dir/train.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.json" "${url}/resolve/main/json_dir/validated.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac.json" "${url}/resolve/main/json_dir/ko_snac.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_1.json" "${url}/resolve/main/json_dir/ko_snac_1.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_3.json" "${url}/resolve/main/json_dir/ko_snac_3.json?download=true" || true

echo "snac conversion files downloaded and saved to ${out_dir_ko}."
popd

output_snac_txt="snac_text.txt"
python3 "$script_dir"/utils/extract_json_snactext.py "$out_dir_ko" "text" "$output_snac_txt" --directory
echo "snac-text extraction finished."

output_snac_ipa="snac_ipa.txt"
python3 "$script_dir"/utils/extract_json_snactext.py "$out_dir_ko" "ipa" "$output_snac_ipa" --directory
echo "snac-ipa extraction finished."

# Tokenization step to create train.bin and val.bin files.
python3 "$script_dir"/prepare.py -t "$output_snac_ipa" --method custom_char_byte_fallback --custom_chars_file "$script_dir"/utils/phoneme_snac.txt
# python3 "$script_dir"/prepare.py -t "$output_snac_txt" --method tiktoken

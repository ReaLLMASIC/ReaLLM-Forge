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
hf auth login --token "${HF_TOKEN}"

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

output_txt="ja_text.txt"
for jsonfile in "$out_dir"/*.json; do
    # Check if the .json file exists (handles the case where no .json files are present)
    if [ -f "$jsonfile" ]; then
        echo "Processing $jsonfile..."
        # Get the filename without the extension for output filename
        filename=$(basename "${jsonfile%.json}")
        python3 "$script_dir"/utils/extract_json_values.py "$jsonfile" "sentence" "$output_txt"
    fi
done

# Tokenization step to create train.bin and val.bin files.
python3 "$script_dir"/prepare.py -t "$output_txt" --method byte

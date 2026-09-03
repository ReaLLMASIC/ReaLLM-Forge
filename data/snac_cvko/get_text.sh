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

url="https://huggingface.co/datasets/xinyixuu/ko_snac"
out_dir="json_outs"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

# Download transcription files under "transcription" directory.
pushd "$script_dir/${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "dev.json" "${url}/resolve/main/json_dir/dev.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "other.json" "${url}/resolve/main/json_dir/other.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "test.json" "${url}/resolve/main/json_dir/test.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "invalidated.json" "${url}/resolve/main/json_dir/invalidated.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "train.json" "${url}/resolve/main/json_dir/train.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "validated.json" "${url}/resolve/main/json_dir/validated.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac.json" "${url}/resolve/main/json_dir/ko_snac.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_1.json" "${url}/resolve/main/json_dir/ko_snac_1.json?download=true" || true
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "ko_snac_3.json" "${url}/resolve/main/json_dir/ko_snac_3.json?download=true" || true

echo "snac conversion files downloaded and saved to ${out_dir}."
popd

output_txt="ko_text.txt"
for jsonfile in "$out_dir"/*.json; do
    # Check if the .tsv file exists (handles the case where no .tsv files are present)
    if [ -f "$jsonfile" ]; then
        echo "Processing $jsonfile..."
        # Get the filename without the extension for output filename
        python3 "$script_dir"/utils/extract_json_values.py "$jsonfile" "text" "$output_txt"
    fi
done

#TODO(gkielian): see if we can fix the parsing of rows instead of deleting
# Remove lines which were not correclty processed (and start with numberic hash)
wc -l "$output_txt"
sed -i "/^[0-9].*/g" "$output_txt"
wc -l "$output_txt"

# Tokenization step to create train.bin and val.bin files.
#python3 "$script_dir"/prepare.py -t "$output_txt" --method char
python3 "$script_dir"/prepare.py -t "$output_txt" --method tiktoken

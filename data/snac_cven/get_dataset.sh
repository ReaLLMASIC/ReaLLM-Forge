# !/bin/bash
# data/snac_cvzh/get_text.sh

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

url="https://huggingface.co/datasets/xinyixuu/en_snac"
out_dir="out_ipa"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

pushd "${out_dir}"
wget --header="Authorization: Bearer ${HF_TOKEN}" -nc -O "en_transcription.json" "${url}/resolve/main/en_transcription.json?download=true" || true

echo "json files downloaded and saved to out_ipa."
popd

output_txt="en_snac_text.txt"
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
# python3 "$script_dir"/prepare.py -t "$output_snac_ipa" --method custom_char_byte_fallback --custom_chars_file "$script_dir"/utils/phoneme_snac.txt
python3 "$script_dir"/prepare.py -t "$output_txt" --method tiktoken

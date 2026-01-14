#!/bin/bash
# demos/multidataset_conversation_demo.sh

set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RESET='\033[0m'

datasets=(minipile databricks-dolly-15k)

# --- Obtain + tokenize datasets ---
for dataset in "${datasets[@]}"; do
  if [ ! -d "data/${dataset}" ]; then
    echo -e "${RED}[ERROR]${RESET} Missing dataset directory: data/${dataset}" >&2
    exit 1
  fi

  echo -e "${CYAN}=== Processing dataset: ${dataset} ===${RESET}"
  echo -e "${BLUE}[ENTER]${RESET} data/${dataset}"
  pushd "data/${dataset}" > /dev/null

  # obtain dataset
  if [ ! -f "input.txt" ] && [ ! -f "train.bin" &&  [ ! -f "val.bin" ] && [ ! -f "meta.pkl" ]; then
    echo -e "${MAGENTA}[OBTAIN]${RESET} Downloading dataset: ${dataset}"
    bash get_dataset.sh
  else
    echo -e "${YELLOW}[SKIP]${RESET} input.txt already exists for ${dataset}"
  fi

  # tokenize dataset
  if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
    echo -e "${GREEN}[TOKENIZE]${RESET} Running prepare.py for ${dataset}"
    python3 prepare.py -t input.txt --method tiktoken
  else
    echo -e "${YELLOW}[SKIP]${RESET} Tokenized files (train.bin, val.bin, meta.pkl) already exist for ${dataset}"
  fi

  popd > /dev/null
  echo -e "${BLUE}[EXIT]${RESET} data/${dataset}"
  echo
done

# --- Run training ---
echo -e "${CYAN}=== Starting training run ===${RESET}"
python3 train.py \
    --dataset "${datasets[1]}" \
    --training_mode multidataset \
    --log_interval 10 \
    --batch_size 64 \
    --dataset_list "${datasets[@]}" \
    --dataset_sampling_probs 20 1 \
    --dataset_sampling_probs_final 1 20 \
    --dataset_sampling_probs_transition_method "linear" \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --max_iters 5000 \
    --eval_interval 5000 \
    --sample_start_tokens $'\n\n#U:\nWhat do you think would be a good vacation plan?\n#B:\n' \
    --init_from "scratch" \
    --compile


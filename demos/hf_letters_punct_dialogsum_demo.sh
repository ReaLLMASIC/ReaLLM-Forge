#!/bin/bash
# demos/hf_letters_punct_dialogsum_demo.sh
#
# Companion to hf_local_json_byte_fallback_demo.sh. Instead of authoring a
# fresh JSON vocab inline, this demo pulls a premade vocab of just
# letters + digits + punctuation + whitespace (including newlines) from
#
#     data/template/premade_vocab_sets/letters_punctuation.json
#
# and materializes it into a local HuggingFace tokenizer directory that
# keeps byte_fallback=true. Everything outside this tiny vocab (emoji,
# CJK, accented characters, …) falls through the <0xNN> byte pieces, so
# the tokenizer is still total over UTF-8.
#
# Pipeline:
#   1. Obtain dialogsum (if missing).
#   2. Copy the premade JSON vocab alongside the dataset for reproducibility.
#   3. Build tokenizer.json + tokenizer_config.json with byte_fallback.
#   4. prepare.py --method huggingface --hf_tokenizer_name <abs local dir>.
#   5. train.py --dataset dialogsum with default-ish settings.

set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
RESET='\033[0m'

DATASET="dialogsum"
DATA_DIR="data/${DATASET}"
PREMADE_JSON="data/template/premade_vocab_sets/letters_punctuation.json"
TOK_DIR="${DATA_DIR}/hf_letters_punct_tok"
JSON_VOCAB_LOCAL="${DATA_DIR}/hf_letters_punct_tokens.json"

if [ ! -d "${DATA_DIR}" ]; then
  echo -e "${RED}[ERROR]${RESET} Missing ${DATA_DIR}. Run this from the repo root." >&2
  exit 1
fi
if [ ! -f "${PREMADE_JSON}" ]; then
  echo -e "${RED}[ERROR]${RESET} Missing premade vocab: ${PREMADE_JSON}" >&2
  exit 1
fi

mkdir -p "${TOK_DIR}"
TOK_DIR_ABS="$(cd "${TOK_DIR}" && pwd)"

# ---------------------------------------------------------------------------
# 1) Obtain dataset
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [1/5] Obtaining dialogsum dataset ===${RESET}"
pushd "${DATA_DIR}" > /dev/null
if [ ! -f "input.txt" ]; then
  echo -e "${MAGENTA}[OBTAIN]${RESET} Running get_dataset.sh"
  bash get_dataset.sh
else
  echo -e "${YELLOW}[SKIP]${RESET} input.txt already exists"
fi
popd > /dev/null

# ---------------------------------------------------------------------------
# 2) Snapshot the premade JSON vocab next to the dataset so a meta.pkl reader
#    sees exactly which tokens produced the run.
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [2/5] Using premade vocab ${PREMADE_JSON} ===${RESET}"
cp -f "${PREMADE_JSON}" "${JSON_VOCAB_LOCAL}"
echo -e "${GREEN}[COPY]${RESET} ${PREMADE_JSON} -> ${JSON_VOCAB_LOCAL}"

# ---------------------------------------------------------------------------
# 3) Build the HF tokenizer directory from the premade JSON.
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [3/5] Building local HF tokenizer directory -> ${TOK_DIR_ABS} ===${RESET}"

python3 - "$JSON_VOCAB_LOCAL" "$TOK_DIR_ABS" <<'PY'
import json
import os
import sys

from tokenizers import Tokenizer, decoders, models

json_vocab_path, out_dir = sys.argv[1], sys.argv[2]

with open(json_vocab_path, "r", encoding="utf-8") as f:
    custom_tokens = json.load(f)
assert isinstance(custom_tokens, list), "JSON vocab must be a flat list of strings"

# Canonical byte_fallback layout: <unk>, <s>, </s>, then <0x00>..<0xFF>
# for raw-byte fallback, then the user-supplied pieces.
specials = ["<unk>", "<s>", "</s>"]
byte_tokens = [f"<0x{b:02X}>" for b in range(256)]

vocab_list = []
seen = set()
for tok in specials + byte_tokens + custom_tokens:
    if tok in seen:
        continue
    seen.add(tok)
    vocab_list.append(tok)

vocab = {tok: i for i, tok in enumerate(vocab_list)}

tok = Tokenizer(models.BPE(
    vocab=vocab,
    merges=[],
    unk_token="<unk>",
    fuse_unk=True,
    byte_fallback=True,
))

# No normalizer / pre-tokenizer: we want the model to see raw characters,
# including spaces, newlines, and tabs, so every piece in the premade vocab
# (which already contains " ", "\n", "\t", "\r") can match directly. Anything
# else (emoji, CJK, accented Latin) falls through to <0xNN> byte pieces.
tok.normalizer = None
tok.pre_tokenizer = None
tok.decoder = decoders.ByteFallback()

tok.add_special_tokens(["<unk>", "<s>", "</s>"])

os.makedirs(out_dir, exist_ok=True)
tok.save(os.path.join(out_dir, "tokenizer.json"))

with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1_000_000_000,
    }, f, indent=2)

print(f"[hf-letters-punct] wrote {out_dir}/tokenizer.json "
      f"(vocab_size={tok.get_vocab_size()}, byte_fallback=True)")

# Smoke-test the round-trip so any layout bug surfaces immediately.
probe = "Hi Bob!\nHow was your weekend? 👋 (Let's talk — 안녕.)"
enc = tok.encode(probe)
dec = tok.decode(enc.ids)
print(f"[hf-letters-punct] probe len(ids)={len(enc.ids)} round-trip={dec!r}")
PY

# ---------------------------------------------------------------------------
# 4) Tokenize via the existing HuggingFace pipeline in prepare.py
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [4/5] Tokenizing dialogsum via --method huggingface ===${RESET}"
pushd "${DATA_DIR}" > /dev/null

need_tokenize=0
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  need_tokenize=1
else
  stored_name="$(python3 -c "import pickle,sys; m=pickle.load(open('meta.pkl','rb')); print(m.get('hf_tokenizer_name',''))" 2>/dev/null || true)"
  if [ "${stored_name}" != "${TOK_DIR_ABS}" ]; then
    echo -e "${YELLOW}[STALE]${RESET} meta.pkl stored hf_tokenizer_name='${stored_name}', expected '${TOK_DIR_ABS}'. Re-tokenizing."
    rm -f train.bin val.bin meta.pkl
    rm -rf hf_tokenizer
    need_tokenize=1
  fi
fi

if [ "${need_tokenize}" -eq 1 ]; then
  echo -e "${GREEN}[TOKENIZE]${RESET} prepare.py --method huggingface --hf_tokenizer_name ${TOK_DIR_ABS}"
  python3 prepare.py \
    -t input.txt \
    --method huggingface \
    --hf_tokenizer_name "${TOK_DIR_ABS}" \
    --hf_use_fast \
    -T
else
  echo -e "${YELLOW}[SKIP]${RESET} train.bin/val.bin/meta.pkl already up to date"
fi
popd > /dev/null

# ---------------------------------------------------------------------------
# 5) Train
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [5/5] Starting training on dataset=${DATASET} ===${RESET}"
python3 train.py \
    --dataset "${DATASET}" \
    --log_interval 10 \
    --batch_size 32 \
    --max_iters 2000 \
    --eval_interval 500 \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --sample_start_tokens $'\n\n#U:\nPlease summarize the following:\nHi Bob, how was your weekend?\n#B:\n' \
    --init_from "scratch" \
    --compile

echo -e "${GREEN}[DONE]${RESET} hf_letters_punct_dialogsum_demo.sh finished."

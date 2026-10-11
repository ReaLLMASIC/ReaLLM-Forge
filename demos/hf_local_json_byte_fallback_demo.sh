#!/bin/bash
# demos/hf_local_json_byte_fallback_demo.sh
#
# End-to-end demo that shows how to reuse the repo's HuggingFace tokenizer
# pipeline with a *local* JSON vocabulary while keeping byte fallback.
#
# Pipeline:
#   1. Obtain the dialogsum dataset (input.txt) if missing.
#   2. Author a small JSON list of custom tokens (the "local JSON").
#   3. Use the `tokenizers` Python library (installed transitively by
#      `transformers`) to materialize a local HF tokenizer directory
#      (tokenizer.json + tokenizer_config.json) with byte_fallback=true.
#   4. Run `prepare.py --method huggingface --hf_tokenizer_name <local_dir>`
#      to tokenize train.bin / val.bin and drop a snapshot in meta.pkl.
#   5. Kick off `train.py --dataset dialogsum` with default settings so the
#      HuggingFace tokenizer path is exercised end-to-end.

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
TOK_DIR="${DATA_DIR}/hf_local_tok"
JSON_VOCAB="${DATA_DIR}/hf_local_tokens.json"

if [ ! -d "${DATA_DIR}" ]; then
  echo -e "${RED}[ERROR]${RESET} Missing ${DATA_DIR}. Run this from the repo root." >&2
  exit 1
fi

# Absolute path for the tokenizer directory. prepare.py stamps this value
# into meta.pkl as `hf_tokenizer_name`, and train.py/sample.py later re-resolve
# it from the repo root. A relative path like "./hf_local_tok" would be
# interpreted as a Hub repo id at train time and blow up the regex validator
# (see HuggingFaceTokenizer.__init__ + sample.py:get_tokenizer_functions).
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
# 2) Author the local JSON vocabulary
#    This is the "format" the repo convention expects: a flat JSON array of
#    strings, exactly the same shape consumed by JsonByteTokenizerWithByteFallback.
#    Byte fallback guarantees that *any* UTF-8 input is representable, so this
#    short list is just priming the custom-piece region of the vocab.
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [2/5] Writing local JSON token list -> ${JSON_VOCAB} ===${RESET}"
cat > "${JSON_VOCAB}" <<'JSON'
[
  "<unk>",
  "<s>",
  "</s>",
  "#U:",
  "#B:",
  "Please summarize the following:",
  "▁the",
  "▁and",
  "▁to",
  "▁of",
  "▁a",
  "▁in",
  "▁is",
  "▁that",
  "▁I",
  "▁you",
  "▁it",
  "▁for",
  "▁on",
  "▁with",
  "▁as",
  "▁was",
  "▁at",
  "▁by",
  "▁be",
  "▁this",
  "▁have",
  "▁not",
  "▁are",
  "▁from",
  "▁or",
  "▁he",
  "▁she",
  "▁they",
  "▁we",
  "▁his",
  "▁her",
  "▁but",
  "▁do",
  "▁what",
  "▁so",
  "▁all",
  "▁will",
  "▁can",
  "▁if",
  "▁about",
  "▁my",
  "▁your",
  "▁me",
  "▁"
]
JSON

# ---------------------------------------------------------------------------
# 3) Build the HF tokenizer directory (tokenizer.json + tokenizer_config.json)
#    Llama-style recipe: BPE model with byte_fallback=true, <0x00>..<0xFF>
#    tokens reserved for raw-byte fallback, Metaspace-style whitespace handling,
#    ByteFallback decoder that stitches <0xNN> pieces back into UTF-8.
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [3/5] Building local HF tokenizer directory -> ${TOK_DIR_ABS} ===${RESET}"

python3 - "$JSON_VOCAB" "$TOK_DIR_ABS" <<'PY'
import json
import os
import sys

from tokenizers import Tokenizer, decoders, models, normalizers

json_vocab_path, out_dir = sys.argv[1], sys.argv[2]

with open(json_vocab_path, "r", encoding="utf-8") as f:
    custom_tokens = json.load(f)
assert isinstance(custom_tokens, list), "JSON vocab must be a flat list of strings"

# --- Build the vocab in the canonical "byte_fallback" layout ---
# 1) <unk> at id 0 (required by BPE when byte_fallback=true and fuse_unk)
# 2) <s> / </s> added as special tokens later
# 3) <0x00>..<0xFF> reserved as the 256 byte-fallback pieces
# 4) user-supplied custom pieces after that
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

# Llama/Gemma-style whitespace handling: replace regular spaces with ▁ so
# piece boundaries align with word starts. Anything outside the custom vocab
# (emoji, CJK, punctuation, ...) drops to <0xNN> pieces via byte_fallback.
tok.normalizer = normalizers.Replace(" ", "\u2581")
tok.pre_tokenizer = None
tok.decoder = decoders.Sequence([
    decoders.Replace("\u2581", " "),
    decoders.ByteFallback(),
    decoders.Fuse(),
])

tok.add_special_tokens(["<unk>", "<s>", "</s>"])

os.makedirs(out_dir, exist_ok=True)
tok.save(os.path.join(out_dir, "tokenizer.json"))

# Minimal tokenizer_config.json so AutoTokenizer.from_pretrained() treats
# the directory as a fast tokenizer.
with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 1_000_000_000,
    }, f, indent=2)

print(f"[hf-local-json] wrote {out_dir}/tokenizer.json "
      f"(vocab_size={tok.get_vocab_size()}, byte_fallback=True)")

# Smoke-test the round-trip so CI-style failures show up immediately.
probe = "Hello 👋! #U: Please summarize the following:\n안녕하세요 — Привет."
enc = tok.encode(probe)
dec = tok.decode(enc.ids)
print(f"[hf-local-json] probe len(ids)={len(enc.ids)} round-trip={dec!r}")
PY

# ---------------------------------------------------------------------------
# 4) Tokenize via the existing HuggingFace pipeline in prepare.py
#    Nothing special here — --hf_tokenizer_name accepts the local dir directly
#    (data/template/nanogpt_tokenizers.py:300, AutoTokenizer.from_pretrained).
# ---------------------------------------------------------------------------
echo -e "${CYAN}=== [4/5] Tokenizing dialogsum via --method huggingface ===${RESET}"
pushd "${DATA_DIR}" > /dev/null

# Detect stale artifacts: if meta.pkl was produced with a different
# hf_tokenizer_name (e.g. a relative path from a previous run) we MUST
# re-tokenize so the stored path matches TOK_DIR_ABS.
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
# 5) Train with defaults on dialogsum. The HF tokenizer snapshot written by
#    prepare.py at data/dialogsum/hf_tokenizer/ makes sample.py reloadable
#    without network access (sample.py:1191).
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

echo -e "${GREEN}[DONE]${RESET} hf_local_json_byte_fallback_demo.sh finished."

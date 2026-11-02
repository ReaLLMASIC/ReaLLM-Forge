#!/usr/bin/env bash
# Reuse the CJK splits, IPA conversion, training, metrics and summary pipeline.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
export COMPARISON_TOKENIZER="${COMPARISON_TOKENIZER:-char_byte_fallback}"
export FALLBACK_CHAR_LIMIT="${FALLBACK_CHAR_LIMIT:-7936}"
export CJK_DATA_ROOT="${CJK_DATA_ROOT:-${ROOT_DIR}/data/cjk_ipa_${COMPARISON_TOKENIZER}}"
export CJK_OUT_ROOT="${CJK_OUT_ROOT:-${ROOT_DIR}/out/cjk_ipa_${COMPARISON_TOKENIZER}_12l_3epochs}"
export CJK_LOG_ROOT="${CJK_LOG_ROOT:-${ROOT_DIR}/logs/cjk_ipa_${COMPARISON_TOKENIZER}_12l_3epochs}"

source "${SCRIPT_DIR}/cjk_ipa_charbpe_compare.sh"

usage() {
  cat <<'EOF'
Usage: bash demos/cjk_ipa_char_byte_fallback_compare.sh [prepare|train|all|download|download-train]

download        Fetch and verify selected tokenized datasets from Hugging Face.
download-train  Fetch prepared data and train directly, without IPA/tokenization.
The published package uses K=7936 and includes char / char_byte_fallback.
HF_DATASET_REPO, HF_DATASET_REVISION, HF_PREPARED_SUBDIR select the package.

Compare char with char+byte fallback on Japanese, Korean and Chinese,
in both original and IPA text (12 runs). Shared training defaults: 3 epochs,
12 layers, 12 heads, embedding 768, MLP 3072, batch 16, block 1024,
QK normalization and scale enabled, torch.compile enabled, TensorBoard disabled.

PYTHON_BIN         Python executable (default: python3 on PATH).
JA_TEXT, KO_TEXT, ZH_CN_TEXT  Optional local paths; default: download xinyixuu/ted-2020.
FALLBACK_CHAR_LIMIT  Keep the K most frequent non-whitespace training characters
                    (default: 7936, total vocabulary at most 8192 including bytes).
FALLBACK_CHARS_FILE  Optional fixed single-character-per-line whitelist;
                    overrides automatic frequency selection for all six streams.
COMPARISON_TOKENIZER  char_byte_fallback (default) or byte (pure 256-token bytes).
RUN_FILTER          Regex matching run names (e.g. ^ja_original_).
CJK_DATA_ROOT, CJK_OUT_ROOT, CJK_LOG_ROOT  Independent artifact directories.
Other training overrides are identical to cjk_ipa_charbpe_compare.sh.

All runs record BPB and avg_target_probability / avg_target_rank /
avg_left_probability in their per-token CSV, summary, HTML and PNG reports.
Use a new output/log directory when changing K, whitelists or training settings.
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi

#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

HF_DATASET_REPO="${HF_DATASET_REPO:-xinyixuu/ted-2020}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-main}"
HF_DATASET_DIR="${HF_DATASET_DIR:-${ROOT_DIR}/data/ted-2020}"
# Empty overrides select the Hugging Face files; explicit paths work offline.
JA_TEXT="${JA_TEXT:-}"
KO_TEXT="${KO_TEXT:-}"
ZH_CN_TEXT="${ZH_CN_TEXT:-}"

DATA_ROOT="${CJK_DATA_ROOT:-${ROOT_DIR}/data/cjk_ipa_charbpe}"
SOURCE_ROOT="${DATA_ROOT}/_source"
OUT_ROOT="${CJK_OUT_ROOT:-${ROOT_DIR}/out/cjk_ipa_charbpe_12l_3epochs}"
LOG_ROOT="${CJK_LOG_ROOT:-${ROOT_DIR}/logs/cjk_ipa_charbpe_12l_3epochs}"

BPE_VOCAB_SIZE="${BPE_VOCAB_SIZE:-8192}"
KOREAN_WORKERS="${KOREAN_WORKERS:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_ITERS="${EVAL_ITERS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
MLP_SIZE="${MLP_SIZE:-$((4 * N_EMBD))}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-1337}"
DTYPE="${DTYPE:-bfloat16}"
RUN_FILTER="${RUN_FILTER:-}"
PER_TOKEN_METRICS_SCHEMA="target_probability_rank_left_v1"
COMPARISON_TOKENIZER="${COMPARISON_TOKENIZER:-char_bpe}"
FALLBACK_CHAR_LIMIT="${FALLBACK_CHAR_LIMIT:-256}"
FALLBACK_CHARS_FILE="${FALLBACK_CHARS_FILE:-}"
FALLBACK_VOCAB_SCRIPT="${ROOT_DIR}/data/template/utils/build_char_fallback_vocab.py"

PREPARE_SCRIPT="${ROOT_DIR}/data/template/prepare.py"
TOKENIZER_SCRIPT="${ROOT_DIR}/data/template/nanogpt_tokenizers.py"
JA_IPA_SCRIPT="${ROOT_DIR}/data/template/utils/ja2ipa.py"
KO_IPA_SCRIPT="${ROOT_DIR}/data/template/utils/espeak2ipa.py"
ZH_IPA_SCRIPT="${ROOT_DIR}/data/template/utils/zh_to_ipa.py"

RUN_SPECS=(
  "ja|original|char"
  "ja|original|char_bpe"
  "ja|ipa|char"
  "ja|ipa|char_bpe"
  "ko|original|char"
  "ko|original|char_bpe"
  "ko|ipa|char"
  "ko|ipa|char_bpe"
  "zh_cn|original|char"
  "zh_cn|original|char_bpe"
  "zh_cn|ipa|char"
  "zh_cn|ipa|char_bpe"
)

if [[ "${COMPARISON_TOKENIZER}" != char_bpe ]]; then
  for spec_index in "${!RUN_SPECS[@]}"; do
    RUN_SPECS[spec_index]="${RUN_SPECS[spec_index]/char_bpe/${COMPARISON_TOKENIZER}}"
  done
fi

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  log "FAILED at line ${BASH_LINENO[0]} (exit ${exit_code})." >&2
  exit "${exit_code}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash demos/cjk_ipa_charbpe_compare.sh [prepare|train|all]

Stages:
  prepare  Normalize/split source text, create IPA text, and tokenize 12 datasets.
  train    Train all selected datasets sequentially on one visible GPU.
  all      Run prepare followed by train (default).

Useful environment overrides:
  PYTHON_BIN, JA_TEXT, KO_TEXT, ZH_CN_TEXT
  CJK_DATA_ROOT, CJK_OUT_ROOT, CJK_LOG_ROOT
  HF_DATASET_REPO, HF_DATASET_REVISION, HF_DATASET_DIR
  BPE_VOCAB_SIZE, KOREAN_WORKERS, MAX_EPOCHS, EVAL_INTERVAL, EVAL_ITERS
  BATCH_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, MLP_SIZE, LEARNING_RATE, SEED, DTYPE
  RUN_FILTER  Bash regular expression matched against names such as ja_original_char_bpe.

Source defaults to xinyixuu/ted-2020 (ja.txt, ko.txt, zh_cn.txt).
Training defaults: 3 epochs, 12 layers/heads, embedding 768, MLP 3072,
batch 16, block 1024, QK norm + scale and compile enabled, TensorBoard disabled.

The default charBPE path intentionally uses nanoGPT's existing Python BPE trainer.
On the full CJK corpora, vocabulary training can take many hours or longer.
Per-token reports include average target probability, target rank, and left
probability. Existing runs without these fields need a fresh CJK_OUT_ROOT so
that the complete metric history can be collected during retraining.
EOF
}

require_nonempty_file() {
  local path=$1
  [[ -s "${path}" ]] || die "Expected a non-empty file: ${path}"
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

write_marker() {
  local path=$1
  local value=$2
  local temporary="${path}.tmp.$$"
  printf '%s\n' "${value}" >"${temporary}"
  mv -f "${temporary}" "${path}"
}

marker_matches() {
  local path=$1
  local expected=$2
  [[ -f "${path}" ]] && [[ "$(<"${path}")" == "${expected}" ]]
}

validate_positive_integer() {
  local name=$1
  local value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )) \
    || die "${name} must be a positive integer, got '${value}'."
}

validate_configuration() {
  case "${MODE}" in
    prepare|train|all) ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown stage '${MODE}'."
      ;;
  esac

  [[ -x "${PYTHON_BIN}" ]] || die "Python interpreter is not executable: ${PYTHON_BIN}"

  validate_positive_integer BPE_VOCAB_SIZE "${BPE_VOCAB_SIZE}"
  validate_positive_integer KOREAN_WORKERS "${KOREAN_WORKERS}"
  "${PYTHON_BIN}" -c 'import math,sys; n=float(sys.argv[1]); sys.exit(0 if math.isfinite(n) and n > 0 else "MAX_EPOCHS must be positive and finite")' "${MAX_EPOCHS}"
  validate_positive_integer EVAL_INTERVAL "${EVAL_INTERVAL}"
  validate_positive_integer EVAL_ITERS "${EVAL_ITERS}"
  validate_positive_integer BATCH_SIZE "${BATCH_SIZE}"
  validate_positive_integer BLOCK_SIZE "${BLOCK_SIZE}"
  validate_positive_integer N_LAYER "${N_LAYER}"
  validate_positive_integer N_HEAD "${N_HEAD}"
  validate_positive_integer N_EMBD "${N_EMBD}"
  validate_positive_integer MLP_SIZE "${MLP_SIZE}"
  validate_positive_integer SEED "${SEED}"
  case "${COMPARISON_TOKENIZER}" in
    char_bpe|byte) ;;
    char_byte_fallback)
      validate_positive_integer FALLBACK_CHAR_LIMIT "${FALLBACK_CHAR_LIMIT}"
      require_nonempty_file "${FALLBACK_VOCAB_SCRIPT}"
      if [[ -n "${FALLBACK_CHARS_FILE}" ]]; then
        require_nonempty_file "${FALLBACK_CHARS_FILE}"
        FALLBACK_CHARS_FILE="$(realpath "${FALLBACK_CHARS_FILE}")"
      fi
      ;;
    *) die "Unsupported COMPARISON_TOKENIZER: ${COMPARISON_TOKENIZER}" ;;
  esac

  (( BPE_VOCAB_SIZE > 256 )) \
    || die "BPE_VOCAB_SIZE must exceed 256 because IDs 0..255 are byte fallback tokens."
  (( N_EMBD % N_HEAD == 0 )) \
    || die "N_EMBD (${N_EMBD}) must be divisible by N_HEAD (${N_HEAD})."

  case "${DTYPE}" in
    bfloat16|float16|float32) ;;
    *) die "DTYPE must be bfloat16, float16, or float32; got '${DTYPE}'." ;;
  esac

  require_nonempty_file "${PREPARE_SCRIPT}"
  require_nonempty_file "${TOKENIZER_SCRIPT}"
  require_nonempty_file "${JA_IPA_SCRIPT}"
  require_nonempty_file "${KO_IPA_SCRIPT}"
  require_nonempty_file "${ZH_IPA_SCRIPT}"
}

selected_run() {
  local run_name=$1
  [[ -z "${RUN_FILTER}" || "${run_name}" =~ ${RUN_FILTER} ]]
}

source_for_language() {
  case "$1" in
    ja) printf '%s\n' "${JA_TEXT}" ;;
    ko) printf '%s\n' "${KO_TEXT}" ;;
    zh_cn) printf '%s\n' "${ZH_CN_TEXT}" ;;
    *) die "Unsupported language '$1'." ;;
  esac
}

download_source_texts() {
  local files=()
  [[ -n "${JA_TEXT}" ]] || files+=(ja.txt)
  [[ -n "${KO_TEXT}" ]] || files+=(ko.txt)
  [[ -n "${ZH_CN_TEXT}" ]] || files+=(zh_cn.txt)
  if (( ${#files[@]} )); then
    log "Downloading ${HF_DATASET_REPO}@${HF_DATASET_REVISION}: ${files[*]}"
    "${PYTHON_BIN}" "${ROOT_DIR}/data/template/utils/download_cjk_ted.py" \
      --repo-id "${HF_DATASET_REPO}" --revision "${HF_DATASET_REVISION}" \
      --local-dir "${HF_DATASET_DIR}" --files "${files[@]}"
  fi
  JA_TEXT="${JA_TEXT:-${HF_DATASET_DIR}/ja.txt}"
  KO_TEXT="${KO_TEXT:-${HF_DATASET_DIR}/ko.txt}"
  ZH_CN_TEXT="${ZH_CN_TEXT:-${HF_DATASET_DIR}/zh_cn.txt}"
}

normalize_and_split() {
  local language=$1
  local input_file
  input_file="$(source_for_language "${language}")"
  require_nonempty_file "${input_file}"

  local normalized="${SOURCE_ROOT}/${language}.original.normalized.txt"
  local train_file="${SOURCE_ROOT}/${language}.train.original.txt"
  local val_file="${SOURCE_ROOT}/${language}.val.original.txt"
  local marker="${SOURCE_ROOT}/${language}.split.sha256"
  local fingerprint
  fingerprint="$(hash_file "${input_file}")"

  if marker_matches "${marker}" "${fingerprint}" \
      && [[ -s "${normalized}" && -s "${train_file}" && -s "${val_file}" ]]; then
    log "Split is current for ${language}; skipping."
    return
  fi

  log "Normalizing and splitting ${language}: ${input_file}"
  local normalized_tmp="${normalized}.tmp.$$"
  local train_tmp="${train_file}.tmp.$$"
  local val_tmp="${val_file}.tmp.$$"

  awk 'NF { print }' "${input_file}" >"${normalized_tmp}"

  local total_lines
  total_lines="$(wc -l <"${normalized_tmp}")"
  (( total_lines >= 10 )) \
    || die "${language} has only ${total_lines} non-empty lines; at least 10 are required."

  local train_lines=$(( total_lines * 9 / 10 ))
  head -n "${train_lines}" "${normalized_tmp}" >"${train_tmp}"
  tail -n "+$(( train_lines + 1 ))" "${normalized_tmp}" >"${val_tmp}"

  require_nonempty_file "${train_tmp}"
  require_nonempty_file "${val_tmp}"
  mv -f "${normalized_tmp}" "${normalized}"
  mv -f "${train_tmp}" "${train_file}"
  mv -f "${val_tmp}" "${val_file}"
  write_marker "${marker}" "${fingerprint}"

  log "${language}: ${train_lines} train lines, $(( total_lines - train_lines )) validation lines."
}

ipa_fingerprint() {
  local source_file=$1
  local converter_file=$2
  local converter_options=$3
  {
    printf '%s\n' "${converter_options}"
    sha256sum "${source_file}" "${converter_file}"
  } | sha256sum | awk '{print $1}'
}

convert_ipa_split() {
  local language=$1
  local split=$2
  local source_file="${SOURCE_ROOT}/${language}.${split}.original.txt"
  local output_file="${SOURCE_ROOT}/${language}.${split}.ipa.txt"
  local stats_file="${SOURCE_ROOT}/${language}.${split}.ipa.stats.json"
  local marker="${SOURCE_ROOT}/${language}.${split}.ipa.sha256"
  local converter_file
  local converter_options

  case "${language}" in
    ja)
      converter_file="${JA_IPA_SCRIPT}"
      converter_options="unspaced_ipa/no_mecab"
      ;;
    ko)
      converter_file="${KO_IPA_SCRIPT}"
      converter_options="espeak-ng/ko/no-wrapper/workers=${KOREAN_WORKERS}"
      ;;
    zh_cn)
      converter_file="${ZH_IPA_SCRIPT}"
      converter_options="dragonmapper/text/no-wrapper"
      ;;
    *) die "Unsupported language '${language}'." ;;
  esac

  require_nonempty_file "${source_file}"
  local fingerprint
  fingerprint="$(ipa_fingerprint "${source_file}" "${converter_file}" "${converter_options}")"

  if marker_matches "${marker}" "${fingerprint}" \
      && [[ -s "${output_file}" && -s "${stats_file}" ]]; then
    log "IPA output is current for ${language}/${split}; skipping."
    return
  fi

  log "Converting ${language}/${split} to IPA."
  local output_tmp="${output_file}.tmp.$$"
  local stats_tmp="${stats_file}.tmp.$$"

  case "${language}" in
    ja)
      "${PYTHON_BIN}" "${JA_IPA_SCRIPT}" \
        "${source_file}" "${output_tmp}" \
        --text_output \
        --text_no_sentence \
        --text_field unspaced_ipa \
        --stats_json "${stats_tmp}"
      ;;
    ko)
      command -v espeak-ng >/dev/null 2>&1 \
        || die "espeak-ng is required for Korean IPA conversion."
      "${PYTHON_BIN}" "${KO_IPA_SCRIPT}" "${source_file}" \
        --lang ko \
        --mode text \
        --output_file "${output_tmp}" \
        --no-wrapper \
        --multithread \
        --workers "${KOREAN_WORKERS}" \
        --stats_json "${stats_tmp}"
      ;;
    zh_cn)
      # zh_to_ipa prints every converted line. Keep only its final diagnostics in
      # the driver log; the complete IPA text is written to output_tmp.
      "${PYTHON_BIN}" "${ZH_IPA_SCRIPT}" \
        "${source_file}" "${output_tmp}" \
        --input_type text \
        --no-wrapper \
        --stats_json "${stats_tmp}" 2>&1 | tail -n 20
      ;;
  esac

  require_nonempty_file "${output_tmp}"
  require_nonempty_file "${stats_tmp}"
  mv -f "${output_tmp}" "${output_file}"
  mv -f "${stats_tmp}" "${stats_file}"
  write_marker "${marker}" "${fingerprint}"
  log "Finished IPA conversion for ${language}/${split}."
}

dataset_fingerprint() {
  local train_file=$1
  local val_file=$2
  local method=$3
  {
    printf '%s\n' "method=${method}" "bpe_vocab_size=${BPE_VOCAB_SIZE}" \
      "complete_char_coverage=true" "track_token_counts=true"
    sha256sum "${train_file}" "${val_file}" "${PREPARE_SCRIPT}" "${TOKENIZER_SCRIPT}"
    if [[ "${method}" == char_byte_fallback ]]; then
      printf '%s\n' "fallback_char_limit=${FALLBACK_CHAR_LIMIT}"
      sha256sum "${FALLBACK_VOCAB_SCRIPT}"
      if [[ -n "${FALLBACK_CHARS_FILE}" ]]; then
        sha256sum "${FALLBACK_CHARS_FILE}"
      fi
    fi
  } | sha256sum | awk '{print $1}'
}

validate_dataset_meta() {
  local meta_file=$1
  local method=$2
  "${PYTHON_BIN}" -c '
import pickle
import sys

path, method, expected_text_vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, "rb") as handle:
    meta = pickle.load(handle)
vocab_size = int(meta.get("vocab_size", 0))
if vocab_size <= 0:
    raise SystemExit(f"invalid vocab_size={vocab_size} in {path}")
if method == "char_bpe":
    if meta.get("tokenizer") != "char_bpe":
        raise SystemExit(f"{path} is not char_bpe metadata")
    if vocab_size != expected_text_vocab:
        raise SystemExit(
            f"expected char_bpe vocab_size={expected_text_vocab}, got {vocab_size} in {path}"
        )
    if meta.get("byte_fallback") is not True:
        raise SystemExit(f"byte fallback is not enabled in {path}")
elif method == "byte":
    if meta.get("tokenizer") != "byte" or vocab_size != 256:
        raise SystemExit(f"{path} is not a 256-token byte vocabulary")
elif method == "char_byte_fallback":
    chars = meta.get("custom_chars", [])
    if (meta.get("tokenizer") != "custom_char_with_byte_fallback"
            or vocab_size != 256 + len(chars)
            or not chars or any(len(char) != 1 for char in chars)):
        raise SystemExit(f"{path} is not a single-character byte-fallback vocabulary")
    if any(meta["itos"].get(i) != bytes([i]) for i in range(256)):
        raise SystemExit(f"{path} is missing byte fallback IDs")
elif "chars" not in meta:
    raise SystemExit(f"{path} is missing the character vocabulary")
' "${meta_file}" "${method}" "${BPE_VOCAB_SIZE}"
}

dataset_is_current() {
  local dataset_dir=$1
  local method=$2
  local fingerprint=$3
  local marker="${dataset_dir}/.prepared.sha256"
  marker_matches "${marker}" "${fingerprint}" \
    && [[ -s "${dataset_dir}/train.bin" ]] \
    && [[ -s "${dataset_dir}/val.bin" ]] \
    && [[ -s "${dataset_dir}/meta.pkl" ]] \
    && validate_dataset_meta "${dataset_dir}/meta.pkl" "${method}"
}

prepare_dataset() {
  local language=$1
  local representation=$2
  local tokenizer=$3
  local run_name="${language}_${representation}_${tokenizer}"

  selected_run "${run_name}" || return 0

  local train_file="${SOURCE_ROOT}/${language}.train.${representation}.txt"
  local val_file="${SOURCE_ROOT}/${language}.val.${representation}.txt"
  local dataset_dir="${DATA_ROOT}/${run_name}"
  local method="${tokenizer}"
  require_nonempty_file "${train_file}"
  require_nonempty_file "${val_file}"

  local fingerprint
  fingerprint="$(dataset_fingerprint "${train_file}" "${val_file}" "${method}")"
  if dataset_is_current "${dataset_dir}" "${method}" "${fingerprint}"; then
    log "Dataset ${run_name} is current; skipping tokenization."
    return
  fi

  mkdir -p "${dataset_dir}"
  log "Tokenizing ${run_name}."
  if [[ "${method}" == "char_bpe" ]]; then
    log "Using the legacy full-corpus charBPE trainer; this step may take many hours."
  fi

  (
    cd "${dataset_dir}"
    if [[ "${method}" == "char" ]]; then
      "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
        --train_input "${train_file}" \
        --val_input "${val_file}" \
        --method char \
        --track_token_counts
    elif [[ "${method}" == char_bpe ]]; then
      "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
        --train_input "${train_file}" \
        --val_input "${val_file}" \
        --method char_bpe \
        --vocab_size "${BPE_VOCAB_SIZE}" \
        --no-char_bpe_incomplete_coverage_uses_bpe \
        --track_token_counts
    elif [[ "${method}" == byte ]]; then
      "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
        --train_input "${train_file}" \
        --val_input "${val_file}" \
        --method byte \
        --track_token_counts
    else
      local vocab_args=(--train_input "${train_file}" --limit "${FALLBACK_CHAR_LIMIT}")
      if [[ -n "${FALLBACK_CHARS_FILE}" ]]; then
        vocab_args+=(--characters_file "${FALLBACK_CHARS_FILE}")
      fi
      "${PYTHON_BIN}" "${FALLBACK_VOCAB_SCRIPT}" "${vocab_args[@]}" \
        --output custom_chars.txt
      "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
        --train_input "${train_file}" \
        --val_input "${val_file}" \
        --method custom_char_byte_fallback \
        --custom_chars_file custom_chars.txt \
        --track_token_counts
    fi
  )

  require_nonempty_file "${dataset_dir}/train.bin"
  require_nonempty_file "${dataset_dir}/val.bin"
  require_nonempty_file "${dataset_dir}/meta.pkl"
  validate_dataset_meta "${dataset_dir}/meta.pkl" "${method}"
  write_marker "${dataset_dir}/.prepared.sha256" "${fingerprint}"
  log "Dataset ${run_name} is ready."
}

prepare_all() {
  mkdir -p "${SOURCE_ROOT}" "${DATA_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}"

  "${PYTHON_BIN}" -c 'import numpy, tqdm, pykakasi, konlpy, dragonmapper, jieba' >/dev/null
  download_source_texts

  local language
  for language in ja ko zh_cn; do
    normalize_and_split "${language}"
  done
  for language in ja ko zh_cn; do
    convert_ipa_split "${language}" train
    convert_ipa_split "${language}" val
  done

  local spec representation tokenizer
  for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r language representation tokenizer <<<"${spec}"
    prepare_dataset "${language}" "${representation}" "${tokenizer}"
  done

  log "All selected datasets are prepared."
}

assert_dataset_ready_for_training() {
  local run_name=$1
  local method=$2
  local dataset_dir="${DATA_ROOT}/${run_name}"
  require_nonempty_file "${dataset_dir}/train.bin"
  require_nonempty_file "${dataset_dir}/val.bin"
  require_nonempty_file "${dataset_dir}/meta.pkl"
  validate_dataset_meta "${dataset_dir}/meta.pkl" "${method}"
}

dataset_argument_for_run() {
  local run_name=$1
  local dataset_dir="${DATA_ROOT}/${run_name}"
  local repository_data_root="${ROOT_DIR}/data/"
  if [[ "${dataset_dir}" == "${repository_data_root}"* ]]; then
    printf '%s\n' "${dataset_dir#${repository_data_root}}"
  else
    # train.py joins dataset arguments beneath data/. os.path.join correctly
    # preserves an absolute component, which makes isolated smoke roots work.
    printf '%s\n' "${dataset_dir}"
  fi
}

validate_training_outputs() {
  local out_dir=$1
  local metrics_dir="${out_dir}/per_token_metrics"
  require_nonempty_file "${out_dir}/best_val_loss_and_iter.txt"
  require_nonempty_file "${metrics_dir}/per_token_metrics.csv"
  require_nonempty_file "${metrics_dir}/per_token_summary.csv"
  require_nonempty_file "${metrics_dir}/per_token_metrics.html"
  require_nonempty_file "${metrics_dir}/per_token_target_probability.html"
  require_nonempty_file "${metrics_dir}/per_token_target_rank.html"
  require_nonempty_file "${metrics_dir}/per_token_left_probability.html"
  require_nonempty_file "${metrics_dir}/per_token_target_probability_by_iteration.html"
  require_nonempty_file "${metrics_dir}/per_token_target_rank_by_iteration.html"
  require_nonempty_file "${metrics_dir}/per_token_left_probability_by_iteration.html"

  "${PYTHON_BIN}" -c '
import csv
import math
import sys

detail_path, summary_path = sys.argv[1:]
required_fields = {
    "avg_target_probability", "avg_target_rank", "avg_left_probability",
}
with open(detail_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    missing = required_fields.difference(reader.fieldnames or ())
    if missing:
        raise SystemExit(f"{detail_path} is missing metric columns: {sorted(missing)}")
    rows = list(reader)
if not rows:
    raise SystemExit(f"{detail_path} contains no metric rows")
latest_iteration = max(int(row["iteration"]) for row in rows)
latest = [row for row in rows if int(row["iteration"]) == latest_iteration]
for field in required_fields:
    if not any(math.isfinite(float(row[field])) for row in latest):
        raise SystemExit(
            f"{detail_path} has no finite {field} values at iteration {latest_iteration}"
        )

with open(summary_path, newline="", encoding="utf-8") as handle:
    summary_rows = list(csv.DictReader(handle))
available = {
    row["metric"] for row in summary_rows
    if int(row["iteration"]) == latest_iteration
}
missing = required_fields.difference(available)
if missing:
    raise SystemExit(
        f"{summary_path} is missing latest-iteration summaries: {sorted(missing)}"
    )
' "${metrics_dir}/per_token_metrics.csv" "${metrics_dir}/per_token_summary.csv"
}

write_training_token_stats() {
  local out_dir=$1 stats iteration processed unique vocab_size
  stats="$("${PYTHON_BIN}" "${ROOT_DIR}/utils/cjk_comparison_metrics.py" \
    "${out_dir}/per_token_metrics/per_token_metrics.csv" --training-stats \
    --output-json "${out_dir}/training_token_stats.json.tmp")"
  mv -f "${out_dir}/training_token_stats.json.tmp" "${out_dir}/training_token_stats.json"
  IFS=$'\t' read -r iteration processed unique <<<"${stats}"
  vocab_size="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["vocab_size"])' "${out_dir}/training_token_stats.json")"
  log "${out_dir##*/}: vocab_size=${vocab_size}, training_unique_token_ids=${unique}, training_tokens_processed=${processed}, snapshot_iteration=${iteration}"
}

train_one() {
  local language=$1
  local representation=$2
  local tokenizer=$3
  local run_name="${language}_${representation}_${tokenizer}"

  selected_run "${run_name}" || return 0

  local dataset_name
  dataset_name="$(dataset_argument_for_run "${run_name}")"
  local out_dir="${OUT_ROOT}/${run_name}"
  local complete_marker="${out_dir}/.complete"

  assert_dataset_ready_for_training "${run_name}" "${tokenizer}"

  if [[ -f "${complete_marker}" ]]; then
    grep -Fxq "per_token_metrics_schema=${PER_TOKEN_METRICS_SCHEMA}" "${complete_marker}" \
      || die "Run ${run_name} predates ${PER_TOKEN_METRICS_SCHEMA}. Use a fresh CJK_OUT_ROOT to collect the complete metric history."
    validate_training_outputs "${out_dir}"
    write_training_token_stats "${out_dir}"
    log "Run ${run_name} is already complete; skipping."
    return
  fi

  if [[ -d "${out_dir}" ]] \
      && [[ -n "$(find "${out_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "Run ${run_name} has partial output in ${out_dir}. Move it aside before retrying; automatic resume would reset cumulative per-token counts."
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}/csv"
  log "Starting training run ${run_name}."

  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" train.py \
      --training_mode single \
      --dataset "${dataset_name}" \
      --out_dir "${out_dir}" \
      --init_from scratch \
      --training_limiters max_epochs \
      --max_epochs "${MAX_EPOCHS}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --eval_iters "${EVAL_ITERS}" \
      --batch_size "${BATCH_SIZE}" \
      --gradient_accumulation_steps 1 \
      --block_size "${BLOCK_SIZE}" \
      --n_layer "${N_LAYER}" \
      --n_head "${N_HEAD}" \
      --n_embd "${N_EMBD}" \
      --mlp_size "${MLP_SIZE}" \
      --use_qk_norm \
      --use_qk_norm_scale \
      --dropout 0.0 \
      --learning_rate "${LEARNING_RATE}" \
      --seed "${SEED}" \
      --device cuda:0 \
      --dtype "${DTYPE}" \
      --compile \
      --always_save_checkpoint \
      --no-tensorboard_log \
      --csv_log \
      --csv_dir "${LOG_ROOT}/csv" \
      --csv_name "${run_name}" \
      --log_bits_per_byte \
      --log_per_token_metrics \
      --per_token_metrics_dir "${out_dir}/per_token_metrics"
  )

  validate_training_outputs "${out_dir}"
  write_training_token_stats "${out_dir}"
  printf 'completed_at=%s\nper_token_metrics_schema=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${PER_TOKEN_METRICS_SCHEMA}" \
    >"${complete_marker}"
  log "Completed training run ${run_name}."
}

write_comparison_summary() {
  local summary_file="${OUT_ROOT}/comparison_summary.tsv"
  local temporary="${summary_file}.tmp.$$"
  printf 'run\tlanguage\trepresentation\ttokenizer\tvocab_size\tbest_val_loss\tbest_bits_per_byte\tbest_iter\tbest_tokens\tval_avg_target_probability\tval_avg_target_rank\tval_avg_left_probability\ttraining_tokens_iteration\ttraining_tokens_processed\ttraining_unique_token_ids\treport_html\n' >"${temporary}"

  local spec language representation tokenizer run_name dataset_dir out_dir
  local vocab_size best_val_loss best_bits_per_byte best_iter best_tokens
  local metric_values training_stats
  for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r language representation tokenizer <<<"${spec}"
    run_name="${language}_${representation}_${tokenizer}"
    selected_run "${run_name}" || continue
    dataset_dir="${DATA_ROOT}/${run_name}"
    out_dir="${OUT_ROOT}/${run_name}"
    [[ -f "${out_dir}/.complete" ]] || continue

    vocab_size="$("${PYTHON_BIN}" -c 'import pickle,sys; print(pickle.load(open(sys.argv[1], "rb"))["vocab_size"])' "${dataset_dir}/meta.pkl")"
    IFS=',' read -r best_val_loss best_bits_per_byte best_iter best_tokens _ \
      <"${out_dir}/best_val_loss_and_iter.txt"
    best_val_loss="${best_val_loss//[[:space:]]/}"
    best_bits_per_byte="${best_bits_per_byte//[[:space:]]/}"
    best_iter="${best_iter//[[:space:]]/}"
    best_tokens="${best_tokens//[[:space:]]/}"
    metric_values="$("${PYTHON_BIN}" "${ROOT_DIR}/utils/cjk_comparison_metrics.py" \
      "${out_dir}/per_token_metrics/per_token_metrics.csv" "${best_iter}")"
    training_stats="$("${PYTHON_BIN}" "${ROOT_DIR}/utils/cjk_comparison_metrics.py" \
      "${out_dir}/per_token_metrics/per_token_metrics.csv" --training-stats)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${run_name}" "${language}" "${representation}" "${tokenizer}" \
      "${vocab_size}" "${best_val_loss}" "${best_bits_per_byte}" \
      "${best_iter}" "${best_tokens}" "${metric_values}" "${training_stats}" \
      "${out_dir}/per_token_metrics/per_token_metrics.html" >>"${temporary}"
  done

  mv -f "${temporary}" "${summary_file}"
  log "Wrote comparison summary to ${summary_file}."
}

train_all() {
  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/csv"
  "${PYTHON_BIN}" -c '
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if sys.argv[1] == "bfloat16" and not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU does not support bfloat16")
' "${DTYPE}" >/dev/null

  local spec language representation tokenizer
  for spec in "${RUN_SPECS[@]}"; do
    IFS='|' read -r language representation tokenizer <<<"${spec}"
    train_one "${language}" "${representation}" "${tokenizer}"
  done

  write_comparison_summary
  log "All selected training runs are complete."
}

main() {
  validate_configuration
  cd "${ROOT_DIR}"

  case "${MODE}" in
    prepare) prepare_all ;;
    train) train_all ;;
    all)
      prepare_all
      train_all
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi

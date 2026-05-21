#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="$ROOT_DIR/data"
INPUT_FILE=""
DECOMP_SOURCE="cjkvi"
NORMALIZATION="conservative"
SERIALIZATION_MODE="decomp-normalized"
CLEAN=0
SKIP_SELF_TESTS=0

usage() {
  cat <<'USAGE'
Usage: bash demo.sh [options]

Walk through the full pipeline step by step:
  1) optional self-tests
  2) download Unicode/Unihan/IDS source files
  3) build data/han_main_block.jsonl
  4) build data/mapped.json
  5) build data/input.readable.hsym
  6) inspect the symbolic file
  7) recover data/recovered_input.txt
  8) compare recovered output against the original input
  9) recover data/serialized_view.txt

Options:
  --input PATH            Input file. Default: data/input.txt, then data/text_zho_Hans.txt
  --data-dir PATH         Data directory. Default: ./data
  --decomp-source NAME    cjkvi | chise | cjk-decomp. Default: cjkvi
  --normalization NAME    conservative | aggressive. Default: conservative
  --mode NAME             Symbolic serialization mode. Default: decomp-normalized
  --clean                 Remove generated outputs first.
  --skip-self-tests       Skip self-tests.
  --help                  Show this help.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

step() {
  local msg="$1"
  printf '\n========== %s ==========' "$msg"
  printf '\n'
}

run() {
  printf '+ ' >&2
  printf '%q ' "$@" >&2
  printf '\n' >&2
  "$@"
}

resolve_input_file() {
  if [[ -n "$INPUT_FILE" ]]; then
    [[ -f "$INPUT_FILE" ]] || die "Input file not found: $INPUT_FILE"
    return 0
  fi
  if [[ -f "$DATA_DIR/input.txt" ]]; then
    INPUT_FILE="$DATA_DIR/input.txt"
    return 0
  fi
  if [[ -f "$DATA_DIR/text_zho_Hans.txt" ]]; then
    INPUT_FILE="$DATA_DIR/text_zho_Hans.txt"
    return 0
  fi
  die "Could not find an input file. Expected data/input.txt or data/text_zho_Hans.txt, or pass --input."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || die "--input requires a value"
      INPUT_FILE="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value"
      DATA_DIR="$2"
      shift 2
      ;;
    --decomp-source)
      [[ $# -ge 2 ]] || die "--decomp-source requires a value"
      DECOMP_SOURCE="$2"
      shift 2
      ;;
    --normalization)
      [[ $# -ge 2 ]] || die "--normalization requires a value"
      NORMALIZATION="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      SERIALIZATION_MODE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-self-tests)
      SKIP_SELF_TESTS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

mkdir -p "$DATA_DIR"
resolve_input_file

DATASET_FILE="$DATA_DIR/han_main_block.jsonl"
MAP_FILE="$DATA_DIR/mapped.json"
SYMBOLIC_FILE="$DATA_DIR/input.readable.hsym"
RECOVERED_FILE="$DATA_DIR/recovered_input.txt"
SERIALIZED_VIEW_FILE="$DATA_DIR/serialized_view.txt"
COMPARE_REPORT_FILE="$DATA_DIR/readable_compare_report.json"

BUILDER_PRESENT=0
if [[ -f "$ROOT_DIR/han_main_block_decomp.py" ]]; then
  BUILDER_PRESENT=1
fi
[[ -f "$ROOT_DIR/han_file_decomp_map_transform.py" ]] || die "Missing $ROOT_DIR/han_file_decomp_map_transform.py"
[[ -f "$ROOT_DIR/han_file_symbolic_serialize.py" ]] || die "Missing $ROOT_DIR/han_file_symbolic_serialize.py"
[[ -f "$ROOT_DIR/han_file_symbolic_reverse.py" ]] || die "Missing $ROOT_DIR/han_file_symbolic_reverse.py"

if (( CLEAN )); then
  step "Cleaning previous generated outputs"
  rm -f "$DATASET_FILE" "$MAP_FILE" "$SYMBOLIC_FILE" "$RECOVERED_FILE" "$SERIALIZED_VIEW_FILE" "$COMPARE_REPORT_FILE"
  find "$DATA_DIR" -maxdepth 1 -type f -print | sort || true
fi

step "Environment"
printf 'folder:      %s\n' "$ROOT_DIR"
printf 'python:      %s\n' "$PYTHON_BIN"
printf 'data dir:    %s\n' "$DATA_DIR"
printf 'input file:  %s\n' "$INPUT_FILE"
printf 'decomp src:  %s\n' "$DECOMP_SOURCE"
printf 'normalize:   %s\n' "$NORMALIZATION"
printf 'mode:        %s\n' "$SERIALIZATION_MODE"

if (( ! SKIP_SELF_TESTS )); then
  step "Step 1: run self-tests"
  if (( BUILDER_PRESENT )); then
    run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" self-test
  else
    printf 'Skipping han_main_block_decomp.py self-test because that file is not present in this folder.\n'
  fi
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_serialize.py" self-test
fi

if (( BUILDER_PRESENT )); then
  step "Step 2: download Unicode / Unihan / IDS sources (only if missing)"
  run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" download \
    --data-dir "$DATA_DIR" \
    --decomp-source "$DECOMP_SOURCE"

  step "Step 3: build the main-block Han dataset"
  run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" build \
    --data-dir "$DATA_DIR" \
    --decomp-source "$DECOMP_SOURCE" \
    --download-missing \
    --normalization "$NORMALIZATION" \
    --output "$DATASET_FILE"

  step "Step 4: show dataset stats"
  run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" stats \
    --dataset "$DATASET_FILE"
else
  [[ -f "$DATASET_FILE" ]] || die "han_main_block_decomp.py is missing and $DATASET_FILE does not exist, so the dataset cannot be built from scratch."
  step "Steps 2-4: reuse the existing dataset"
  printf 'han_main_block_decomp.py is not present, so this demo is reusing:\n  %s\n' "$DATASET_FILE"
fi

step "Step 5: build the detailed reversible token map"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_decomp_map_transform.py" build \
  --input "$INPUT_FILE" \
  --dataset "$DATASET_FILE" \
  --output "$MAP_FILE"

step "Step 6: build the readable symbolic compact file from mapped.json"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_serialize.py" from-map \
  --map "$MAP_FILE" \
  --output "$SYMBOLIC_FILE" \
  --mode "$SERIALIZATION_MODE" \
  --include-han-metadata

step "Step 7: inspect the symbolic file"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" inspect \
  --symbolic "$SYMBOLIC_FILE" \
  --include-previews \
  --preview-chars 240

step "Step 8: preview the first lines of the symbolic file"
sed -n '1,40p' "$SYMBOLIC_FILE"

step "Step 9: recover the exact original input bytes"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" recover \
  --symbolic "$SYMBOLIC_FILE" \
  --mode original-bytes \
  --output "$RECOVERED_FILE"

step "Step 10: compare the recovered bytes to the original input"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" compare \
  --symbolic "$SYMBOLIC_FILE" \
  --mode original-bytes \
  --compare-to "$INPUT_FILE" \
  --report-json "$COMPARE_REPORT_FILE"

if cmp -s "$INPUT_FILE" "$RECOVERED_FILE"; then
  printf 'Exact byte-for-byte round trip confirmed.\n'
else
  die "Recovered file differs from the original. Inspect $COMPARE_REPORT_FILE"
fi

step "Step 11: also recover the serialized/decomposed text view"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" recover \
  --symbolic "$SYMBOLIC_FILE" \
  --mode serialized-text \
  --output "$SERIALIZED_VIEW_FILE"

step "Step 12: generated outputs"
ls -lh "$DATASET_FILE" "$MAP_FILE" "$SYMBOLIC_FILE" "$RECOVERED_FILE" "$SERIALIZED_VIEW_FILE" "$COMPARE_REPORT_FILE"

printf '\nDemo complete.\n'

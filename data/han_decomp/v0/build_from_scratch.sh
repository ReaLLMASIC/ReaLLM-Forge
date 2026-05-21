#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="$ROOT_DIR/data"
INPUT_FILE=""
DATASET_FILE=""
MAP_FILE=""
SYMBOLIC_FILE=""
RECOVERED_FILE=""
SERIALIZED_VIEW_FILE=""
COMPARE_REPORT_FILE=""
DECOMP_SOURCE="cjkvi"
NORMALIZATION="conservative"
SERIALIZATION_MODE="decomp-normalized"
FORCE_DOWNLOAD=0
FORCE_REBUILD_DATASET=0
FORCE_REBUILD_MAP=0
FORCE_REBUILD_SYMBOLIC=0
FORCE_RECOVER=0
RUN_SELF_TESTS=1
CLEAN=0

usage() {
  cat <<'USAGE'
Usage: bash build_from_scratch.sh [options]

Build the full Han decomposition + readable symbolic round-trip pipeline in the
current folder.

Default paths:
  data/input.txt            input text (falls back to data/text_zho_Hans.txt)
  data/han_main_block.jsonl built Han dataset
  data/mapped.json          detailed reversible token map
  data/input.readable.hsym  readable symbolic compact form
  data/recovered_input.txt  exact recovered original bytes
  data/serialized_view.txt  readable serialized/decomposition view
  data/readable_compare_report.json  compare report

Options:
  --input PATH                    Input text file.
  --data-dir PATH                 Data directory. Default: ./data
  --dataset PATH                  Dataset output path.
  --map PATH                      Mapped JSON output path.
  --symbolic PATH                 Readable symbolic output path.
  --recovered PATH                Recovered original output path.
  --serialized-view PATH          Serialized text view output path.
  --compare-report PATH           Compare report JSON path.
  --decomp-source NAME            cjkvi | chise | cjk-decomp. Default: cjkvi
  --normalization NAME            conservative | aggressive. Default: conservative
  --mode NAME                     Symbolic serialization mode. Default: decomp-normalized
  --force-download                Re-download source data.
  --force-rebuild-dataset         Rebuild han_main_block.jsonl.
  --force-rebuild-map             Rebuild mapped.json.
  --force-rebuild-symbolic        Rebuild input.readable.hsym.
  --force-recover                 Re-run recovery even if outputs are up to date.
  --clean                         Remove generated outputs before rebuilding.
  --no-self-tests                 Skip Python script self-tests.
  --help                          Show this help.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

note() {
  printf '[build] %s\n' "$*" >&2
}

run() {
  printf '+ ' >&2
  printf '%q ' "$@" >&2
  printf '\n' >&2
  "$@"
}

require_file() {
  local path="$1"
  local hint="${2:-}"
  [[ -f "$path" ]] || die "Missing required file: $path${hint:+ ($hint)}"
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
  die "Could not find an input text file. Expected data/input.txt or data/text_zho_Hans.txt, or pass --input."
}

needs_rebuild() {
  local target="$1"
  shift || true
  [[ ! -f "$target" ]] && return 0
  local dep
  for dep in "$@"; do
    [[ -e "$dep" ]] || continue
    [[ "$dep" -nt "$target" ]] && return 0
  done
  return 1
}

missing_downloads() {
  local downloads_dir="$1"
  local decomp_source="$2"
  local decomp_name=""
  case "$decomp_source" in
    cjkvi) decomp_name="ids.txt" ;;
    chise) decomp_name="IDS-UCS-Basic.txt" ;;
    cjk-decomp) decomp_name="cjk-decomp.txt" ;;
    *) die "Unsupported --decomp-source: $decomp_source" ;;
  esac
  local needed=(
    "$downloads_dir/UnicodeData.txt"
    "$downloads_dir/Unihan.zip"
    "$downloads_dir/CJKRadicals.txt"
    "$downloads_dir/EquivalentUnifiedIdeograph.txt"
    "$downloads_dir/$decomp_name"
  )
  local f
  for f in "${needed[@]}"; do
    [[ -f "$f" ]] || return 0
  done
  return 1
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
    --dataset)
      [[ $# -ge 2 ]] || die "--dataset requires a value"
      DATASET_FILE="$2"
      shift 2
      ;;
    --map)
      [[ $# -ge 2 ]] || die "--map requires a value"
      MAP_FILE="$2"
      shift 2
      ;;
    --symbolic)
      [[ $# -ge 2 ]] || die "--symbolic requires a value"
      SYMBOLIC_FILE="$2"
      shift 2
      ;;
    --recovered)
      [[ $# -ge 2 ]] || die "--recovered requires a value"
      RECOVERED_FILE="$2"
      shift 2
      ;;
    --serialized-view)
      [[ $# -ge 2 ]] || die "--serialized-view requires a value"
      SERIALIZED_VIEW_FILE="$2"
      shift 2
      ;;
    --compare-report)
      [[ $# -ge 2 ]] || die "--compare-report requires a value"
      COMPARE_REPORT_FILE="$2"
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
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    --force-rebuild-dataset)
      FORCE_REBUILD_DATASET=1
      shift
      ;;
    --force-rebuild-map)
      FORCE_REBUILD_MAP=1
      shift
      ;;
    --force-rebuild-symbolic)
      FORCE_REBUILD_SYMBOLIC=1
      shift
      ;;
    --force-recover)
      FORCE_RECOVER=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --no-self-tests)
      RUN_SELF_TESTS=0
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

DATASET_FILE="${DATASET_FILE:-$DATA_DIR/han_main_block.jsonl}"
MAP_FILE="${MAP_FILE:-$DATA_DIR/mapped.json}"
SYMBOLIC_FILE="${SYMBOLIC_FILE:-$DATA_DIR/input.readable.hsym}"
RECOVERED_FILE="${RECOVERED_FILE:-$DATA_DIR/recovered_input.txt}"
SERIALIZED_VIEW_FILE="${SERIALIZED_VIEW_FILE:-$DATA_DIR/serialized_view.txt}"
COMPARE_REPORT_FILE="${COMPARE_REPORT_FILE:-$DATA_DIR/readable_compare_report.json}"
DOWNLOADS_DIR="$DATA_DIR/downloads"

# Normalize to absolute paths when possible.
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")" && pwd)/$(basename "$INPUT_FILE")"
DATASET_FILE="$(mkdir -p "$(dirname "$DATASET_FILE")" && cd "$(dirname "$DATASET_FILE")" && pwd)/$(basename "$DATASET_FILE")"
MAP_FILE="$(mkdir -p "$(dirname "$MAP_FILE")" && cd "$(dirname "$MAP_FILE")" && pwd)/$(basename "$MAP_FILE")"
SYMBOLIC_FILE="$(mkdir -p "$(dirname "$SYMBOLIC_FILE")" && cd "$(dirname "$SYMBOLIC_FILE")" && pwd)/$(basename "$SYMBOLIC_FILE")"
RECOVERED_FILE="$(mkdir -p "$(dirname "$RECOVERED_FILE")" && cd "$(dirname "$RECOVERED_FILE")" && pwd)/$(basename "$RECOVERED_FILE")"
SERIALIZED_VIEW_FILE="$(mkdir -p "$(dirname "$SERIALIZED_VIEW_FILE")" && cd "$(dirname "$SERIALIZED_VIEW_FILE")" && pwd)/$(basename "$SERIALIZED_VIEW_FILE")"
COMPARE_REPORT_FILE="$(mkdir -p "$(dirname "$COMPARE_REPORT_FILE")" && cd "$(dirname "$COMPARE_REPORT_FILE")" && pwd)/$(basename "$COMPARE_REPORT_FILE")"
DOWNLOADS_DIR="$DATA_DIR/downloads"

require_file "$ROOT_DIR/han_file_decomp_map_common.py"
require_file "$ROOT_DIR/han_file_decomp_map_transform.py"
require_file "$ROOT_DIR/han_file_symbolic_common.py"
require_file "$ROOT_DIR/han_file_symbolic_serialize.py"
require_file "$ROOT_DIR/han_file_symbolic_reverse.py"

if (( CLEAN )); then
  note "Removing generated outputs."
  rm -f "$DATASET_FILE" "$MAP_FILE" "$SYMBOLIC_FILE" "$RECOVERED_FILE" "$SERIALIZED_VIEW_FILE" "$COMPARE_REPORT_FILE"
fi

if (( RUN_SELF_TESTS )); then
  if [[ -f "$ROOT_DIR/han_main_block_decomp.py" ]]; then
    note "Running han_main_block_decomp.py self-test"
    run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" self-test
  fi
  note "Running han_file_symbolic_serialize.py self-test"
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_serialize.py" self-test
fi

if (( FORCE_DOWNLOAD )); then
  require_file "$ROOT_DIR/han_main_block_decomp.py" "required to download source data"
  note "Force-downloading source data into $DOWNLOADS_DIR"
  run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" download \
    --data-dir "$DATA_DIR" \
    --decomp-source "$DECOMP_SOURCE" \
    --force
fi

if (( FORCE_REBUILD_DATASET )) || [[ ! -f "$DATASET_FILE" ]]; then
  require_file "$ROOT_DIR/han_main_block_decomp.py" "required to build $DATASET_FILE from scratch"
  if missing_downloads "$DOWNLOADS_DIR" "$DECOMP_SOURCE"; then
    note "Ensuring source data downloads exist in $DOWNLOADS_DIR"
    run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" download \
      --data-dir "$DATA_DIR" \
      --decomp-source "$DECOMP_SOURCE"
  else
    note "Reusing existing source downloads in $DOWNLOADS_DIR"
  fi
  note "Building main-block Han decomposition dataset"
  run "$PYTHON_BIN" "$ROOT_DIR/han_main_block_decomp.py" build \
    --data-dir "$DATA_DIR" \
    --decomp-source "$DECOMP_SOURCE" \
    --download-missing \
    --normalization "$NORMALIZATION" \
    --output "$DATASET_FILE"
else
  note "Reusing existing dataset: $DATASET_FILE"
fi

if (( FORCE_REBUILD_MAP )) || needs_rebuild "$MAP_FILE" "$INPUT_FILE" "$DATASET_FILE" "$ROOT_DIR/han_file_decomp_map_transform.py" "$ROOT_DIR/han_file_decomp_map_common.py"; then
  note "Building reversible mapped JSON"
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_decomp_map_transform.py" build \
    --input "$INPUT_FILE" \
    --dataset "$DATASET_FILE" \
    --output "$MAP_FILE"
else
  note "Reusing existing mapped file: $MAP_FILE"
fi

if (( FORCE_REBUILD_SYMBOLIC )) || needs_rebuild "$SYMBOLIC_FILE" "$MAP_FILE" "$ROOT_DIR/han_file_symbolic_serialize.py" "$ROOT_DIR/han_file_symbolic_common.py"; then
  note "Building readable symbolic compact file"
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_serialize.py" from-map \
    --map "$MAP_FILE" \
    --output "$SYMBOLIC_FILE" \
    --mode "$SERIALIZATION_MODE" \
    --include-han-metadata
else
  note "Reusing existing symbolic file: $SYMBOLIC_FILE"
fi

if (( FORCE_RECOVER )) || needs_rebuild "$RECOVERED_FILE" "$SYMBOLIC_FILE" "$ROOT_DIR/han_file_symbolic_reverse.py"; then
  note "Recovering exact original bytes"
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" recover \
    --symbolic "$SYMBOLIC_FILE" \
    --mode original-bytes \
    --output "$RECOVERED_FILE"
else
  note "Reusing existing recovered file: $RECOVERED_FILE"
fi

if (( FORCE_RECOVER )) || needs_rebuild "$SERIALIZED_VIEW_FILE" "$SYMBOLIC_FILE" "$ROOT_DIR/han_file_symbolic_reverse.py"; then
  note "Recovering serialized text view"
  run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" recover \
    --symbolic "$SYMBOLIC_FILE" \
    --mode serialized-text \
    --output "$SERIALIZED_VIEW_FILE"
else
  note "Reusing existing serialized view: $SERIALIZED_VIEW_FILE"
fi

note "Comparing recovered original bytes against the input"
run "$PYTHON_BIN" "$ROOT_DIR/han_file_symbolic_reverse.py" compare \
  --symbolic "$SYMBOLIC_FILE" \
  --mode original-bytes \
  --compare-to "$INPUT_FILE" \
  --report-json "$COMPARE_REPORT_FILE"

if cmp -s "$INPUT_FILE" "$RECOVERED_FILE"; then
  note "Round trip confirmed: recovered file matches input byte-for-byte."
else
  die "Recovered file does not match the original input. See $COMPARE_REPORT_FILE"
fi

note "Done. Outputs:"
printf '  input           %s\n' "$INPUT_FILE"
printf '  dataset         %s\n' "$DATASET_FILE"
printf '  mapped          %s\n' "$MAP_FILE"
printf '  symbolic        %s\n' "$SYMBOLIC_FILE"
printf '  recovered       %s\n' "$RECOVERED_FILE"
printf '  serialized view %s\n' "$SERIALIZED_VIEW_FILE"
printf '  compare report  %s\n' "$COMPARE_REPORT_FILE"

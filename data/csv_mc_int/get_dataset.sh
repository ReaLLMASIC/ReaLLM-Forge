#!/usr/bin/env bash
# Convert a generic integer CSV into per-column multicontext datasets.
# Usage:
#   ./get_dataset.sh [input.csv] --range col:int_min:int_max ...
#   ./get_dataset.sh [input.csv] --default_range int_min:int_max

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_CSV="${SCRIPT_DIR}/input.csv"

if [[ $# -gt 0 && "$1" != -* ]]; then
  INPUT_CSV="$1"
  shift
fi

python3 "${SCRIPT_DIR}/prepare_csv_integer_multicontext.py" \
  --input_csv "${INPUT_CSV}" \
  "$@"

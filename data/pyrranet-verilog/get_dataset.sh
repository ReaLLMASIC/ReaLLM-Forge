#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/orig"
HIGHLIGHTS_SCM="${SCRIPT_DIR}/highlights.scm"
HIGHLIGHTS_URL="https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/queries/verilog/highlights.scm"

# Download highlights.scm if missing (Tree-sitter highlight queries)
if [[ ! -f "${HIGHLIGHTS_SCM}" ]]; then
  echo "[get_dataset.sh] highlights.scm not found; downloading from nvim-treesitter..."
  command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found. Please install curl."; exit 1; }
  curl -L "${HIGHLIGHTS_URL}" -o "${HIGHLIGHTS_SCM}"
  echo "[get_dataset.sh] wrote ${HIGHLIGHTS_SCM}"
else
  echo "[get_dataset.sh] highlights.scm already exists; skipping download."
fi

# Extract dataset -> orig/orig_<index>.v and cache CSV locally
python3 "${SCRIPT_DIR}/organize_datasets.py" \
  --output-dir "${OUTPUT_DIR}"


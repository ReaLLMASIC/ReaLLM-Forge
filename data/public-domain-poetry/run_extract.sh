#!/usr/bin/env bash
# Usage: bash run_extract.sh [INPUT_JSON] [OUTPUT_TEXT]
# Writes one exact output filename. Use the Python converter directly for shards.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/json_poetry_to_espeak_text.py" \
  "${1:-$SCRIPT_DIR/downloaded_jsons/poems.json}" \
  --output "${2:-$SCRIPT_DIR/poems_espeak.txt}" --max-output-size 0

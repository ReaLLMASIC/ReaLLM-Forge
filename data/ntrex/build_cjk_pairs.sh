#!/usr/bin/env bash
set -euo pipefail

# Download/extract NTREX CJK aligned text if needed, then build data/cjk_sentence_pairs.json.
#
# Run from repo root:
#   bash data/ntrex/build_cjk_pairs.sh
#
# Useful overrides:
#   FORCE_SOURCE_DATA=1 bash data/ntrex/build_cjk_pairs.sh
#   SKIP_DOWNLOAD=1 bash data/ntrex/build_cjk_pairs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT="${OUT:-data/cjk_sentence_pairs.json}"
FORCE_SOURCE_DATA="${FORCE_SOURCE_DATA:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

if [[ "$FORCE_SOURCE_DATA" == "1" || ! -s input.txt ]]; then
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "Missing NTREX input.txt and SKIP_DOWNLOAD=1; cannot build $OUT" >&2
    exit 1
  fi
  echo "Downloading/extracting NTREX CJK aligned text..."
  bash get_dataset.sh
fi

if [[ -f "$OUT" && "$FORCE_SOURCE_DATA" != "1" ]]; then
  echo "NTREX CJK pair file already exists: $SCRIPT_DIR/$OUT"
  exit 0
fi

echo "Building NTREX CJK pair file..."
python3 -m cjk_sentence_pairs \
  --input input.txt \
  --out "$OUT"

python3 -m json.tool "$OUT" > /dev/null
echo "Wrote $SCRIPT_DIR/$OUT"

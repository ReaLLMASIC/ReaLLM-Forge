#!/usr/bin/env bash
set -euo pipefail

# Download/extract FLORES CJK text files if needed, then build cjk_sentence_pairs.json.
#
# Run from repo root:
#   bash data/flores200-res/build_cjk_pairs.sh
#
# Useful overrides:
#   FORCE_SOURCE_DATA=1 bash data/flores200-res/build_cjk_pairs.sh
#   SKIP_DOWNLOAD=1 bash data/flores200-res/build_cjk_pairs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT="${OUT:-cjk_sentence_pairs.json}"
FORCE_SOURCE_DATA="${FORCE_SOURCE_DATA:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

needs_text=0
for path in text_kor_Hang.txt text_zho_Hans.txt text_jpn_Jpan.txt; do
  if [[ ! -s "$path" ]]; then
    needs_text=1
  fi
done

if [[ "$FORCE_SOURCE_DATA" == "1" || "$needs_text" == "1" ]]; then
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "Missing FLORES text files and SKIP_DOWNLOAD=1; cannot build $OUT" >&2
    exit 1
  fi
  echo "Downloading/extracting FLORES CJK text files..."
  bash get_dataset.sh
fi

if [[ -f "$OUT" && "$FORCE_SOURCE_DATA" != "1" ]]; then
  echo "FLORES CJK pair file already exists: $SCRIPT_DIR/$OUT"
  exit 0
fi

echo "Building FLORES CJK pair file..."
python3 -m cjk_sentence_pairs \
  --input "$SCRIPT_DIR" \
  --out "$OUT"

python3 -m json.tool "$OUT" > /dev/null
echo "Wrote $SCRIPT_DIR/$OUT"

#!/usr/bin/env bash
# Build simplified-Hanzi radical-location multicontext lanes, then tokenize each
# lane into a labeled subfolder with prepare.py -s -S.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_TXT="${1:-${SCRIPT_DIR}/input.txt}"
LABEL="${LABEL:-simplified_hanzi_mc}"
METHOD="${METHOD:-char}"
python3 "${SCRIPT_DIR}/build_simplified_hanzi_mc.py" --input "${INPUT_TXT}" --output_root "${SCRIPT_DIR}" --label "${LABEL}"
LANES=(char non_hanzi whole left right top bottom enclosure inside corner overlay other)
for lane in "${LANES[@]}"; do
  echo "[prepare] ${lane}"
  (cd "${SCRIPT_DIR}/${lane}" && python3 "${SCRIPT_DIR}/prepare.py" -t input.txt --method "${METHOD}" -s -S "${LABEL}")
done

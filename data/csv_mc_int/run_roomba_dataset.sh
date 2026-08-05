
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_RAW="${1:-${SCRIPT_DIR}/input.csv}"
CONDITIONED_CSV="${SCRIPT_DIR}/roomba_integer.csv"

python3 "${SCRIPT_DIR}/roomba_data_conditioning.py" \
  --input_csv "${INPUT_RAW}" \
  --output_csv "${CONDITIONED_CSV}" \
  --mapping_csv "${SCRIPT_DIR}/action_mapping.csv"

RANGE_ARGS=$(python3 - <<PY
import pandas as pd

df = pd.read_csv("${CONDITIONED_CSV}")

args = []

for col in df.columns:
    min_val = int(df[col].min())
    max_val = int(df[col].max())

    if col == "timestamp":
        min_val, max_val = 0, 9
    elif col == "speed_mm_s":
        min_val, max_val = -400, 400
    elif col == "total_distance_mm":
        min_val, max_val = 0, 999
    elif col == "battery_percent":
        min_val, max_val = 0, 1000
    elif col == "action":
        min_val, max_val = 0, int(df[col].max())
    elif col.startswith("p") and col[1:].isdigit():
        min_val, max_val = 0, 255

    args.append(f"--range {col}:{min_val}:{max_val}")

print(" ".join(args))
PY
)

echo "Running:"
echo "${SCRIPT_DIR}/get_dataset.sh ${CONDITIONED_CSV} ${RANGE_ARGS}"

"${SCRIPT_DIR}/get_dataset.sh" "${CONDITIONED_CSV}" ${RANGE_ARGS}

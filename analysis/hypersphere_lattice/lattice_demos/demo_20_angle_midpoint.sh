#!/usr/bin/env bash
# Demo 20 — Angle & SLERP midpoint
# --------------------------------
# What this does:
#   - If missing, generates a small 384-D set (N=3000) using rseq.
#   - Computes the angular distance between indices i=42 and j=117.
#   - Computes the SLERP midpoint and saves it.
#
# What to expect:
#   - The printed JSON includes "angle_degrees" ~ around 90° in high dimensions (±10° typical).
#   - A file ./data/s384_mid_42_117.npy containing a single 384-D unit vector (the midpoint).
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
OUT=./data
mkdir -p "$OUT"
PTS="$OUT/s384_rseq_3k.npy"
if [[ ! -f "$PTS" ]]; then
  echo "Generating $PTS ..."
  python3 "$SCRIPT" generate --dim 384 --n 3000 --method rseq --output "$PTS"
fi

echo "Angular distance between indices 42 and 117:"
python3 "$SCRIPT" angle --points "$PTS" --i 42 --j 117

echo
echo "Saving SLERP midpoint to ./data/s384_mid_42_117.npy"
python3 "$SCRIPT" midpoint --points "$PTS" --i 42 --j 117 --output "$OUT/s384_mid_42_117.npy"

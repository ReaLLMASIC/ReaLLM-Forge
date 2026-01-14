#!/usr/bin/env bash
# Demo 10 — Generate S^2 point sets
# ---------------------------------
# What this does:
#   - Generates ~6,000 points on S^2 using four methods: rseq, halton, sfib, random.
#   - Saves outputs under ./data/
#
# What to expect:
#   - Four .npy files in ./data with shapes (6000, 3).
#   - These are used by other demos.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
OUT=./data
mkdir -p "$OUT"

python3 "$SCRIPT" generate --dim 3 --n 6000 --method rseq   --output "$OUT/s2_rseq_6000.npy"
python3 "$SCRIPT" generate --dim 3 --n 6000 --method halton --output "$OUT/s2_halton_6000.npy"
python3 "$SCRIPT" generate --dim 3 --n 6000 --method sfib   --output "$OUT/s2_sfib_6000.npy"
python3 "$SCRIPT" generate --dim 3 --n 6000 --method random --output "$OUT/s2_random_6000.npy"

echo
echo "Generated files:"
ls -lh "$OUT"/s2_*_6000.npy

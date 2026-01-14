#!/usr/bin/env bash
# Demo 11 — Generate S^383 (R^384) point sets
# -------------------------------------------
# What this does:
#   - Generates 10,000 points on S^383 using rseq and halton.
#   - Saves outputs under ./data/
#
# What to expect:
#   - Two .npy files in ./data with shapes (10000, 384).
#   - Generation is deterministic for rseq; halton is quasi-deterministic.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
OUT=./data
mkdir -p "$OUT"

python3 "$SCRIPT" generate --dim 384 --n 10000 --method rseq   --output "$OUT/s384_rseq_10k.npy"
python3 "$SCRIPT" generate --dim 384 --n 10000 --method halton --output "$OUT/s384_halton_10k.npy"

echo
echo "Generated files:"
ls -lh "$OUT"/s384_*_10k.npy

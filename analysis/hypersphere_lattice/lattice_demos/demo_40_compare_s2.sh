#!/usr/bin/env bash
# Demo 40 — Compare methods on S^2 with plots
# -------------------------------------------
# What this does:
#   - Runs the 'compare' mode on S^2 for methods: rseq, halton, random, sfib, latlong.
#   - Outputs plots (pair-angle vs theory, NN-angle hist, equal-area heatmap) to ./plots_s2.
#   - Writes a CSV summary of metrics.
#
# What to expect:
#   - ./plots_s2 has several PNGs: one per method per plot type, plus bar charts for metrics.
#   - ./plots_s2/s2_summary.csv contains metrics like min separation, covering radius, cap discrepancy.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
PLOTS=./plots_s2
mkdir -p "$PLOTS"

python3 "$SCRIPT" compare --dim 3 --n 6000 --plots-dir "$PLOTS"   --summary-csv "$PLOTS/s2_summary.csv"

echo
echo "Wrote plot PNGs and summary CSV in $PLOTS"
ls -1 "$PLOTS" | head -n 20

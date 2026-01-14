#!/usr/bin/env bash
# Demo 41 — Compare methods on S^383 (R^384) with plots
# -----------------------------------------------------
# What this does:
#   - Runs 'compare' for dim=384 with methods: rseq, halton, random.
#   - Outputs pair-angle vs theory & NN-angle hist plots to ./plots_hd.
#   - Writes a CSV summary of metrics.
#
# What to expect:
#   - Pairwise angle histograms concentrated near 90 degrees.
#   - NN-angle distributions show local spacing differences across methods.
#   - ./plots_hd/hd_summary.csv contains metrics for each method.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
PLOTS=./plots_hd
mkdir -p "$PLOTS"

python3 "$SCRIPT" compare --dim 384 --n 8000 --plots-dir "$PLOTS"   --summary-csv "$PLOTS/hd_summary.csv"

echo
echo "Wrote plot PNGs and summary CSV in $PLOTS"
ls -1 "$PLOTS" | head -n 20

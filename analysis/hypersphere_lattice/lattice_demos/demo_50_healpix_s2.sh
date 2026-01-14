#!/usr/bin/env bash
# Demo 50 — HEALPix on S^2 (optional; requires healpy)
# ----------------------------------------------------
# What this does:
#   - If 'healpy' is installed, runs 'compare' on S^2 including HEALPix with NSIDE≈sqrt(n/12).
#   - Otherwise, prints a message and exits.
#
# What to expect (if healpy present):
#   - ./plots_hp contains plots including a 'healpix' method.
#   - The HEALPix set will be truncated to n points if 12*NSIDE^2 > n.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
PLOTS=./plots_hp
N=6000
# Choose NSIDE ~ sqrt(N/12) rounded
NSIDE=$(python3 - <<'PY'
import math; N=6000; nside=round(math.sqrt(N/12)); print(max(1,int(nside)))
PY
)

if python3 -c "import healpy" >/dev/null 2>&1; then
  mkdir -p "$PLOTS"
  echo "Running compare with HEALPix (NSIDE=$NSIDE) ..."
  python3 "$SCRIPT" compare --dim 3 --n "$N" --healpix-nside "$NSIDE" --plots-dir "$PLOTS"     --summary-csv "$PLOTS/hp_summary.csv"
  echo
  echo "Wrote plot PNGs and summary CSV in $PLOTS"
  ls -1 "$PLOTS" | head -n 20
else
  echo "healpy not installed; skipping HEALPix demo. Install with: pip install healpy"
fi

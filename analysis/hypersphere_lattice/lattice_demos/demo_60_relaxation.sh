#!/usr/bin/env bash
# Demo 60 — Relaxation vs. baseline (S^2)
# ---------------------------------------
# What this does:
#   - Generates two S^2 rseq sets with N=6000:
#       * baseline (no relaxation)
#       * relaxed (3 steps, k=16)
#   - Computes two metrics for each: min separation angle (approx) and covering radius (approx).
#     (Uses a tiny inline Python snippet importing functions from hypersphere_lattices.py)
#
# What to expect:
#   - The relaxed set usually shows a slightly **higher** min separation and **lower** covering radius.
#   - Printed table with the two metrics for both sets.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
OUT=./data
mkdir -p "$OUT"

BASE="$OUT/s2_rseq_6k_base.npy"
RELX="$OUT/s2_rseq_6k_relaxed.npy"

if [[ ! -f "$BASE" ]]; then
  python3 "$SCRIPT" generate --dim 3 --n 6000 --method rseq --output "$BASE"
fi

# 3 relax steps (demo); increase for stronger effect if you can afford O(N^2) work
python3 "$SCRIPT" generate --dim 3 --n 6000 --method rseq --relax-steps 3 --relax-k 16 --relax-lr 0.1 --output "$RELX"

python3 - <<'PY'
import numpy as np, json
import hypersphere_lattices as H

def metrics(X):
    return {
        "min_sep_deg": H.separation_min_deg(X, anchors=800, cand=2048),
        "cover_deg": H.covering_radius_est_deg(X, probes=400, cand=2048),
    }

base = np.load("./data/s2_rseq_6k_base.npy")
relx = np.load("./data/s2_rseq_6k_relaxed.npy")
mb = metrics(base); mr = metrics(relx)
print("\nResults (approx):")
print("  baseline: min_sep_deg = %.3f | covering_radius_deg = %.3f" % (mb["min_sep_deg"], mb["cover_deg"]))
print("  relaxed : min_sep_deg = %.3f | covering_radius_deg = %.3f" % (mr["min_sep_deg"], mr["cover_deg"]))
print("\nExpectation: relaxed >= baseline (min_sep), relaxed <= baseline (covering radius).")
PY

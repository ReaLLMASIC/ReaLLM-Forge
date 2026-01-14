#!/usr/bin/env bash
# Demo 30 — Build KNN graph & approximate path
# --------------------------------------------
# What this does:
#   - Generates a modest S^2 set (N=5000) for speed, builds a k=16 KNN graph (O(N^2) demo).
#   - Queries an approximate shortest path between two indices (i=42, j=117).
#
# What to expect:
#   - A .npz graph file with neighbor indices and edge angles.
#   - The printed JSON includes:
#       * approx_graph_geodesic_degrees  (path length over the KNN graph)
#       * great_circle_angle_radians     (true angle between endpoints)
#     Typically, the graph geodesic ≥ great-circle angle; ratio often ~1.0–1.2 for k≈16.
set -euo pipefail
SCRIPT=./hypersphere_lattices.py
OUT=./data
mkdir -p "$OUT"
PTS="$OUT/s2_rseq_5000.npy"
GRAPH="$OUT/s2_rseq_5000_k16.npz"

if [[ ! -f "$PTS" ]]; then
  echo "Generating $PTS ..."
  python3 "$SCRIPT" generate --dim 3 --n 5000 --method rseq --output "$PTS"
fi

echo "Building KNN graph (k=16) ..."
python3 "$SCRIPT" knn --points "$PTS" --k 16 --graph "$GRAPH"

echo
echo "Approximate shortest path from i=42 to j=117:"
python3 "$SCRIPT" path --points "$PTS" --graph "$GRAPH" --i 42 --j 117

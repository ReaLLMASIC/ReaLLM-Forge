#!/bin/bash

set -x 
# 1) Generate 10k points on S^383 (R^384)
python hypersphere_lattices.py generate --dim 384 --n 10000 --method rseq --output s384_rseq_10k.npy

# # 2) S^2 Fibonacci spiral
# python hypersphere_lattices.py generate --dim 3 --n 20000 --method sfib --output s2_sfib_20k.npy

# 3) HEALPix (requires: pip install healpy) — N implied by NSIDE=64
# python hypersphere_lattices.py generate --dim 3 --method healpix --healpix-nside 64 --output s2_hp_n64.npy

# 4) Angular distance & SLERP midpoint
python hypersphere_lattices.py angle --points s384_rseq_10k.npy --i 42 --j 117
python hypersphere_lattices.py midpoint --points s384_rseq_10k.npy --i 42 --j 117 --output mid.npy

# 5) Simple KNN graph + approximate path (demo O(N^2))
python hypersphere_lattices.py knn --points s384_rseq_10k.npy --k 16 --graph knn.npz
python hypersphere_lattices.py path --points s384_rseq_10k.npy --graph knn.npz --i 42 --j 117

# 6) Compare methods and create plots & summary
#    S^2 defaults: rseq halton random sfib latlong (+ healpix if --healpix-nside set)
python hypersphere_lattices.py compare --dim 3 --n 6000 \
  --plots-dir ./plots --summary-csv ./plots/s2_summary.csv

# #    High-d defaults: rseq halton random
python hypersphere_lattices.py compare --dim 384 --n 8000 \
  --plots-dir ./plots_hd --summary-csv ./plots_hd/hd_summary.csv


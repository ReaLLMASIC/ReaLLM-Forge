python hypersphere_grid.py \
  --benchmark --dim 384 \
  --bench-methods kronecker,halton,random \
  --bench-Ns 1024:8192:1024 \
  --bench-repeats 2 \
  --bench-out out/bench.csv \
  --bench-plots out/bench_plots


#!/bin/bash
# analysis/high_dim_overlap/demo_merge.sh

python s2_plotly_rotation_demo.py \
  --metric-mode merge \
  --mode gen --nA 1200 --nB 1200 \
  --gen-model vmf --kappa1 25 --kappa2 50 \
  --beta 12 --theta-deg 20 --prox-deg 20 --alpha-chord 0.9 \
  --fig-height 1200 --scene-frac 0.60 \
  --fib-N 4096 --healpix-nside 192 --angle-step 10 \
  --metrics  bc,mmd,mean_cos,mean_nn,kl,cov_dir,haus_dir,fib_recall,chord_merge,mean_nn \
  --color-A "#1f77b4" --color-B "#ff7f0e" \
  --plot-subset 1000 --marker-size 1 \
  --out-html s2_merge_vmf.html


# LM Head Pairwise Angle Comparison

Tools for comparing whether the geometry of `lm_head` vocabulary vectors is preserved between two nanoGPT checkpoints.

The analysis computes all vocabulary-vector pairwise angles from each checkpoint's `lm_head.weight`, then compares the two angle matrices. For small vocabularies such as `shakespeare_char` this is easy to visualize directly.

## CLI

```bash
python3 analysis/lm_head_pairwise_angles/compare_lm_head_pairwise_angles.py \
  out_a/ckpt.pt out_b/ckpt.pt \
  --meta data/shakespeare_char/meta.pkl \
  --min-angle 0 --max-angle 180 \
  --html out/lm_head_pairwise_angles/report.html \
  --csv out/lm_head_pairwise_angles/pairs.csv
```

Use `--device cuda` for CUDA acceleration when available, or `--device auto` to prefer CUDA if `torch.cuda.is_available()`.

## Webapp

```bash
python3 analysis/lm_head_pairwise_angles/app.py --ckpt-root . --host 127.0.0.1 --port 7860
```

The webapp scans for checkpoint files, lets you choose two checkpoints, set an angle window, and renders A-sorted pair-angle plots, difference histograms, larger square A/B pairwise-angle heatmaps, difference heatmaps, a baseline-alignment scatter, metric cards, and heatmap color/direction toggle menus.

## Trend across training iterations

The demo saves major checkpoints every 200 iterations, including the initial
iteration-0 checkpoint when training starts. To plot how pairwise-angle
preservation changes over time:

```bash
python3 analysis/lm_head_pairwise_angles/plot_lm_head_pairwise_angle_trend.py \
  out/shakespeare_lm_head_pairwise_angles_demo/seed_1337 \
  out/shakespeare_lm_head_pairwise_angles_demo/seed_2024 \
  --iterations 0,200,400,600,800 \
  --meta data/shakespeare_char/meta.pkl \
  --csv out/shakespeare_lm_head_pairwise_angles_demo/analysis/trend.csv \
  --html out/shakespeare_lm_head_pairwise_angles_demo/analysis/trend.html
```

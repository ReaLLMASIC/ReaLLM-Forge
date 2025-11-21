# Demos

This folder will hold repeatable demonstrations of features and results.

## Shifted GELU

This shows that the GELU wants to shift:


Call this from either repo root dir, or from demos dir, but ensure the relative
path for ckpt_path is from your present directory.

Example for calling from repo root directory:

```bash
python3 demos/check_ckpt_for_gelu_shift.py \
        --ckpt_path out/ckpt.pt
```

## Optimizer Comparison

`adam_vs_adamw.sh` trains two tiny Shakespeare models, one with Adam and one
with AdamW, then compares their statistics using `view_model_stats.py`.

## Snap-to-Grid Projections

`snap_to_grid_demo.sh` prepares the Shakespeare character dataset, trains a
small model with snap-to-grid enabled, evaluates multiple grid sizes, and then
generates text with and without the projections. Run it from the repository
root:

```bash
bash demos/snap_to_grid_demo.sh
```

You can override the default model or snap-to-grid settings by exporting
environment variables before running the script, for example:

```bash
N_HEAD=3 N_EMBD=384 SNAP_SIZES="100 1000 10000" bash demos/snap_to_grid_demo.sh
```

Ensure that `N_EMBD` remains divisible by `N_HEAD`; otherwise the
multi-head attention projection will be invalid and the script will exit with
an explanatory error.

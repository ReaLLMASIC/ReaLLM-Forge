# Model Parameter Exploration

These are utility scripts designed to inspect and navigate through the parameter
tree of a trained GPT model checkpoint. It allows users to explore the hierarchy
of model parameters interactively and inspect their values.

## Features

- Load a GPT model checkpoint.
- Display the hierarchical structure of the model's parameters.
- Interactively navigate through the parameter tree.
- View parameter tensor values.

## Requirements

- Python 3.x
- PyTorch
- A valid GPT checkpoint file (e.g., `out/ckpt.pt`).

## Usage

### Running the Script

This script must be executed from the **main directory of the repository**. The
typical path to a checkpoint file is `out/ckpt.pt`.

```bash
python checkpoint_analysis/checkpoint_explorer.py <ckpt_path> [--device <device>]
```

### Positional Arguments

- `<ckpt_path>`: The path to the model checkpoint file (e.g., `out/ckpt.pt`).

### Optional Arguments

- `--device`: The device to load the model on (`cpu` by default, e.g., `cuda` for GPU).

### Example Command

```bash
python checkpoint_analysis/checkpoint_explorer.py out/ckpt.pt --device cuda
```

### Interactive Navigation

1. **Start at Root**: The script begins at the root of the parameter tree.
2. **Explore Submodules**: Use numbers to explore submodules or parameters.
3. **Go Back**: Enter `b` to go back to the previous level.
4. **Quit**: Enter `q` to exit the script.
5. **View Parameter Values**: When selecting a parameter (leaf node), its tensor value will be displayed. Press `Enter` to return to the previous level.

### Notes

- The script processes the checkpoint to ensure compatibility by renaming keys starting with `_orig_mod.`.
- Parameter tensor values longer than 1000 characters will be truncated in the display for readability.

## Troubleshooting

- Ensure the script is run from the **main repository directory**.
- Verify that the checkpoint file exists at the specified path.
- Check for dependencies like PyTorch before running the script.

## JL Transforming a Checkpoint

The script `jl_transform_ckpt.py` applies a Johnson–Lindenstrauss transform to
every weight tensor in a checkpoint. It can also change the model’s embedding
dimension while keeping attention head dimensions and MLP sizes intact. The transformed checkpoint and the
original `meta.pkl` are written to a new directory. Optimizer and scheduler
states are removed so training restarts cleanly. Use `--jl_type` to select the
kind of JL transform (e.g. `sign`, `gaussian`, `sparse`, or `srht`).
When using the `gaussian` type you may set `--gaussian_mean` and
`--gaussian_std` to control the distribution of the projection matrix.  The
optional `--cproj_vertical` flag projects any `c_proj.weight` tensors along their
first dimension instead of the default behaviour.
The script also resets `best_val_loss` and `best_iter` in the new checkpoint so
training restarts from scratch after transformation.

```bash
python checkpoint_analysis/jl_transform_ckpt.py out \
    --out_dir out_jl --out_embd <new_dim> --jl_type sign
```


## Pairwise-Dot Island Analysis

Use `analyze_dot_islands_ckpt.py` to scan checkpoint tensors for groups of vectors
("islands") that are strongly similar under pairwise cosine/dot product.

The script writes three artifacts in `--out_dir` (default: `<ckpt_dir>/island_analysis`):
- `islands_detailed.json`: per-tensor / per-threshold island membership and provider vectors
- `islands_summary.csv`: summary stats per tensor and threshold
- `islands_dashboard.html`: Plotly dashboard with a tensor selector to compare island metrics

Example:

```bash
python3 analysis/checkpoint_analysis/analyze_dot_islands_ckpt.py out_shakespeare_checkpoint_demo \
  --pattern "wte|attn|mlp" \
  --metric cosine \
  --thresholds 0.2,0.35,0.5 \
  --min_island_size 4 \
  --top_providers 6
```


## Island-Routing Augmentation + Speed Comparison

Use `augment_ckpt_with_island_routing.py` after generating `islands_detailed.json`.
It creates routing metadata with per-island probe vectors (default: island means)
for first-stage routing, then applies final multiplication only on the selected
row group. It also adds a fallback probe/group for rows not covered by any
island, so routing always has 100% coverage.

Outputs in `--out_dir` (default: `<ckpt_dir>/island_routing`):
- `island_routing.pt`: routing metadata for each eligible 2D tensor
- `island_routing_speed.csv`: per-tensor TTFT/decode before-vs-after speed table
- `island_routing_speed.json`: JSON copy of speed stats
- `island_routing_speed.html`: Plotly visualization of the comparison

Example:

```bash
python3 analysis/checkpoint_analysis/augment_ckpt_with_island_routing.py out_shakespeare_checkpoint_demo \
  --threshold 0.35 \
  --provider_mode top \
  --probe_mode mean
```


## Validation-Loss-Constrained Island Search

Use `search_island_tradeoff.py` for a **greedy per-tensor threshold-step search**:
1) initialize each tensor near `--start_threshold` (default `0.5`),
2) test one tensor at a time by lowering its threshold by `--threshold_step` (default `0.05`),
3) for each round, evaluate the full pass and pick the candidate with the **lowest validation-loss increase**,
4) stop if the best candidate exceeds the loss tolerance; otherwise accept and continue.

Outputs in `--out_dir` (default: `<ckpt_dir>/island_tradeoff_search`):
- `search_log.yaml`: round-by-round tested candidates, losses, selected candidate, decode latency estimates, stop reason
- `search_results.json`: JSON mirror of the final search state including baseline-vs-selected speed comparison
- `selected/ckpt.pt`: checkpoint for selected configuration
- per-candidate subdirs with eval artifacts
- by default, unselected `candidates/*/ckpt.pt` files are deleted after evaluation (`--delete_unselected_ckpts`) to reduce disk usage

Example:

```bash
python3 analysis/checkpoint_analysis/search_island_tradeoff.py out_shakespeare_checkpoint_demo \
  --start_threshold 0.5 \
  --threshold_step 0.05 \
  --loss_tolerance_pct 2.0 \
  --eval_dataset shakespeare_char \
  --eval_iters 100 \
  --device cpu --dtype float32
```

The selected checkpoint is exported by default (`--export_selected_ckpt`), unselected candidate checkpoint files are deleted by default (`--delete_unselected_ckpts`), and speed is compared against baseline using eval-derived iteration/token latency metrics.

TUI log viewer (template-style similar to `view_hp_log.py`):

```bash
python3 analysis/checkpoint_analysis/view_island_tradeoff_log.py out_shakespeare_checkpoint_demo/island_tradeoff_search/search_log.yaml
```

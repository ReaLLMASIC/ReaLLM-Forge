# Quick depth sweep: width 512, context 2048

This separate entry point compares **softmax Flash SDPA and fused ReLU²**, at
**16, 24, 32 layers** by default. It uses 4 heads of dimension 128, FFN width
2048, QK norm, learned QK scale, RoPE, no output norm, and the existing Muon
decoder-matrix / AdamW auxiliary parameter split. Muon weight decay is **0.0**;
auxiliary AdamW matrix decay is 0.1; gains, biases and scalars have no decay.
It does not change the original depth or context presets.

Each of the six runs trains for **100,007,936 tokens** (3,052 optimizer steps,
rounding 100M up to the next 32,768-token batch). Microbatch size is 2 and
gradient accumulation is 8. Training has **no wall-clock cutoff**. “Quick”
means a small, explicit six-run comparison rather than a maximum-depth search.
The listed depths are candidate settings, not a claim that they fit a 4090.

| Layers | Parameters per model |
|---:|---:|
| 16 | 76,104,208 |
| 24 | 101,278,232 |
| 32 | 126,452,256 |

Softmax and ReLU² have identical parameter counts at each depth. These counts
include the tied 50,304-token embedding / LM-head matrix.

From the extracted package:

```bash
# Reuse the tested existing environment if it lives elsewhere:
# export LAB_PYTHON=/absolute/path/to/previous/lab/venv/bin/python
bash run_quick_depth.sh plan
bash run_quick_depth.sh all
```

`all` verifies packaging, runs the included tests, measures **every requested
depth/attention pair** in a disposable CUDA process, prepares/reuses shared
token data, and trains sequentially. The capacity probe performs two complete
optimizer steps and validation, including optimizer states and the real loss
chunking. It reserves 1 GiB of GPU memory. Any failed fit stops before training;
compiler errors are reported separately from capacity failures. CUDA capacity
and performance still need measurement on your GPU.

For separate checks and training:

```bash
bash run_quick_depth.sh probe --layers 16 24 32
bash run_quick_depth.sh train --layers 16 24 32
```

Change depths, token budget or microbatch size explicitly, with a fresh output:

```bash
bash run_quick_depth.sh all --layers 8 12 16 --tokens 100000000 \
  --microbatch-tokens 2048 --output runs/quick_d512_ctx2048_smaller
```

Reducing microbatch tokens to 2048 uses batch size 1 and accumulation 16, so
the optimizer token batch stays identical. Do not adjust layer/batch/token
options inside an existing run. For an interrupted run, use the same options:

```bash
bash run_quick_depth.sh train --resume
```

The default resource policy budgets at most **30 GB additional disk**, keeps
**2 GB free**, retains all final model weights and reports, and retires optimizer
state only after a completed checkpoint and final export are verified equal.
Interrupted jobs retain exact resume state. Completed jobs can be evaluated or
loaded as models but cannot resume the original optimizer trajectory after
retirement. Only one training job is active; a failed job stops the sweep.
The disk estimate includes final FP32 weights, checkpoint rotation, and 3 GB
for data/logs/caches; it excludes a newly installed CUDA environment. Reuse your
existing venv and prepared data when space is tight. Actual free space is checked
before every run. External disk use/download caches can still consume the reserve.

Monitor directly from files; no localhost connection is required:

```bash
watch -n 10 cat runs/quick_d512_ctx2048/live/report.md
# If training/its monitor stopped, render once or restart just the monitor:
bash run_quick_depth.sh report
bash run_quick_depth.sh monitor
```

`live/index.html` auto-refreshes when opened as a file. The monitor refreshes
tables every 10 seconds and PNG/PDF figures every 60 seconds:

- `validation_progress.png`: native-2048 validation versus training tokens,
  including partial progress, with each depth and attention identified.
- `best_validation_vs_layers.png`: best scheduled native-2048 validation
  within the full token budget, **completed runs only**.
- `final_validation_vs_layers.png`: final native-2048 validation at equal
  training tokens, **completed runs only**.
- Matching PDF figures, `depth_runs.csv`, `validations.csv`, and
  `quick_summary.json` provide exportable data.

Each depth's existing context report is also retained under `L0016/live/`, etc.
Shared-512-context validation remains available there. Scheduled validation,
excluding initialization and pause-only checks, controls “best loss.” This is a
single-seed screen: compare attention methods within the same depth/token budget.
Increasing depth changes parameter count and total compute; equal tokens do not
mean equal compute or wall time.

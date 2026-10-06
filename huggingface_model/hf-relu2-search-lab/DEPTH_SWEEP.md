# Depth sweep v1.1 — three depths starting at 16 layers, with a disk budget

Contexts remain **512 and 1024**, with identical depths at both. Width 384, 3 heads,
FFN 1536, eight attention arms, Muon and both output-norm settings remain unchanged.
Default: **96 runs**, each with 100,007,936 tokens (3,052 updates), without a time limit.
The original trainer, model, kernels and accuracy tests are preserved.

## Choosing the three depths

Fresh GPU workers probe all selected conditions at both contexts, using validation,
two complete accumulated optimizer steps and validation again. This includes lazy
optimizer state and the real chunked LM-head loss. Capacity search retains **1 GiB
GPU headroom**. Compiler/numerical/process errors stop calibration; only CUDA OOM
or measured reserve overflow counts as a capacity failure. If every candidate
through 256 layers fits, increase `--max-search-layers`; the cap is not a measured max.

The nominal endpoint is floor(70% of the common usable layer limit). The disk
planner **lowers this endpoint if necessary** to fit all final models. It selects
three distinct, evenly spaced integer depths including 16 and that endpoint.
Every selected depth still passes the GPU probe. `depth_plan.json` records both
endpoints and whether disk space was limiting. “70%” applies to layer count, not
VRAM utilization. The empirical two-step probe is not a stability guarantee for
every later training step.

## Disk accounting and checkpoint policy

Usable additional storage is the smaller of **30 GB** and actual free space,
minus a **2 GB reserve**. GB means decimal bytes, not GiB. Installed environments
already reduce the measured free space. The estimate includes:

- Every final FP32 model, using exact meta-device parameter counts.
- A conservative 32 bytes/parameter for the largest job's temporary checkpoint
  rotation/export, in addition to the retained final-model total.
- 3 GB for packed data, reports, logs, metadata and compiler/download caches.

If even the smallest three-depth grid cannot fit, the sweep stops before training.
It never silently drops arms or deletes earlier results. Reuse your existing
environment/data, free space, or explicitly choose fewer variants/norm conditions.

The scheduler completes **one job at a time**, preventing optimizer-state buildup
across unfinished runs. Each job retains the original 25M/50M/100M milestones, LR
horizon and checkpoint/evaluation cadence. After a successful full-budget run:

1. Compare the final HF export with the final resume checkpoint, including every
   tensor's shape, dtype and content hash.
2. Keep final model weights, tokenizer/config, all metrics, summaries and reports.
3. Replace redundant checkpoint weights with **hardlinks** to the verified final
   weights (one physical copy), and remove `training_state.pt`. Retain
   `progress.json` for reports and completed-run skipping. Record final-file hashes
   and the policy in `storage_retention.json`.

This requires hardlink support on the output filesystem (normal Linux ext4/XFS
supports it). If linking fails, optimizer state is preserved and cleanup stops.
Both final and checkpoint paths remain usable for model loading and diagnostics;
their weights share storage, so they are not independent copies to edit in place.

**Completed jobs cannot resume with their original optimizer state.** Their final
models remain usable for inference/evaluation or new fine-tuning. Interrupted and
failed runs keep full resume state. A failed job stops the sweep so failed states
cannot accumulate. Older sweeps are untouched; use a fresh output for this version.

Actual free space is checked before every new job, allowing checkpoint rotation
and the reserve. These are estimates and runtime checks, not a filesystem quota:
unrelated programs can fill the disk between checks, and caches can exceed their
allowance. Global caches, datasets, prior results and final models are never
automatically deleted.

## Run

Extract over the package source directory to reuse its `venv/` and `data/`; neither
is present in the ZIP. The new default output avoids the earlier depth sweep:

```bash
bash run_depth.sh plan
bash run_depth.sh all
```

For a fresh environment only, avoid retaining downloaded pip wheels:

```bash
PIP_NO_CACHE_DIR=1 bash scripts/setup.sh
bash run_depth.sh all
```

Or reuse another lab's environment and data without copying:

```bash
export LAB_PYTHON=/absolute/path/to/existing-lab/venv/bin/python
bash run_depth.sh all --data /absolute/path/to/existing-lab/data/fineweb_100m
```

FineWeb is streamed into one shared packed token stream, not downloaded in full.
100M GPT-2 tokens occupy about 200 MB as uint16, plus validation and metadata.
If data is on a different filesystem, check that filesystem's free space as well.

`all` runs the full correctness gate. For separate calibration and training:

```bash
bash run.sh test-cuda
bash run_depth.sh probe
cat runs/depth_16_3_disk30/depth_plan.json
bash run_depth.sh train
```

Resume the frozen grid and the original recipe with:

```bash
bash run_depth.sh train --resume
```

For only the three main attention variants (**36 runs**):

```bash
bash run_depth.sh all --variants softmax_sdpa relu2 linear16 \
  --output runs/depth_main3_disk30
```

Use identical selection flags on resume. Overrides for new outputs include
`--start-layers`, `--points`, `--disk-budget-gb`, `--disk-reserve-gb`, and the GPU
reserve `--reserve-gib`. Changed recipes, sources, GPU/allocator settings, or data
are rejected on resume. A five-depth v1 plan cannot be reused as this experiment.

## Monitoring without localhost

```bash
watch -n 10 cat runs/depth_16_3_disk30/live/report.md
bash run_depth.sh report
```

During training, tables update every 10 seconds and figures every 60 seconds:

- `live/report.md`, `depth_runs.csv`, `depth_summary.json`, `index.html`.
- `live/best_validation_vs_layers.png` / `.pdf`.
- `live/best_common512_vs_layers.png` / `.pdf` (identical evaluation windows).
- `live/final_validation_vs_layers.png` / `.pdf`.
- `L0016/live/` etc. contain the original per-depth context comparisons.

Plots require completed equal-token budgets; incomplete values are progress only,
never zero scores. Parameter count and compute grow with depth. This is a fixed
recipe, equal-token comparison, not equal compute or a tuned optimum. Single-seed
results have no seed uncertainty estimate. Capacity logs/progress are recorded in
`capacity/`, `capacity_progress.json`, and `depth_status.json` before training.
No GPU capacity or model-quality result is precomputed here. See
`DEPTH_VALIDATION.md` for executed tests.

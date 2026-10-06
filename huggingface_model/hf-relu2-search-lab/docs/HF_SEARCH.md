# Hugging Face / Triton architecture search

This additive runner compares `softmax_sdpa`, fused `relu2`, and fused `linear16` for every selected architecture. It uses `SearchConfig` and `SearchForCausalLM`, QK normalization, the learned QK logit scale, RoPE, and the existing chunked LM-head loss. Existing comparison models, kernels and run directories are preserved.

## Start with the bounded baseline

Use the package's pinned environment (`bash scripts/setup.sh` if needed), then:

```bash
bash run_hf_search.sh plan
bash run_hf_search.sh all --data data/hf_search_100m
```

`all` verifies the package manifest and runs the model/search/Trainer tests (including available CUDA tests), prepares one shared packed FineWeb-Edu token stream if absent, runs an isolated real CUDA training/validation capacity check for each job, and trains jobs serially. `run` skips that initial test gate for an already-tested package but still performs each job's capacity check. Reuse an existing compatible packed stream with `--data /path/to/prepared-data`; no dataset copy is made. A plan only constructs models on the meta device and never initializes CUDA or downloads data.

The launcher/factory rejects unsupported Transformers/tokenizers versions with a direct environment error before model initialization. Use the pinned 4.44.2/0.19.1 pair; this avoids the earlier tied-weight metadata crash caused by launching with a different Python environment.

The default preset has **one architecture and three runs**, not an unbounded search: 384 model channels, 16 layers, 3 heads, QK and V head dimensions 128, context 1024. Each run sees 100,007,936 target tokens (100M rounded up to 3,052 complete 32,768-token optimizer steps). There is no training deadline. Matched variants consume the same data cursor, seed, token budget, schedule, validation cadence and initial parameter values.

```bash
# Explicitly sample two architectures, always train all three arms for each.
bash run_hf_search.sh plan --preset configs/hf_search_random_example.json --output runs/hf_random2
bash run_hf_search.sh run --preset configs/hf_search_random_example.json \
  --output runs/hf_random2 --data data/hf_search_100m

# Continue the exact same experiment after an interruption.
bash run_hf_search.sh run --output runs/hf_search --data data/hf_search_100m --resume

# Optional early inspection: absolute per-run step, original LR horizon retained.
bash run_hf_search.sh run --output runs/hf_early --data data/hf_search_100m --stop-after 200
bash run_hf_search.sh run --output runs/hf_early --data data/hf_search_100m --resume
```

Pass the same `--preset` and `--output` when resuming a nondefault search. Source, config, data manifest, environment and retention-policy changes are rejected on resume. Repeating an already-reached `--stop-after` does not train an extra step. Completed jobs are skipped. Ctrl-C/SIGTERM tells the active worker to checkpoint after its current optimizer step, with a 120-second shutdown grace. An exclusive output lock prevents concurrent launchers from writing the same run.

## Finite search format

Copy a preset before editing. `search_space` maps fields to finite numeric lists. `method="grid"` expands the full Cartesian product and refuses to exceed the explicit `max_architectures`. `method="random"` samples without replacement using `search_seed`; it does not silently trim a grid. The maximum allowed architecture budget is 64.

Supported fields:

| Field | Meaning |
|---|---|
| `hidden_size` | Residual stream / model width |
| `intermediate_size` | FFN hidden width; independent of model width |
| `num_hidden_layers` | Decoder depth |
| `num_attention_heads` | Head count; need not divide model width |
| `n_qk_head_dim` | Query/key width per head; even for the full RoPE path |
| `n_v_head_dim` | Value width per head |
| `context` | Training context |
| `muon_learning_rate` | Decoder matrix learning rate |
| `adamw_learning_rate` | Auxiliary AdamW learning rate |
| `qk_norm_scale_init` | Initial learned QK logit scale |
| `relu2max_divisor` | Fixed divisor for unnormalized ReLU attention |
| `rope_theta` | Rotary frequency base |

Every architecture receives all three attention treatments. A change to a learning rate counts as a distinct trial. This is a concrete finite-search adapter, **not a port of upstream greedy width/depth growth or weight transplantation**. Layer-specific head lists, GQA, recurrent Infini-attention memory and a differentiable search controller are not implemented here.

## Independent dimensions and performance interpretation

Projections are `Q,K: d_model -> H*d_qk`, `V: d_model -> H*d_v`, and `O: H*d_v -> d_model`. QK normalization and RoPE operate only on the actual QK channels. There is no `d_model % H == 0` constraint. CUDA head dimensions are currently limited to 256; use even QK widths for full RoPE.

The additive adapter uses exact zero-padding to reuse the proven equal-width fused kernels. The stored padded width is `max(d_qk,d_v)`; the tensor-core tile width is the next power of two, with a minimum of 16. `plan.json` records both widths and their ratios. This makes arbitrary widths mathematically possible, **not automatically efficient**. For example, QK=64/V=128 uses a 128-wide kernel; some QK arithmetic is padded. Unequal widths can also require extra padding/slicing allocations outside the fused attention operation. These costs are included in full-model tokens/s. The report records the actual backend labels; CUDA fallback to generic PyTorch or math SDPA is forbidden for this comparison.

## Optimizer convention

Decoder `nn.Linear.weight` matrices use native `torch.optim.Muon`, with **weight decay exactly 0.0**. The embedding matrix, LM head, normalization gains, QK scale and biases use AdamW. Eligible matrices in the AdamW group have decay **0.1**; scalar/vector gains and biases have decay **0.0**. Tied WTE/LM-head weights enter AdamW once. `optimizer_groups.json` records every parameter, shape, group, LR and decay; group coverage is checked before training.

Muon LR and auxiliary AdamW LR are independently searchable. The scheduler applies the same token-step horizon to both groups. The original NS coefficients, momentum and LR adjustment remain explicit in the recipe. This runner is a custom fixed-token loop around Hugging Face model APIs; it does not delegate dataset batching or evaluation to `Trainer`.

## Files and graphs to inspect during training

```bash
# No HTTP server or localhost port is required.
cat runs/hf_search/report.md
tail -f runs/hf_search/logs/arch*.log
bash run_hf_search.sh report --output runs/hf_search
bash run_hf_search.sh monitor --output runs/hf_search
```

The active launcher refreshes tables about every 15 seconds and graphs about every 60 seconds. A separate `monitor` keeps refreshing after the launcher exits. `report.html` can be opened directly as a local file; it auto-refreshes while a launcher/monitor is updating its source files.

| File | Contents |
|---|---|
| `results.csv`, `results.json` | Status, tokens, native/common loss, throughput, training time, peak memory, actual backends and padding ratios |
| `report.md`, `report.html` | Readable progress and links to the correct failed-training or failed-preflight logs |
| `validation_progress.png/.pdf` | Live validation vs tokens, separate architecture panels, including unfinished runs |
| `loss_by_architecture.png/.pdf` | Best common-context loss, completed equal-token runs only |
| `loss_vs_training_time.png/.pdf` | Completed-run loss/time trade-off; timing excludes evaluation/checkpoints |
| `runs/<job>/metrics.jsonl` | Append-only step, throughput, validation and learned-scale records |
| `probes/<job>.result.json` | Real CUDA capacity check, optimizer-state allocations, peaks and backend |

Native losses should only be compared at the same context; common-context validation enables a consistent cross-context view. Graphs show individual seeds, not invented confidence intervals. Failed/missing runs are missing measurements, never zero loss. Prefer final validation for a fixed-budget primary comparison; best loss is also recorded using only shared scheduled evaluations and the final budget, never pause-only evaluations.

## 30 GB disk policy

The planner estimates retained checkpoints, final FP32 weights, transient atomic checkpoint rotation and a 3 GB data/cache/log allowance, then reserves another 2 GB of free drive space. Training checks free space again before every job. The default retains full optimizer checkpoints for later exact continuation within the configured horizon. It refuses oversized plans rather than deleting older work.

`--retire-completed` is an explicit optional space reduction: after complete training, checkpoint tensors are checked against final safetensors, duplicate model files become hardlinks and the completed optimizer state is removed. **Those completed jobs can then be loaded for inference/evaluation but cannot resume their exact optimizer trajectory.** Interrupted jobs retain optimizer state. Use the same flag on resume; do not toggle retention inside an existing search. No prior run directory is deleted.

The 3 GB overhead estimate is not a promise about a newly downloaded PyTorch wheel or system-wide caches. Reuse the pinned environment and prepared data on a drive with only 30 GB free.

## Offline validation

```bash
venv/bin/python -m pytest -q tests/test_hf_search.py tests/test_search_model.py
venv/bin/python -m hf_search.smoke --output runs/hf_offline_smoke
```

The tiny CPU smoke uses d_model=20, three heads, QK=8 and V=6, exercising both non-divisible model width and unequal head dimensions. Tests cover all three training arms, identical initial weights, exact Muon resume and HF export/reload. CPU runs are functionality checks, never CUDA throughput claims. Run the included CUDA head-dimension tests on the 4090 before interpreting a new GPU architecture search.

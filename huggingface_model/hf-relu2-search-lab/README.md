# Depth sweep v1.1 — 16-layer start, three depths, 30 GB disk budget

**Use `bash run_depth.sh all`.** See [DEPTH_SWEEP.md](DEPTH_SWEEP.md). Every final
model and report is retained; optimizer state is retired after a completed run.
Interrupted runs remain exactly resumable. The older context-only instructions
below are for `run.sh` and do not enable this disk policy.

# ReLU² context lab — integrated release

**BF16 accuracy-check update:** fixes the cached-attention dQ false-failure pattern without changing kernels or the RMS error limit. See [BF16_ACCURACY_FIX.md](BF16_ACCURACY_FIX.md) for the diagnosis, GPU rerun commands, and JSON diagnostics.

A self-contained **source package** for a matched small-model experiment, with data preparation,
training, exact resume, Triton kernels, correctness tests, checkpoint diagnostics, and live reports.
No previous ZIP, checkout, or Python environment is required. Dependencies and the training corpus
are downloaded during setup/preparation; datasets and trained weights are not included in this ZIP.
Extract into a fresh directory to keep earlier experiments intact.

## Start from this ZIP

Linux x86-64, Python 3.10–3.13, an NVIDIA SM80+ GPU (including RTX 4090), and a driver compatible
with the CUDA 12.8 PyTorch wheel are required for training. CPU is supported for functional tests.

```bash
unzip relu2-context-lab.zip
cd relu2-context-lab
bash scripts/setup.sh
bash run.sh plan
bash run.sh all
```

`setup.sh` creates **this folder's `venv/`** and installs PyTorch 2.9.1, Triton 3.5.1,
Transformers 4.44.2, tokenizers 0.19.1, and the other pinned dependencies. It installs the
bundled FLA/einops wheels for optional KDA. It does not select a sibling Conda environment.
The Transformers pin avoids the incompatible tied-weight API that caused the earlier startup failure.
CUDA wheel installation follows [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/).
Use `LAB_BOOTSTRAP_PYTHON=/path/to/python3 bash scripts/setup.sh` to choose the bootstrap interpreter.

`all` verifies the archive manifest and runs the test suite **on your GPU** before preparing data.
It then packs a shared FineWeb-Edu sample, executes the selected sweep, probes each completed
checkpoint, and exports reports. Workers use the same Python executable as the launcher.
The first CUDA tests/training calls compile and autotune kernels; this is additional startup time.
No CUDA correctness or performance result is claimed from the CPU-only packaging environment.

To reuse prepared data with its manifest and tokenizer:

```bash
bash run.sh all --data /absolute/path/to/data/fineweb_full
```

The default data path is `data/fineweb_100m`. The default output is
`runs/context_ablation_50m`. A failed/incomplete data download has no completed manifest;
prepare a fresh data directory rather than appending to a partial token stream.

## What runs

**48 runs = 8 variants × 3 contexts × 2 output-norm conditions × 1 seed.**
Every run trains for **100,007,936 tokens** (3,052 updates × 32,768 tokens), rounded up from
100M. This is **4,800,380,928 training tokens across the default sweep**, not 100M shared
between all runs. There is **no time limit**. Contexts are **512, 1024, 2048**.

The decoder has **17 layers, width 384, 3 heads per layer, head width 128, FFN width 1536**,
GELU, tied WTE/LM head, RoPE, QK L2 normalization, and a learned scalar QK scale per layer.
The original no-output-norm controls have **49,411,217 parameters**. Each bias arm adds
**51 parameters** (17 × 3). Enabling both output norms adds **13,056** channel gains.

| Variant | Learned per-head subtractive bias | Causal length exponent α |
|---|---|---:|
| `softmax_sdpa` | No | — |
| `relu2` | No | 0 |
| `linear16` | No | 0 |
| `relu2_bias` | Yes | 0 |
| `relu2_scale_half` | No | ½ |
| `relu2_scale_one` | No | 1 |
| `relu2_bias_scale_half` | Yes | ½ |
| `relu2_bias_scale_one` | Yes | 1 |

All eight run with `output_norm=none` and `capped`. The latter applies a capped RMS
normalization to the attention output projection and FFN output **before residual addition**:
`y = x / max(1, rms(x)) * (1 + gain)`. Gain is learned per channel, starts at zero, and applies
to all vectors. The radial operation alone caps the norm at √384; gain can subsequently change it.

Muon updates decoder matrices. Auxiliary AdamW updates tied WTE/LM head, norm gains,
QK scales, and the new per-head biases. The biases have **no weight decay**. The exact recipes
and tensor assignments are saved in each run's `optimizer_groups.json`.

## New attention rules

For head h and query i, after QK normalization and positional encoding:

```python
score_ij = learned_qk_scale * dot(q_i, k_j) - learned_bias[h]
visible_i = number_of_causally_visible_unmasked_keys(i)
factor_i = max(1, visible_i / 512) ** (-alpha)
weight_ij = relu(score_ij)**2 / 256 * factor_i  # zero for masked/future keys
output_i = sum_j(weight_ij * v_j)
```

The bias is **signed and unconstrained**, initialized at zero independently in every layer/head.
A positive value suppresses weak matches; a negative value admits more matches. It is not a
linear projection bias or a softmax logit shift. Scaling uses actual causal prefix counts,
including valid cached keys, never the padded sequence length. Rows with ≤512 visible keys
retain the original scale. At 1024 keys, α=½ multiplies by 1/√2 and α=1 by ½.
`causal_length_anchor` is configurable before training and is saved in the HF config.

The new fused forward/backward live in `hf_model/triton_context_attention.py`; the original
kernels remain unchanged. It includes dQ/dK/dV, learned scale and per-head bias gradients,
mask handling, and split-cache decoding. No N×N score matrix is written. A padding mask
needs one O(batch × keys) integer prefix sum. See [CONTEXT_ATTENTION.md](CONTEXT_ATTENTION.md)
for the equations, diagram, code map, and test coverage. Speed and quality improvements
must be measured on the GPU; these are hypotheses about extending the earlier ReLU² win.

## Smaller selections and optional presets

```bash
# Inspect or execute only context 1024, with both norms off.
bash run.sh plan --contexts 1024 --output-norms none
bash run.sh all --contexts 1024 --output-norms none

# Fill the remaining cells later, preserving optimizer state and LR horizon.
bash run.sh train --resume
```

A selection still has the full plan in its report: unselected cells remain pending. Use the
same flags on resume to continue just that selection. Editing a preset, environment,
model code or training data requires a new output directory. Ctrl+C requests a checkpoint
after the active optimizer step; wait for the process to finish saving before restarting.

| Preset | Runs | Purpose |
|---|---:|---|
| `configs/context_ablation_50m.json` | 48 | Default bias/length ablation at 512/1024/2048 |
| `configs/context_ablation_extended.json` | 80 | Adds contexts 256 and 4096; common evaluation at 256 |
| `configs/context_ablation_with_kda.json` | 54 | All default ablations plus the existing Triton KDA arm |
| `configs/context_ablation_muon_adamw.json` | 96 | Default arms with Muon+AdamW and full AdamW |
| `configs/sweep_50m.json` | 24 | Preserved original softmax/ReLU²/linear16/KDA output-norm sweep |

Example:

```bash
bash run.sh all --preset configs/context_ablation_with_kda.json --output runs/context_with_kda
```

KDA is the preserved pure recurrent arm with its own gated head RMSNorm, extra gates/convolutions,
and no RoPE; it is not the released Kimi hybrid. Its parameter count differs. Row-based
attention diagnostics are marked **not applicable** for KDA, while validation, training speed,
and output-norm diagnostics still run. Architecture notes: `docs/KDA_AND_OUTPUT_NORM.md`.

## Reports you can monitor as files

The coordinator trains in matched rounds near **25M → 50M → 100M tokens**, retaining the full
100M-token LR schedule. Validation runs every 150 steps and at the final budget. Checkpoints
are saved every 300 steps and at round boundaries. The latest resumable checkpoint and final
HF export are retained. There is no periodic data download once preparation completes.

Default report directory: `runs/context_ablation_50m/live/`.

| File | Contents |
|---|---|
| `report.md` | Progress, errors, tables, and graph links; readable without a server |
| `best_validation_vs_context.png` | Best scheduled validation loss within each run's budget |
| `final_validation_vs_context.png` | Validation loss at matched completed training tokens |
| `common_context_validation.png` | Validation at a shared 512-token window |
| `output_norm_delta.png` | Paired capped-minus-no-norm loss difference |
| `sweep_results.csv`, `validation.csv` | Progress, training throughput, and losses |
| `attention_parameters.csv` | **Every layer/head's learned bias, QK scale, α, anchor, and token count** |
| `output_norm_diagnostics.csv` | Per-layer branch RMS, cap activation, and gain statistics |
| `diagnostics/index.html` | Completed-checkpoint attention and loss-by-position graphs |
| `diagnostic_attention.csv` | Per-head row mass, energy, effective keys, sparsity and weight shares |
| `diagnostic_loss_by_position.csv` | Held-out cross-entropy by causal position bin |
| `diagnostic_branch_norms.csv` | Position-binned branch input/output RMS |

Validation graphs are also exported as PDF/SVG. Detailed probe graphs have PNG/PDF exports.
The default dashboard is **http://localhost:8774** on the training host. If the port is busy,
the monitor continues writing files. If connecting remotely, use your existing SSH tunnel;
`localhost` in your browser refers to the browser's machine.

```bash
# Read a report without a web connection.
cat runs/context_ablation_50m/live/report.md

# Regenerate all reports once or restart the live monitor.
bash run.sh report
bash run.sh monitor --port 8775
```

The monitor refreshes tables every 10 seconds and graphs every 60 seconds during training.
It stops with the coordinator; saved files persist. Failed or missing jobs have no score.
Filled comparison points require the full budget; hollow points are incomplete best-so-far
results. A single seed gives a screening result, not an uncertainty estimate.

## Run diagnostics directly

The training launcher probes every completed non-KDA run at all contexts in the selected
preset, using **32,768 held-out targets per evaluation context**. Probes do not train or alter
weights and are outside training-step timing. Results include actual training token counts.
Evaluating beyond a model's training context is labeled extrapolation in the probe notes.

```bash
# Recover missing/failed probes, reusing complete matching results.
bash run.sh diagnose --probe-tokens 32768

# Probe a saved intermediate checkpoint; default batch mode only probes complete runs.
venv/bin/python -m diagnostics.sweep \
  --root runs/context_ablation_50m --data data/fineweb_100m \
  --run ctx1024_relu2_bias_scale_half_muon_norm-none_seed0 \
  --include-partial --device cuda:0 --tokens 32768

# Probe a specific checkpoint/run at custom lengths into a fresh directory.
venv/bin/python -m diagnostics.probe_context \
  --project . \
  --checkpoint runs/context_ablation_50m/ctx1024_relu2_bias_scale_half_muon_norm-none_seed0 \
  --data data/fineweb_100m --contexts 256 512 1024 2048 \
  --tokens 32768 --device cuda:0 --output runs/manual_probe_01
```

Each requested context must divide the probe token count and fit the model maximum.
Diagnostic errors are recorded separately in `diagnostics_state.json`; they do not replace
training measurements. Re-running `diagnose` recovers incomplete results into new attempts.
Raw records remain under the sweep's `diagnostics/` directory. Reconstructed attention rows
use FP32 arithmetic; they are measurements of the learned rule, not fused-kernel error tests.

## Tests and offline smoke

```bash
bash run.sh test                 # all CPU tests; CUDA tests also run if CUDA is available
bash run.sh test-cuda            # refuses to pass when no CUDA device is available
bash run.sh smoke                # 32 tiny runs on synthetic text, then diagnostic exports

# The two explicitly requested correctness suites:
venv/bin/python -m pytest -q tests/test_context_attention.py tests/test_context_cuda.py

# Actual Triton math on CPU, and offline GPU compilation (no timing claims):
TRITON_INTERPRET=1 venv/bin/python scripts/check_context_kernels.py \
  --interpreter --output runs/interpreter_checks.json
venv/bin/python scripts/check_context_kernels.py \
  --compile --output runs/compilation_checks.json
```

For a CPU-only installation: `bash scripts/setup.sh --cpu`, then `bash run.sh smoke`.
Synthetic smoke results validate the pipeline, not model quality. The CPU tests include
finite-difference and closed-form bias gradients, length factors at the anchor, causal
prefix/cache invariance, masks/all-masked rows, optimizer routing, HF save/load, exact
Muon resume, diagnostic recovery, and file monitoring with a busy HTTP port.
GPU cases test the **actual fused outputs and all gradients** in BF16 and actual HF/Muon
updates for all new arms with output norms on/off. See `VALIDATION.md` for checks performed
before packaging and the remaining GPU gate.

Final checkpoints include all custom Python modules and work with:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "runs/context_ablation_50m/ctx1024_relu2_bias_muon_norm-none_seed0/final",
    trust_remote_code=True,
)
```

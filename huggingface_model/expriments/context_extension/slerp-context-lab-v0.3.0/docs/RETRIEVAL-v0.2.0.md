# SLERP Context Lab v0.2.0

Full-weight finetuning for a single Linux RTX 4090 (24 GB VRAM), about 50 GB free
on the internal drive, and occasional transfers to an external drive.

**Start with Hunyuan-0.5B-Instruct (539,010,048 backbone parameters).** MiniCPM5-1B
is included as an alternative. This is a new experiment: v0.1 used Pythia with
LoRA; its adapter checkpoints cannot be resumed here.

## What changed

- Every backbone parameter is trainable. No LoRA, PEFT, or quantized base weights.
- FP32 weights, gradients and AdamW updates; BF16 CUDA computation. Optional
  8-bit optimizer states still update every backbone weight.
- Hunyuan dense and standard Llama integration, with a Pythia control config.
  Hunyuan's post-RoPE Q/K normalization, native NTK-alpha RoPE and actual head
  dimensions are preserved. Native context metadata is not enlarged.
- A native baseline calls Hugging Face generation directly. It has no SLERP
  wrapper in its forward pass. Both instruct tokenizers use their own chat
  templates with thinking disabled for exact-answer tasks.
- SLERP/NLERP/local arms share seeds, prompts, token budgets and backbone
  initialization. GQA caches and spherical slots use KV heads; reads serve all
  query heads. All default runs keep temporal gradients through the prefix.
- Complete backbone + memory checkpoints with optimizer/RNG/progress resume.
  One completed resumable checkpoint per run is retained by default.
- Random queried records and multi-answer order; 1/8/32-fact evaluation. Two-hop
  tasks require at least two facts and explicitly skip the one-fact combination.
- Optional exact vocabulary KL(reference || student) for short-context retention.
  It is disabled in the main learning experiment; a separate KL config enables it.
- Raw JSONL, summary CSV, PNG/PDF plots and JSON plot data are retained. Strict
  format accuracy and separately parsed code-value accuracy are both reported.

The package's synthetic tasks are diagnostics, **not official RULER or LongBench
scores**. The 4090 memory limit and full training quality must be measured on your
machine. See `docs/VALIDATION.md` for exactly what was checked during packaging.

## Install and start

Use Python 3.10–3.12 and a working NVIDIA driver. Extract this version separately
from the v0.1 package:

```bash
unzip slerp-context-lab-v0.2.0.zip
cd slerp-context-lab-v0.2.0
bash setup.sh
bash scripts/quickstart_4090.sh
```

Setup installs PyTorch 2.7.1 from the CUDA 12.6 wheel index and the pinned Python
requirements, then runs the CPU tests. Set `TORCH_INDEX_URL` if you need another
compatible PyTorch wheel index. `bash setup.sh --cpu` is for development tests.

Quickstart downloads Hunyuan, checks native forward parity, profiles full training
at 2K and 4K, then evaluates the unmodified model at 4K/8K/16K/32K. It does not
launch the 5M-token pilot automatically. Inspect:

- `reports/hunyuan-parity.json`: must pass before training.
- `reports/profile-hunyuan-slerp-*.json`: optimizer states included; at least 2 GiB
  GPU headroom required. OOM results are recorded and stop the script.
- `reports/screen/assessment.json`: actual generated-answer accuracy by length.
- `reports/screen/figures/`: plots and raw plotting data.

To screen both candidates instead, use a fresh screen output directory or run this
before quickstart's screen step:

```bash
bash scripts/screen_candidates.sh both
```

Existing evaluation outputs are never silently appended or overwritten. For a
rerun, choose a new `--out` using the CLI, or intentionally move prior outputs.
A crashed/incomplete evaluation cannot pass the assessment gate.

## Prove the memory path can learn

```bash
bash scripts/learning_gate.sh
```

This trains **full weights** for local, NLERP and SLERP from the same pretrained
backbone with a 256-token local window, 768-token examples, two facts, and early
evidence. Each arm has a 500K-token budget, batch size one with accumulation, and
fresh deterministic training examples. Evaluation uses 32 held-out examples.

The script checks an 80% evicted-evidence code-value recall hurdle for SLERP. This is a
proposed diagnostic threshold, not an expected result or a significance test.
Inspect the local and NLERP results too: a high SLERP score alone does not show an
advantage. If it fails, inspect outputs, learning curves and gradients before
spending the longer training budget. `GATE_THRESHOLD` can change the reported
threshold, but lowering it is not evidence that the method works.

## Run the matched pilot

After reviewing the learning gate, explicitly release its optimizer snapshots:

```bash
bash scripts/finalize_gates.sh
FINALIZE_AFTER_EVAL=1 bash scripts/run_pilot.sh hunyuan
```

`finalize` preserves all trained weights but removes optimizer state, so that
stage can no longer resume exactly. `FINALIZE_AFTER_EVAL=1` does this only after
each pilot arm has completed evaluation. It keeps peak disk use lower while the
next arm trains. Omit the variable if you want to retain resumability and have
room for all three full optimizer snapshots.

Default pilot settings:

| Setting | Value |
|---|---|
| Backbone | Hunyuan-0.5B-Instruct |
| Trainable weights | Entire backbone plus memory |
| Methods | local, NLERP, SLERP |
| Native context | 262,144 tokens, unchanged |
| Local window / eviction chunk | 2,048 / 256 tokens |
| Memory slots | 64 per KV head per layer |
| Training sequence lengths | 2,048 (30%), 4,096 (70%) |
| Evaluation sequence lengths | 4K, 8K, 16K, 32K |
| Generation budget | 128 tokens; limit hits logged |
| Processed-token budget | 5M per arm |
| Accumulation target | 8,192 processed tokens per update |
| Backbone / memory learning rate | 1e-5 / 1e-4 initial maxima |
| Optimizer | AdamW, FP32 state, foreach disabled |
| Activation checkpointing | Enabled |
| Head loss chunk | At most 32 supervised tokens at a time |
| Local K/V gradient detachment | **False** |
| Retention KL | **Off** in the default pilot |

Training uses deterministic complete episodes, so accumulation targets and total
budgets can overshoot by the final episode. Both processed and answer-supervised
token counts are logged. Five million input tokens are a pilot budget, not a claim
of sufficient answer supervision or convergence. Losses are averaged per episode.

The rolling local-only arm can carry information indirectly through recent hidden
states; it has no explicit spherical memory. It is not equivalent to freshly
truncating every prompt to its last 2K tokens.

## Resume or continue with a new stage

```bash
# Identical config and budget: exact training resume.
bash run.sh train --config configs/hunyuan-slerp.json \
  --out runs/hunyuan-slerp-seed17 --resume --device cuda

# New config/budget: initialize full weights into a NEW run, with a fresh optimizer.
bash run.sh train --config configs/hunyuan-slerp-kl.json \
  --initialize runs/hunyuan-slerp-seed17 --out runs/hunyuan-slerp-kl-stage2 --device cuda
```

A resume request with changed experiment settings is rejected. A weights-only
checkpoint can initialize a new stage but cannot resume its previous optimizer.
Re-running an already completed stage with an identical config returns its recorded
completion without loading weights or taking another step.
The saved full model includes a backbone config and tokenizer, so trained-model
reload does not require re-downloading the original base weights.

## Optional short-context KL

`configs/hunyuan-slerp-kl.json` sets `kl_weight=0.02`. The frozen reference is the
original pretrained model. On every fourth episode, the student and reference
process the same first 512 input tokens; exact vocabulary KL is averaged over the
last 16 positions. The term is added to that episode's supervised CE. It has no
rewards, policy ratios, or RL advantage estimates. Its average contribution over
all episodes is reduced by the sampling frequency.

This adds a reference model and an extra short forward/backward pass. Profile that
specific config before training. The profile includes KL on every measured step,
which is conservative relative to its every-fourth-episode training schedule.
`mean_retention_kl` and `kl_samples` are logged. Short prefixes from synthetic data
are only a narrow retention check; mix natural text and evaluate held-out language
loss for stronger claims about preserving general ability.

```bash
bash scripts/profile_lengths.sh configs/hunyuan-slerp-kl.json
```

## MiniCPM5-1B alternative

MiniCPM has 1,080,632,832 parameters and a standard Llama architecture. The provided
training configs use an 8-bit AdamW optimizer to leave more activation space:

```bash
.venv/bin/python -m pip install --no-cache-dir bitsandbytes==0.48.2
bash scripts/screen_candidates.sh minicpm
FINALIZE_AFTER_EVAL=1 bash scripts/run_pilot.sh minicpm
```

These are full-weight updates with compressed optimizer moments, not LoRA or a
4-bit frozen base. Profile before training. The model's tokenizer and architecture
were checked here; its real pretrained weights and the CUDA 8-bit optimizer were
not exercised during packaging. Hunyuan is the more thoroughly checked first run.

## Storage and external drive

Active model downloads, data, environment, logs and checkpoints stay in the
extracted project directory. No CPU/disk optimizer offload or automatic USB sync
is configured. Default guards require 8 GB free after the next write and keep the
project under 40 GB, leaving room within your approximately 50 GB free budget.

Hunyuan has about 2.16 GB of FP32 backbone weights and about 4.31 GB of Adam moments;
a full resumable checkpoint is approximately 6.5 GB plus memory/tokenizer/metadata.
Atomic replacement temporarily needs room for both old and new checkpoints. The
old checkpoint is removed only after the replacement is complete. GPU activations
are separate from these disk numbers.

Do not leave every completed stage's optimizer snapshots on the internal drive.
Use explicit finalization after evaluation, or an occasional archive:

```bash
bash scripts/archive_run.sh runs/hunyuan-slerp-seed17 /media/YOUR_EXTERNAL_DRIVE/slerp-archives 20
```

The archive copies the latest checkpoint and run metadata sequentially at a target
20 MB/s. It never reads the model cache for archiving, runs in the background, or
deletes the local run. Rate limiting does not guarantee a particular drive
temperature. A finalized run's archive contains weights, not a resumable optimizer.

## Natural-text data, scoring and official benchmarks

```bash
bash scripts/prepare_text.sh hunyuan
```

This streams PG-19 into bounded token files (25M train, 2M validation, 2M test),
at most roughly 116 MB of token arrays. Set `natural_fraction` in a new config, e.g. 0.2,
and use the matching `data_dir` and pinned tokenizer revision. Splits are kept
separate and exact source hashes checked. See `docs/EVALUATION.md` for suffix
perplexity, generated-history diagnostics and the official JSONL prediction bridge.

**Interpretation:** a native 256K model evaluated with a 2K working window measures
memory compression and retention at 4K–32K. It does not demonstrate extension past
that pretrained model's native context. A Pythia full-weight control config remains
available for a separate native-context-extension experiment.

## Reading accuracy correctly

Hunyuan sometimes returns the correct values inside `<answer>` tags or a sentence.
Every row retains `exact_match` (strict first-line formatting) and
`value_exact_match` (all six-digit codes in the final answer, in order). The value
parser does not consult the target to decide which codes to extract. Extra,
duplicated or reordered codes fail. A completed thinking prefix is excluded; a
single explicit answer block is used when present.

Plots and candidate/learning-gate assessment default to `value_exact_match`. For
strict formatting results use `--metric exact_match` with `plot` or `assess` and a
separate output directory. Both metrics remain in the raw data and summary CSV.
The parser is specific to these synthetic code tasks and is never substituted for
an official benchmark scorer.

See `docs/METHOD.md`, `docs/EVALUATION.md`, `docs/VALIDATION.md`, and `CHANGELOG.md`.

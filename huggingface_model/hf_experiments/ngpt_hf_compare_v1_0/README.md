# nGPT vs matched GPT — Hugging Face comparison, v1.0

Train **two new language models from scratch**, with the same backbone dimensions,
corpus, tokenization, sampled batches, sequence length and update budget. Save raw
training/validation measurements and plot both models against optimizer iterations.

The implementation targets the **original dense nGPT paper, final ICLR 2025 / arXiv
v2 specification**, not the separate 2026 hybrid-MoE training recipe. This is a
small, readable single-GPU experiment rather than a distributed paper reproduction.

**Testing status of the supplied release:** 20 PyTorch core tests passed, including
exact CPU checkpoint resume, and both models completed a 100-update synthetic CPU
training run. The Hugging Face integration test module was skipped because
Transformers is not installed in the build environment and network installation
was unavailable. CUDA throughput/VRAM and an actual OpenWebText run were not tested
there. `setup.sh` installs dependencies and runs the HF save/load tests on your
machine before you start a long experiment. Included plots are prominently labeled
**synthetic plumbing tests, not language-model benchmark results**.

## 1. Install on Linux / RTX 4090

Use a fresh directory, separate from other experiments. Python 3.10+ is required.

```bash
unzip ngpt_hf_compare_v1_0.zip
cd ngpt_hf_compare_v1_0
bash setup.sh
source .venv/bin/activate
```

The installer uses PyTorch 2.10.0 CUDA 12.8 wheels; it does not build FlashAttention
or modify the NVIDIA driver. For a different compatible wheel index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash setup.sh
```

For an already working isolated CUDA environment, skip `setup.sh` and install
`requirements.txt` there, then run `python -m pytest -q`. Avoid mixing old research
venvs, especially ones with incompatible TorchVision/audio dependencies. Neither
TorchVision nor TorchAudio is needed by this project.

Dependencies deliberately pin Transformers 4.57.6 and Datasets 4.5.0 instead of
silently following future major API changes. These selected HF versions were not
runtime-tested in the build environment. Core tests ran with torch 2.10.0+cpu,
Python 3.13, NumPy 2.3.5 and Matplotlib 3.10.8. Every real run records installed
versions in its `run.json`.

## 2. Offline plumbing test

```bash
python -m pytest -q
python compare_ngpt.py smoke --out runs/synthetic_smoke
```

This runs the **same PyTorch model/loss/training core** on artificial noisy cyclic
token transitions, using width 64, two layers, four heads and 100 updates on CPU.
It does **not** test downloading/tokenizing HF data, the HF model adapter, or GPU
performance. Use a new `--out` to repeat it. See `example_results/synthetic_cpu/`
for the actual build-environment CSVs and plots, without model checkpoints.

## 3. Short real-data pilot

```bash
DATA=data/owt_20m OUT=runs/owt_pilot \
TRAIN_TOKENS=20000000 VAL_TOKENS=200000 \
STEPS=1000 WARMUP_GPT=100 \
bash scripts/run_4090.sh
```

This streams only enough OpenWebText to create the requested local token budgets,
then trains GPT and nGPT **sequentially**, not concurrently. The pilot uses a
shortened baseline warmup; it is a plumbing/learning check, not the exact paper's
2,000-step-warmup experiment. nGPT still has zero warmup. A 1,000-update pilot does
not establish the original paper's long-horizon claims.

If the model runs out of GPU memory, preserve the effective batch while reducing
microbatch size, and enable activation checkpointing:

```bash
DATA=data/owt_20m OUT=runs/owt_pilot_lowmem \
STEPS=1000 WARMUP_GPT=100 BATCH_SIZE=1 ACCUMULATION=32 \
ACTIVATION_CHECKPOINTING=1 bash scripts/run_4090.sh
```

This is a **new run**. Do not change batch/context/schedule settings during resume.

## 4. Main matched comparison

```bash
bash scripts/run_4090.sh
```

Equivalent direct CLI after preparing the data:

```bash
python compare_ngpt.py prepare \
  --dataset Skylion007/openwebtext --tokenizer gpt2 \
  --data data/owt_200m --train-tokens 200000000 --val-tokens 1000000

python compare_ngpt.py train \
  --data data/owt_200m --out runs/owt_4090 \
  --variants gpt ngpt --seeds 0 \
  --width 512 --layers 8 --heads 8 --context 1024 \
  --batch-size 2 --accumulation 16 --steps 10000 \
  --lr-gpt 0.003 --lr-ngpt 0.003 --warmup-gpt 2000 \
  --precision bf16 --weight-storage fp32
```

Do not repeat the direct `prepare` command over an existing data directory: data
are immutable and this intentionally fails rather than silently replacing them.
The shell launcher reuses an existing `metadata.json`; changing `TOKENIZER` or
`DATASET` therefore requires a **new `DATA` directory**.

### Default dimensions and budget

| Setting | Both variants |
|---|---:|
| Transformer blocks | 8 |
| Width | 512 |
| Attention heads / head dimension | 8 / 64 |
| SwiGLU intermediate dimension | 2,048 (4 × width) |
| Context / predicted tokens per sequence | 1,024 |
| Microbatch × gradient accumulation | 2 × 16 |
| Effective batch | 32 sequences |
| Next-token targets per optimizer update | 32,768 |
| Optimizer updates per model | 10,000 |
| Token presentations per model | 327,680,000 |
| Prepared training pool | 200,000,000 tokens |
| Validation / fixed-train evaluation cadence | every 100 updates, plus iteration 0 and the end |
| Probe size per split / evaluation | 32,768 predicted tokens |

With the default unpadded 50,257-token GPT-2 vocabulary, the parameter counts are
**85,026,304 GPT** and **85,112,913 nGPT**. The small difference is from nGPT's learned
scales versus the baseline's RMSNorm gains, not extra backbone capacity. All
backbone matrix shapes match. Initialization uses the same seed and Gaussian
matrix directions, with variant-appropriate scales/projection.

Training samples random contiguous windows with replacement, like the nanoGPT
reference. Thus 327.68M/200M = 1.6384 token-pool equivalents, **not** 1.6384 exact
non-overlapping epochs. Documents are packed with EOS separators; causal attention
can cross document boundaries, as in ordinary packed LM pretraining.

The token files use uint16 for this vocabulary: 201M cached tokens require about
402 MB decimal. Two resumable FP32 Adam checkpoints plus final model exports are
roughly 2.7 GB decimal, before logs and atomic-save temporary space. The virtual
environment, CUDA dependencies and HF cache are additional. The code does not
download a pretrained LLM or save a new checkpoint at every evaluation. It retains
one `last.pt` per run, plus one HF export after completion. Allow extra disk space
for atomic replacement; the runner checks a conservative estimate before training.

GPU peak memory and timing are **measured at runtime**, not promised here. Use the
logged tokens/second and ETA after several updates. Training-time estimates exclude
future validation and checkpoint overhead. The elapsed-time plot includes those
costs within the run, but excludes initial data preparation/model setup and idle
periods between resumed processes.

## 5. What makes this nGPT, rather than just another norm layer

The main implementation is `model.py`; HF integration is in `hf_model.py`.

| Component | GPT baseline | nGPT |
|---|---|---|
| Residual updates | Pre-RMSNorm followed by ordinary addition | Unit-normalized sublayer suggestions and normalized learned interpolation |
| Final normalization | RMSNorm | None: residual stream already normalized |
| Weight constraints | Unconstrained | Unit-vector projection before training and after every optimizer update |
| Queries/keys | RoPE, usual attention | RoPE, per-head unit normalization, shared learned coordinate scales |
| Attention multiplier | 1/sqrt(head dimension) | sqrt(head dimension), passed explicitly to SDPA |
| MLP | Bias-free SwiGLU, 4d intermediate width | Same matrix shapes, with learned branch scales and sqrt(d) gate rescaling |
| LM head | Untied output embedding | Untied row-normalized output embedding plus vocabulary-wise learned scale |
| Optimizer | AdamW, matrix weight decay 0.1 | AdamW with weight decay 0 everywhere (equivalent to Adam) |
| Warmup | 2,000 updates in main run | None |
| Schedule | Cosine to zero | Cosine to zero |

Both models use RoPE base 10,000, no biases, no dropout, Adam betas (0.9, 0.95),
epsilon 1e-8 and gradient clipping at 1.0, following the original paper/reference
recipe. This comparison tests **architecture plus its prescribed training recipe**;
it is not an architecture-only optimizer-controlled ablation.

The residual rule is:

```text
suggestion = unit_norm(sublayer(h))
alpha = abs(alpha_raw * (0.05 / (1/sqrt(d))))
h = unit_norm(unit_norm(h) + alpha * (suggestion - unit_norm(h)))
```

It is normalized linear interpolation with a learned coordinate-wise vector, not
exact trigonometric SLERP. No Muon, ReLU² replacement, custom token interpolation,
or other unrelated architectural modifications have been added.

### Details that are easy to implement incorrectly

**Normalization axes.** PyTorch stores a linear weight as `[output, input]`.
Embedding, LM-head, Q/K/V and MLP up/gate weights normalize over dimension 1.
Attention-output and MLP-down weights normalize over dimension 0. Normalizing
_every_ weight row-wise would be a different model.

**Stored versus effective scale.** Following the final paper v2 Section 2.6,
`alpha_raw` starts at `1/sqrt(d)`, while its effective forward value is 0.05.
Q/K and logit raw scales also start at `1/sqrt(d)`, with effective value 1. MLP
branch scales start at 1. Simply creating a direct trainable alpha of 0.05 changes
its effective optimizer step size, even though the initial forward pass agrees.

**Projection ordering and optimizer ownership.** The crucial loop is:

```python
(loss / accumulation).backward()   # for each microbatch
# ... after all microbatches ...
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
model.project_weights_()
```

Projection updates existing `Parameter` objects in place. With the default FP32
weights, these are the exact weights owned by Adam; there is no separate hidden
master copy to forget to project. Adam moments are not normalized/projected.

**Correct causal target shift.** The memory-efficient loop takes `[B, T+1]` tokens,
encodes the first T and predicts the following T. The HF `forward` separately follows
the standard `labels=input_ids` convention and shifts once internally. Tests compare
the chunked loss and all gradients with a dense next-token reference.

**Bounded vocabulary-head memory.** The loss uses token chunks with checkpointed
recomputation. Slicing logits into chunks without recomputation would leave all
of their backward activations alive and would not solve the memory problem.

## 6. Fidelity boundaries and paper/reference differences

The original experiments used 0.5B/1B models, OpenWebText, a 32k LLaMA-2 tokenizer,
global batch 512, and 64 A100s. The defaults here are a 4090-oriented downscaled
experiment, not a claim to recreate the original absolute loss values or speedups.
The public NVIDIA reference itself switches to GPT-2 vocabulary; this package uses
the accessible GPT-2 tokenizer but does not pad its vocabulary with unused classes.

The default profile intentionally uses **FP32 parameter storage and Adam states,
with BF16 matrix computation**, equally for both variants. NVIDIA's public README
warns that raw BF16 parameter storage can damage the GPT baseline disproportionately
and inflate the apparent nGPT advantage. To examine that public-reference precision
policy in a separate experiment:

```bash
OUT=runs/owt_reference_bf16 WEIGHT_STORAGE=reference_bf16 \
bash scripts/run_4090.sh
```

`reference_bf16` stores block linear matrices in BF16, while input/output embeddings,
scale vectors and RMSNorm parameters stay FP32. It uses ordinary PyTorch AdamW;
its BF16 parameter states are not a production FP32-master-weight optimizer. This
matches the public implementation's precision choice, **not** the unavailable
internal Megatron implementation. Treat this as a sensitivity study, not the
primary fairness result.

Other explicit choices: the MLP uses the paper's gate-only sqrt(d) factor; the
public code applies the same additional positive factor to the ungated branch,
which cancels under output normalization in exact arithmetic. This implementation
uses standard norm-preserving RoPE, native PyTorch causal SDPA instead of a separate
FlashAttention dependency, and FP32 normalization with a numerical epsilon guard.
There is no claim of bitwise equivalence to NVIDIA's illustrative nanoGPT port.

For LLaMA-2 tokenization, request access to the tokenizer repository through HF and
log in, then prepare a new dataset directory:

```bash
hf auth login
python compare_ngpt.py prepare --data data/owt_llama2_200m \
  --tokenizer meta-llama/Llama-2-7b-hf
```

Only tokenizer files are downloaded by that operation, not the LLaMA model weights.
To set paper-scale model dimensions, use `--width 1024 --layers 24 --heads 16`;
use effective batch 512 for a closer batch match. These settings are **not** the
4090 default, and this runner does not implement multi-GPU distributed training.
`torchrun` with more than one process is rejected rather than silently producing
an invalid comparison.

The newer **Training nGPT** paper (arXiv:2608.01284v2, September 2026) adds a different
optimization recipe involving GatedAdamW, logit gradient preconditioning, logarithmic
LR decay, and other mechanisms. Those changes are deliberately not folded into
this original dense-architecture experiment.

## 7. Read and regenerate plots

After training:

```text
runs/owt_4090/
  gpt_seed0/
    run.json             # exact dimensions, data fingerprints, software, device
    metrics.csv          # one row per update; evaluation fields on measured steps
    summary.json
    last.pt              # model + optimizer + RNG + history + fixed schedule spec
    final_hf/            # produced on completion; save_pretrained weights/config/tokenizer
  ngpt_seed0/
    ...
  plots/
    training_loss_vs_iterations.png
    validation_loss_vs_iterations.png
    train_probe_loss_vs_iterations.png
    validation_loss_vs_seconds.png
    ... matching SVGs
```

```bash
python compare_ngpt.py plot --out runs/owt_4090
# Optional smoothing of training minibatches only; raw CSV is unchanged:
python compare_ngpt.py plot --out runs/owt_4090 --smooth 20
```

An iteration is **one completed optimizer update**, not one accumulated microbatch.
All loss values are cross-entropy in nats per predicted token. `train_loss` is the
mean of all microbatch losses for that update, measured before the update.
`train_probe_loss` and `val_loss` are measured after the update on fixed windows,
in evaluation mode. Validation is not smoothed, and missing validation points are
not filled with invented values. The plotter rejects incompatible dimensions/data
budgets/precision policies in one comparison directory.

`summary.json` reports the best **measured** validation loss. Only the last resumable
checkpoint is retained; this does not imply the best-loss step's weights were saved.

## 8. Pause and resume without changing the LR schedule

```bash
# Keep the planned 10,000-update schedule, but pause each model at update 500:
bash scripts/run_4090.sh --stop-after-steps 500

# Resume each model with identical original settings:
RESUME=1 bash scripts/run_4090.sh
```

Use the same DATA/OUT/dimension/batch/LR flags on resume. Changing the planned total
updates after reaching zero LR is not a neutral continuation, so the runner rejects
such changes. Interrupted runs roll CSV history back to the durable checkpoint;
they do not duplicate newer rows. Exact CPU resumed-versus-uninterrupted model
weights and losses are tested. GPU bitwise reproducibility can depend on kernels,
software and hardware; `--deterministic` requests deterministic algorithms and
will report unsupported operations rather than guarantee universal equivalence.

A hard interruption can lose updates since the most recent save. The existing
`last.pt` is replaced atomically after a complete temporary checkpoint is written.

## 9. Fair learning-rate search and multiple seeds

The paper tunes initial learning rate for each method. A shared arbitrary LR is
only a starting point, particularly at reduced width/batch size. Give both models
an equal search budget:

```bash
# THREE matched pairs, each with 10,000 updates per model:
bash scripts/sweep_lr.sh

# Three initializations at a chosen pair of learning rates:
OUT=runs/owt_three_seeds SEEDS="0 1 2" \
LR_GPT=0.0015 LR_NGPT=0.003 bash scripts/run_4090.sh
```

Use validation for selection, not a final test set. Report learning curves,
tokens to a common validation threshold, and elapsed time as separate quantities.
Iteration efficiency is not automatically wall-clock efficiency. A short toy
experiment is not evidence for or against the paper's large-scale reported gains.
Multiple seeds are plotted separately; the script does not manufacture confidence
intervals from a single seed.

## 10. Other HF text datasets and saved-model loading

For example, use MiniPile's provided validation split:

```bash
python compare_ngpt.py prepare --dataset JeanKaddour/minipile \
  --validation-split validation --data data/minipile_200m \
  --train-tokens 200000000 --val-tokens 1000000
python compare_ngpt.py train --data data/minipile_200m --out runs/minipile_4090
```

When a dataset ends before a token cap, preparation records the actual counts.
The cap is not a promise of available validation tokens. The default OWT mode
hash-splits full documents before packing; identical document strings cannot cross
that split, but near-duplicate decontamination is not attempted. When a native
validation split is requested, its upstream split integrity is trusted.

Save/load through the Hugging Face auto APIs:

```python
import torch
import hf_model  # local registration, required before the Auto* calls
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "runs/owt_4090/ngpt_seed0/final_hf"
model = AutoModelForCausalLM.from_pretrained(path).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(path)
batch = tokenizer("A normalized transformer", return_tensors="pt").to("cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
print(output.loss.item(), output.logits.shape)
```

This compact adapter supports packed/unpadded causal-LM forward passes and local
HF checkpoint APIs. KV caching, padded attention masks, a generation pipeline,
Trainer integration, model sharding and distributed training are outside its scope.
Use the supplied training loop: a stock Trainer without the projection step would
not implement this nGPT recipe correctly.

## Source map

- `compare_ngpt.py`: CLI, optimizer, exact next-token loop, fixed evaluation, resume.
- `model.py`: two variants of the same dense backbone, constrained-weight axes.
- `hf_model.py`: HF configuration, AutoModel registration, save/load interface.
- `data.py`: HF streaming, tokenizer, document split, immutable token files.
- `plot_losses.py`: requested plots and time/probe diagnostics.
- `tests/`: numerical and integration checks; see `TEST_REPORT.md`.
- `SOURCES.md`: primary references and explicit version/fidelity decisions.


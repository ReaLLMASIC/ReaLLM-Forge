# Recurrent Attention-Residual Mesh

A standalone Hugging Face/PyTorch research implementation of a recurrent
all-module mesh. It includes device-targeted single-GPU profiles for NVIDIA A100,
H100 SXM, H100 PCIe, and RTX 4090; a conservative generic CUDA profile; and a
portable profile for CPU correctness checks.

Instead of a sequential Transformer stack, the model maintains parallel
attention and feed-forward branches. Every branch reads the original token
embedding at the first step. At each later recurrent step, every destination
branch independently routes over the original embedding and all attention/FFN
outputs from the preceding step. Branches then update synchronously. A final
Attention-Residuals-style router combines the embedding and terminal branch
states before RMSNorm and the tied language-model head.

## Architecture

Let `E` be the input embedding and `H_i^(t)` be branch `i` at recurrent step
`t`:

```text
H_i^(0) = Module_i(E)

H_i^(t) = Module_i(
    Router_i([E, H_1^(t-1), ..., H_M^(t-1)])
)

logits = LMHead(RMSNorm(
    OutputRouter([E, H_1^(T), ..., H_M^(T)])
))
```

The default `attnres` router uses destination-specific learned pseudo-queries,
RMS-normalized source keys, unnormalized values, and softmax attention across
the embedding/module axis.

The hardware-oriented implementation:

- Packs attention weights and FFN weights into batched tensors.
- Folds the attention-module axis into the batch for one causal SDPA call.
- Executes FFN branches with batched matrix multiplications.
- Uses fixed-length packed batches so Flash SDPA needs no padding mask.
- Supports BF16, FP16, or full precision, TF32, fused AdamW, `torch.compile`,
  and recurrent checkpoint chunks.
- Selects Flash-only, cuDNN+Flash fused-only, or automatic SDPA according to
  the device profile and probes required fused policies at the actual dtype
  and head dimension.
- Pads vocabulary dimensions to a Tensor-Core-friendly multiple.
- Can use expandable CUDA allocator segments to reduce fragmentation.

This is optimized for training experiments. Generation works without a KV
cache and is therefore not optimized for inference latency.

## Package contents

```text
recurrent_attnres_mesh/
├── README.md
├── requirements.txt
├── recurrent_attnres_mesh.py
└── scripts/
    ├── _common.sh
    ├── setup_env.sh
    ├── check_install.sh
    ├── run_inspect.sh
    ├── run_smoke.sh
    ├── run_benchmark.sh
    ├── run_train.sh
    ├── run_sweep.sh
    ├── run_untied_sweep.sh
    ├── run_branch_eval_matched.sh
    └── *_a100.sh                 # backward-compatible wrappers
```

Every run launcher resolves its project path independently of the current
working directory and forwards additional CLI arguments. Explicit CLI values
override profile defaults, so experiments do not require editing shell files.

## Installation

Requirements:

- Python 3.10 or newer
- PyTorch 2.5 or newer
- A recent NVIDIA driver for GPU profiles
- Enough local storage for Hugging Face data, compiled kernels, and outputs

From the extracted project directory:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
./scripts/check_install.sh
```

The setup script installs the pinned PyTorch 2.5.1 CUDA 12.4 wheel. Override
the official PyTorch channel and version when required by the host driver:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXXX \
TORCH_VERSION=X.Y.Z \
  ./scripts/setup_env.sh
```

Or install into an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Hardware profiles

Select a target by setting `HARDWARE_PROFILE`. The default is `a100-80gb`.

```bash
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_inspect.sh
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_benchmark.sh
```

The neutral launchers also recognize trailing `--hardware-profile VALUE` and
`--hardware-profile=VALUE` forms. A CLI selection overrides the environment,
and the last CLI occurrence wins for the model, output naming, and sweep batch
policy.

| Profile | Intended device | D / heads / FFN | A / F / steps | Seq / micro / accum | Precision | SDPA | Memory policy |
|---|---|---|---|---|---|---|---|
| `a100-80gb` | A100 80 GB, SM80 | 2048 / 32 / 5504 | 4 / 4 / 6 | 2048 / 8 / 2 | BF16 | Flash | 90%, 8 GiB reserve |
| `h100-sxm-80gb` | H100 SXM 80 GB, SM90 | 2048 / 32 / 5504 | 4 / 4 / 6 | 2048 / 8 / 2 | BF16 | fused | 90%, 8 GiB reserve |
| `h100-pcie-80gb` | H100 PCIe 80 GB, SM90 | 2048 / 32 / 5504 | 4 / 4 / 6 | 2048 / 8 / 2 | BF16 | fused | 90%, 8 GiB reserve |
| `rtx4090-24gb` | RTX 4090 24 GB, SM89 | 1024 / 16 / 2752 | 4 / 4 / 4 | 1024 / 4 / 8 | BF16 | Flash | 85%, 3 GiB reserve |
| `cuda-generic` | Unrecognized CUDA GPU | 768 / 12 / 2048 | 2 / 2 / 4 | 1024 / 2 / 16 | FP16 | auto | 80%, 2 GiB reserve |
| `portable` | CPU or diagnostic run | 256 / 8 / 688 | 1 / 1 / 2 | 256 / 1 / 1 | full | auto | no strict GPU contract |

`A` and `F` are the attention-branch and FFN-branch counts. All named GPU
profiles retain a nominal 32,768 tokens per optimizer update. The A100 and both
H100 profiles intentionally use the same approximately 305M-parameter model
and training batch, making their throughput and memory results directly
comparable. The 4090 profile is an approximately 102M-parameter screening
configuration and targets at most about 20 GiB in normal 24 GB-card conditions.

Dense BF16 utilization uses profile-specific, non-sparse peak denominators:
312 TFLOP/s for A100, 989 TFLOP/s for H100 SXM, 756 TFLOP/s for H100 PCIe,
and 165.2 TFLOP/s for RTX 4090. Generic and portable profiles leave this
denominator unset instead of reporting a misleading MFU. These are not FP8 or
2:4 sparsity peaks.

The named A100, H100, and 4090 profiles are strict: their preflight checks the
GPU name, compute capability, visible memory, precision support, single-process
launch, and free-memory reserve before expensive work. It also executes a small
capability probe for the configured fused-SDPA policy using the selected dtype
and head dimension. This rejects undersized MIG slices and obvious backend
mismatches, but it is not proof that every production sequence/batch shape fits
or uses the same dispatched kernel. `cuda-generic` targets CUDA without assuming
a model name or fixed HBM capacity; it uses FP16 and automatic SDPA so older
CUDA hardware has a useful starting point. `portable` is for CPU smoke tests and
explicit non-contract experiments.

To deliberately allow an unfused fallback while retaining a named profile,
change both controls; disabling only the requirement still leaves the explicit
backend policy active:

```bash
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_benchmark.sh \
  --no-require-fused-sdpa \
  --sdpa-backend auto
```

Select one visible GPU when a machine contains several:

```bash
CUDA_VISIBLE_DEVICES=1 \
HARDWARE_PROFILE=h100-pcie-80gb \
  ./scripts/run_benchmark.sh
```

The launchers intentionally use one process and one GPU. They reject
`WORLD_SIZE` values other than one and multi-device `CUDA_VISIBLE_DEVICES`
lists.

## Device-specific behavior

### A100 80 GB

The established baseline is D=2048, four attention and four FFN branches, six
steps, sequence length 2048, BF16, TF32, required Flash SDPA, fused AdamW,
compilation, and two-step recurrent checkpoint chunks. It targets 90% of HBM
while retaining an 8 GiB free-memory reserve.

### H100 SXM and PCIe 80 GB

Both H100 profiles keep the exact A100 scientific configuration rather than
silently spending H100 throughput on a larger model. Hopper-specific benefits
therefore appear as speed and headroom, not a changed architecture. The runtime
enables TF32 and cuDNN autotuning where PyTorch operations can use them,
requests fused AdamW, and uses a `fused` SDPA policy that enables only cuDNN and
Flash attention while excluding non-fused fallback. PyTorch's dispatcher
selects between those allowed fused backends; the combined policy does not
promise cuDNN priority. H100 SXM and PCIe use separate names so reported results
retain the deployment form factor even though this single-GPU workload does not
depend on NVLink.

CUDA device properties do not reliably distinguish H100 SXM from H100 PCIe.
Select the correct profile from the actual host SKU: both enforce H100/SM90/80
GB, but they use different dense-BF16 peak denominators for MFU.

After measuring the pinned baseline, H100-only capacity experiments can raise
branch count, recurrent steps, sequence length, or microbatch explicitly.
Benchmark each changed point; additional compute capability does not create
additional HBM on an 80 GB H100.

For a dedicated H100 kernel ablation, compare the combined policy against each
fused backend on the same shape:

```bash
for backend in fused cudnn flash; do
  OUTPUT_DIR="$PWD/runs/h100_${backend}" \
  HARDWARE_PROFILE=h100-sxm-80gb \
    ./scripts/run_benchmark.sh \
    --sdpa-backend "$backend" \
    --require-fused-sdpa
done
```

The dedicated `cudnn` and `flash` policies isolate a backend; `fused` lets the
dispatcher choose from both. Either dedicated policy may fail the early
capability probe on an unsupported software build or head dimension. That is a
useful ablation result rather than a silent fallback, but a passing probe still
does not prove full training-shape dispatch.

### RTX 4090 24 GB

The 4090 profile reduces width, recurrent depth, and sequence length, enables
checkpointing and Flash SDPA, and uses microbatch 4 with accumulation 8. Its
85%-of-HBM cap plus 3 GiB reserve is designed to remain near or below 20 GiB,
leaving room for the display/desktop and allocator variation. For a display
GPU, close other CUDA applications before the benchmark and keep the reserve;
strict preflight uses currently free memory, not only the card's advertised
capacity.

### Generic CUDA

Use `cuda-generic` for a CUDA GPU without a device-targeted profile. It starts with
a smaller model, FP16, and automatic SDPA. This prioritizes broad
compatibility. Once the probe passes, opt into supported features individually:

```bash
HARDWARE_PROFILE=cuda-generic ./scripts/run_benchmark.sh \
  --mixed-precision bf16 \
  --sdpa-backend fused \
  --require-fused-sdpa
```

Only do this when the device reports BF16 and the fused-SDPA probe succeeds.

### Portable / forced CPU

The portable profile is intentionally small and disables compilation,
checkpointing, fused AdamW, mixed precision, and required fused SDPA. To make a
CPU check deterministic even on a GPU host, hide CUDA explicitly:

```bash
CUDA_VISIBLE_DEVICES=-1 \
HARDWARE_PROFILE=portable \
  ./scripts/run_benchmark.sh
```

`run_smoke.sh` explicitly selects portable execution independently of the
environment default; as with the other launchers, a trailing CLI profile is an
intentional override.

### FP8 scope

FP8 is deliberately not used by any baseline. The current model CLI exposes
`bf16`, `fp16`, and full precision; mixing an FP8 framework or Transformer
Engine path into only the H100 condition would confound hardware and numerical
format. Treat FP8 as a separate, explicitly labeled ablation with its own
accuracy, convergence, throughput, and memory comparisons against BF16 on the
same H100.

## First run

Inspect the resolved profile without allocating the model:

```bash
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_inspect.sh
```

Run the comprehensive tiny-model correctness suite (portable on every host):

```bash
./scripts/run_smoke.sh
```

The smoke suite covers forward/backward, causality and padding, router
gradients, checkpoint equivalence, attention-only and FFN-only models, tied and
untied recurrence, router controls, padded vocabulary handling, generation,
resizing, and fresh-process Hugging Face `AutoModel` loading.

Run the full synthetic training-step benchmark on the selected GPU:

```bash
HARDWARE_PROFILE=a100-80gb ./scripts/run_benchmark.sh
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_benchmark.sh
HARDWARE_PROFILE=h100-pcie-80gb ./scripts/run_benchmark.sh
HARDWARE_PROFILE=rtx4090-24gb ./scripts/run_benchmark.sh
```

Short capacity probe:

```bash
HARDWARE_PROFILE=rtx4090-24gb \
BENCHMARK_WARMUP=1 BENCHMARK_STEPS=1 \
  ./scripts/run_benchmark.sh
```

The benchmark includes forward, cross-entropy, backward, gradient
accumulation, clipping, and optimizer update. It separates compiler/warmup,
steady-state, and capacity memory. Key output fields include tokens/second,
estimated training TFLOP/s, `estimated_dense_bf16_mfu` where defined, allocated and
reserved peak HBM, profile memory budget, SDPA policy/probe/dtype, exact
parameter count, and branch-evaluation counts.

## Fair A100 versus H100 baseline

Use the same data/tokenizer revisions, seed, step count, and CLI arguments. The
three 80 GB profiles already share model shape, sequence length, microbatch,
accumulation, BF16, checkpoint, and compile settings:

```bash
COMMON_ARGS=(--seed 42 --benchmark-warmup 5 --benchmark-steps 50)

OUTPUT_DIR="$PWD/runs/a100_baseline" \
HARDWARE_PROFILE=a100-80gb \
  ./scripts/run_benchmark.sh "${COMMON_ARGS[@]}"

OUTPUT_DIR="$PWD/runs/h100_sxm_baseline" \
HARDWARE_PROFILE=h100-sxm-80gb \
  ./scripts/run_benchmark.sh "${COMMON_ARGS[@]}"
```

Do not compare the default 4090 or generic profile as if it were the same
architecture. To test one model across unlike memory sizes, explicitly repeat
all architecture arguments and preserve `sequence_length × microbatch ×
accumulation`; first run a one-step capacity probe on the smaller device.

## Train on TinyStories

The neutral launcher defaults to `roneneldan/TinyStories`, GPT-2 tokenization,
1,000 optimizer steps, and a unique output directory:

```bash
HARDWARE_PROFILE=h100-sxm-80gb ./scripts/run_train.sh
```

Configure it through environment variables and trailing model arguments:

```bash
DATASET=roneneldan/TinyStories \
TOKENIZER=gpt2 \
MAX_STEPS=5000 \
OUTPUT_DIR="$PWD/runs/h100_tinystories_5k" \
HARDWARE_PROFILE=h100-sxm-80gb \
  ./scripts/run_train.sh --save-interval 500
```

Pin immutable Hugging Face revisions for publication runs:

```bash
DATASET_REVISION=<commit-sha> \
TOKENIZER_REVISION=<commit-sha> \
HARDWARE_PROFILE=a100-80gb \
  ./scripts/run_train.sh
```

Local text data is supported:

```bash
HARDWARE_PROFILE=rtx4090-24gb ./scripts/run_train.sh \
  --train-file data/train.txt \
  --eval-file data/validation.txt \
  --output-dir runs/local_data
```

Each training run requires a new or empty output directory, preventing stale
metrics or checkpoints from being mistaken for a new experiment. The launcher
does not hardcode Accelerate mixed precision; the model's selected profile or
your trailing `--mixed-precision` argument is authoritative.

## Ablation launchers

### Tied module/step grid

```bash
HARDWARE_PROFILE=h100-sxm-80gb MAX_STEPS=1000 \
  ./scripts/run_sweep.sh
```

The default 45-run grid uses attention modules `1,2,4`, FFN modules `1,2,4`,
steps `1,2,4,6,8`, AttnRes routing, and tied iteration weights. Include
attention-only and FFN-only controls with `FULL_GRID=1`; the invalid
`A=0,F=0` point is filtered automatically.

```bash
FULL_GRID=1 HARDWARE_PROFILE=a100-80gb MAX_STEPS=1000 \
  ./scripts/run_sweep.sh
```

Preview or customize the axes:

```bash
ATTENTION_MODULES=1,2 \
FFN_MODULES=1,2,4 \
ITERATIONS=2,4,8 \
HARDWARE_PROFILE=rtx4090-24gb \
  ./scripts/run_sweep.sh --sweep-dry-run
```

Sweeps use profile-aware, memory-safer batches. GPU profiles preserve 32,768
nominal tokens/update:

| Sweep | A100/H100 | RTX 4090 | Generic/portable |
|---|---|---|---|
| Tied | micro 4 / accum 4 | micro 2 / accum 16 | micro 1 / accum 32 |
| Untied | micro 2 / accum 8 | micro 1 / accum 32 | micro 1 / accum 32 |

The portable profile deliberately uses sequence length 256, so its `1/32`
sweep pair is an 8,192-token CPU diagnostic batch rather than a publication
GPU batch.

Override either value without editing the launcher:

```bash
SWEEP_MICRO_BATCH_SIZE=1 \
SWEEP_GRADIENT_ACCUMULATION_STEPS=32 \
HARDWARE_PROFILE=rtx4090-24gb \
  ./scripts/run_sweep.sh
```

### Untied recurrence

```bash
HARDWARE_PROFILE=h100-pcie-80gb MAX_STEPS=1000 \
  ./scripts/run_untied_sweep.sh
```

Untied depth multiplies branch parameters. The launcher keeps gradient
checkpointing enabled and selects the safer batch pair shown above. Start with
a dry run and lower microbatch while raising accumulation for unusually large
points.

### Branch-evaluation-matched comparison

```bash
HARDWARE_PROFILE=a100-80gb MAX_STEPS=1000 \
  ./scripts/run_branch_eval_matched.sh
```

This sequentially runs `(A=1,F=1,S=8)`, `(A=2,F=2,S=4)`, and
`(A=4,F=4,S=2)`. Every point performs eight attention and eight FFN branch
evaluations per token. Routing/readout costs and parameter counts still differ,
so this is not exact total-FLOP matching; normalize with the analytical FLOP
fields. Screening summaries are saved, but models are not.

For fair architecture comparisons, fix sequence length, optimizer steps,
effective tokens/update, data revisions, tokenizer revision, seed, precision,
and hardware profile. Module count changes parameters and compute. With tied
weights, recurrent depth changes compute with little change to heavy branch
parameters. Untied depth grows branch parameters approximately linearly.

## Compatibility commands

The original commands remain available:

```bash
./scripts/run_benchmark_a100.sh
./scripts/run_train_a100.sh
./scripts/run_sweep_a100.sh
./scripts/run_untied_sweep_a100.sh
./scripts/run_branch_eval_matched_a100.sh
```

They are thin wrappers whose default profile is `a100-80gb`. New automation
should use the neutral filenames.

## Useful architecture controls

```text
--input-injection initial_only     Embed only at the initial step
--no-share-iteration-weights      Separate branch weights per step
--use-step-embeddings             Add learned recurrent-step identities
--branch-output residual          Branch produces an internal residual update
--exclude-self                    Router cannot select its own prior state
--router-type uniform             Uniform-routing control
--router-type static              Learned input-independent routing control
--router-type identity            No cross-module routing control
--router-heads 4                  Multi-head routing over the module axis
```

Run `python recurrent_attnres_mesh.py --help` for the complete CLI.

## Output files

| Path | Contents |
|---|---|
| `run_args.json` | Fully resolved arguments |
| `model_config.json` | Hugging Face model configuration |
| `hardware.json` | GPU, memory, precision, and SDPA policy/preflight |
| `metrics.jsonl` | Loss, evaluation, throughput, FLOPs, utilization, and memory |
| `summary.json` | Final comparable run summary |
| `checkpoint-*` | Optional Accelerate model/optimizer/RNG/data state |
| `model/` | Saved Hugging Face model, tokenizer, config, and custom source |
| `failure.json` | Structured CUDA OOM details and smaller-batch suggestion |

Sweeps add `sweep_results.csv`, one directory and `sweep_run.json` per point,
and `sweep_dry_run.json` for previews. CUDA OOMs are recorded and the grid
continues; other child failures stop it. Add `--sweep-stop-on-oom` to stop at
the first OOM.

## Memory and performance tuning ladder

Profile defaults are conservative starting points, not guarantees for every
untied, long-context, or high-module-count ablation. Tune in this order:

1. Run `BENCHMARK_WARMUP=1 BENCHMARK_STEPS=1` on the exact configuration.
2. Check both peak *allocated* and peak *reserved* HBM against the logged
   budget; reserved memory is the safer capacity signal.
3. Halve microbatch and double accumulation to preserve tokens/update.
4. Keep recurrent gradient checkpointing enabled.
5. Screen tied weights before large untied experiments.
6. Reduce sequence length only if the scientific comparison permits it.
7. Then reduce recurrent steps, branch counts, or model width.
8. Use `--no-compile` only to isolate compilation failures.

For A100/H100 throughput tuning, benchmark microbatch/accumulation pairs
`2/8`, `4/4`, `8/2`, and `16/1`; at sequence length 2048 they all preserve
32,768 tokens/update. Use at least 10 warmup and 50 measured steps for final
reported throughput. For long finalist runs with headroom, compare compile
`default` against `max-autotune-no-cudagraphs`; retain the faster measured
policy rather than assuming the more aggressive mode wins.

Example 4090 fallback preserving 32,768 tokens/update:

```bash
HARDWARE_PROFILE=rtx4090-24gb ./scripts/run_benchmark.sh \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 16
```

Do not disable fused SDPA merely to handle an OOM; a fallback attention backend
may use more memory. With measured headroom, try
`--compile-mode max-autotune-no-cudagraphs` and benchmark before/after.
Disabling checkpointing can improve speed but retains more activations and
must be capacity-tested.

## Checkpoints and exact resume

```bash
OUTPUT_DIR="$PWD/runs/checkpointed" \
HARDWARE_PROFILE=h100-sxm-80gb \
  ./scripts/run_train.sh --save-interval 100

OUTPUT_DIR="$PWD/runs/resumed" \
HARDWARE_PROFILE=h100-sxm-80gb \
  ./scripts/run_train.sh \
  --resume-state "$PWD/runs/checkpointed/checkpoint-00000500"
```

Exact resume verifies model, data/tokenizer identity, batch and worker
settings, optimizer, schedule, precision, compile/checkpoint policy, seed,
process count, and stored data cursor. Keep `--max-steps` identical to the
original run.

## Hugging Face loading

Saved models include custom source and `auto_map` metadata. When directly
loading a GPU-produced checkpoint for portable/CPU execution, override the
saved execution-only SDPA controls before constructing the model:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

path = "runs/tinystories/model"
tokenizer = AutoTokenizer.from_pretrained(path)
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
config.sdpa_backend = "auto"
config.require_fused_sdpa = False
config.require_flash = False  # backward-compatible saved-config field
model = AutoModelForCausalLM.from_pretrained(
    path,
    config=config,
    trust_remote_code=True,
)
```

The custom architecture disables KV caching; generation recomputes the full
prefix. In the training CLI, `--load-model` retains the checkpoint's learned
architecture while applying execution policy from the active
`--hardware-profile`, including SDPA/checkpoint behavior and the run's selected
precision/compile settings.

## Hardware/software references

- [NVIDIA A100 Tensor Core GPU architecture](https://resources.nvidia.com/en-us-tensor-core/nvidia-ampere-architecture-whitepaper)
- [NVIDIA H100 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA Hopper architecture whitepaper](https://resources.nvidia.com/en-us-tensor-core/nvidia-hopper-architecture-whitepaper)
- [NVIDIA Ada GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [PyTorch 2.5 and cuDNN SDPA on H100](https://pytorch.org/blog/pytorch2-5/)
- [PyTorch `sdpa_kernel` backend-selection API](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- [NVIDIA Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

## Validation status

The source and launchers are covered by syntax/lint checks, the complete smoke
suite, eager and compiled accumulated benchmarks, an Accelerate train/evaluate
probe, profile/CLI/run-config precedence, analytical parameter-count checks,
hardware-preflight simulations, allocator merging, structured OOM handling,
and sweep dry runs.

Physical GPU validation is still required for actual fused-SDPA selection,
fused AdamW, CUDA compilation, peak-memory fit, throughput, and utilization on
each A100/H100/4090 system. Run the one-step benchmark before a full grid.

# Tangent-linear ReLU² attention: separate experimental versions

This release adds two variants while leaving every optimized ReLU2Max v4 file
unchanged. Importing `hf_model` still selects the original classes. Import
`hf_model.relu2linear` to opt into the new classes. This changes the attention
score activation only; the MLP activation and the rest of the architecture
retain their existing settings.

## Exact function and threshold placement

For a fixed positive threshold `t`:

```text
f_t(x) = 0                 when x <= 0
         x*x               when 0 < x <= t
         2*t*x - t*t       when x > t

f'_t(x) = 2 * min(max(x, 0), t)
```

At `t=8`, the linear tail is `16*x - 64`. Value and derivative are continuous:
both branches give 64 at x=8 and slope 16. The second derivative changes there;
this is C¹, not C². The zero join also has a continuous first derivative.

The complete attention operation is
`(f_t(scale * (Q @ K.T)) / divisor) @ V`, with causal/padding masks.
The threshold applies **after the QK scale and before the divisor**. When QK
normalization is enabled, it acts on the resulting scaled normalized scores.
There is no row-sum normalization, matching the original ReLU2Max definition.
Dividing by sequence length, when enabled, remains part of the divisor.

Both implementations cap before squaring:
`r=max(x,0); c=min(r,t); f=c*c + (2*t)*(r-c)`.
The linear tail never evaluates the large `x*x` intermediate. Backward uses
`2*c`, including exactly at t. Thresholds must be positive finite host numbers;
values that vanish in FP32 or whose squared threshold overflows FP32 are rejected.
FP32 arithmetic rounds nonrepresentable constants normally (e.g. 3.7).

## Fixed8 and configurable APIs

```python
from hf_model.relu2linear import (
    NanoGPTReLU2LinearConfig,
    NanoGPTReLU2LinearForCausalLM,
)

# Hardcoded, fully fused threshold 8.
fixed_config = NanoGPTReLU2LinearConfig(
    relu2linear_mode="fixed8",
    relu2max_accelerator="auto",
    attention_dropout=0.0,
)
fixed_model = NanoGPTReLU2LinearForCausalLM(fixed_config)

# User choice, fixed before constructing/training the model.
flex_config = NanoGPTReLU2LinearConfig(
    relu2linear_mode="configurable",
    relu2linear_threshold=6.5,  # fractional / non-power-of-two is supported
    relu2max_accelerator="auto",
    attention_dropout=0.0,
)
flex_model = NanoGPTReLU2LinearForCausalLM(flex_config)
```

Set architecture dimensions to your existing model's dimensions for comparisons.
The default examples above construct full-size models. For tiny smoke tests,
use the commands below. The fixed8 configuration rejects a threshold other than
8; the configurable mode accepts 8 too, providing a matched control.

Attention modules snapshot the mode/threshold at construction and bind the
configurable kernel once. Do not mutate these config fields on a live model:
construct a new model/config to run a different threshold. Do not change them
between forward and backward, checkpoint recomputation, or CUDA graph replay.
Threshold is a hyperparameter, not a learnable tensor. Both fields are saved in
`config.json`. The original `relu2max_divisor`, sequence-length divisor option,
and `relu2max_accelerator` field names are retained for matched experiments.

Direct kernel use:

```python
from hf_model.relu2linear import (
    triton_relu2linear8_attention,
    triton_relu2linear_attention,
    make_triton_relu2linear_attention,
)

out = triton_relu2linear8_attention(q, k, v, scale=scale, divisor=256.0)
out = triton_relu2linear_attention(q, k, v, scale=scale, threshold=6.5)

# Preferred for repeated training calls: validate/bind once on the host.
attention = make_triton_relu2linear_attention(threshold=6.5)
out = attention(q, k, v, scale=scale, divisor=256.0)
```

## What is fused and supported

The kernels fuse QK, masking, the activation/divisor, and multiplication by V
inside tensor-core tiles. Forward/backward never materialize the full quadratic
attention matrix. Backward recomputes tiles, owns dQ and dK/dV independently,
and uses no gradient atomics. As in v4, decoding can split KV work and reduce
partial outputs; a learned scalar scale also needs a final reduction. Fusion
does not mean that all of training runs in one GPU launch.

The supported CUDA path is NVIDIA SM80+ (including the RTX 4090), common
FP16/BF16/FP32 Q/K/V dtypes, head size up to 256, unit feature stride and arbitrary
outer strides. Q and KV lengths may differ; bottom-right causal alignment and
nonzero-keeps `[batch, keys]` masks are supported. Q/K/V and a learned scalar
scale have first-order gradients, including negative/zero scale values. FP32
uses TF32x3; mixed precision casts tile weights/derivatives for tensor cores.
Higher-order fused gradients and attention dropout are not implemented.

HF automatically chooses the fused path for supported tensors with zero
training attention dropout. CPU, unsupported shapes/devices, and nonzero
training attention dropout use the differentiable PyTorch fallback; `"torch"`
forces that fallback. The fallback computes scores and activations in FP32 for
FP16/BF16 inputs. The standalone fused functions raise on unsupported inputs.
The CUDA path still uses floating point: no INT8/INT4 or fixed-point kernel is
provided by this change.

## Hugging Face compatibility

Tested with Transformers **4.44.2** and PyTorch **2.11.0+cpu**, including:

- Direct model/config construction and strict baseline state-dict loading.
- Local `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` after importing
  `hf_model.relu2linear`.
- `save_pretrained` / `from_pretrained`, including threshold/mode persistence.
- Saved code loading in a fresh process via
  `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.
- `Trainer`, Muon comparison training, loss/backprop, gradient checkpointing,
  tuple KV caching, greedy generation and beam search.
- BF16 CPU autocast fallback.

The opt-in entry point registers custom code for saved-model Auto loading.
This follows Hugging Face's [custom model integration](https://huggingface.co/docs/transformers/en/custom_models).
Save the models to separate paths. Model parameter keys/shapes and seeded
initialization match the baseline; changing the activation changes the model's
function above the threshold even when weights match.

Newer Transformers major versions, CUDA mixed-precision training, distributed
training, `torch.compile`, and external serving engines have not been certified
by this CPU-only validation. In particular, this code retains the v4 tuple-cache
API; newer HF cache interfaces may need an adapter. It does not register itself
as an SDPA/FlashAttention softmax backend.

## GPU performance and configurable thresholds

Both modes specialize threshold as `tl.constexpr`. There is no runtime GPU
threshold tensor, threshold load, or branch based on a device scalar. Each new
threshold triggers its own compilation/autotuning; warm forward and backward
before timing. HF uses the pre-bound callable, so it also avoids repeating host
threshold validation on the hot path.

Offline compilation with Triton 3.6.0 produced 81 kernel specializations across
SM80/SM89/SM90, FP16/BF16/FP32, head sizes 64/128/256, prefill and split decoding.
For all 27 matched comparisons:

- Fixed8 and configurable8 had identical PTX after removing debug locations.
- Configurable3.7 used the same opcode counts as configurable8.
- Selected compilation tiles met the target's shared-memory limit.

This supports **no inherent GPU arithmetic penalty from configurability or a
fractional threshold** in the tested specializations. It is not a timing result.
Compared with ordinary ReLU², the forward activation adds min/subtract/FMA work
inside the existing tiled matmuls; backward adds the derivative cap. Its speed
relative to v4 must be measured on the 4090. No claim of global optimality,
training-quality equivalence, or new measured speedup is made.

Run the GPU comparison from this version's root directory:

```bash
bash scripts/benchmark_hf_relu2linear_4090.sh --check-install
python -m pytest tests/test_triton_relu2linear_attention.py -q
bash scripts/benchmark_hf_relu2linear_4090.sh \
  --threshold 6.5 --output-dir runs/relu2linear_4090_t6p5
```

It compares SDPA, untouched fused ReLU², fixed8, configurable8, and configurable
6.5 on identical inputs, including QK normalization and a learned scale. Override
`--threshold 3.7` for a nonbinary fractional constant. Default timings include
forward, backward and forward+backward; both CUDA events and bounded CUDA graphs
are used. Graph measurements use v4's process isolation and replay checks.

Each method is checked against the corresponding independent, chunked FP32
reference. **SDPA and different thresholds are different functions**, not mutual
correctness oracles. Failures are recorded, and unverified results get no speedup.
Outputs include JSON/CSV, allocator memory, and the fraction of allowed scores in
the linear tail. Those activation statistics describe synthetic benchmark data,
not a trained-model calibration. The script's eager variant is optional:
`--methods relu2linear_eager relu2linear_configurable`.

For a shorter first pass:

```bash
bash scripts/benchmark_hf_relu2linear_4090.sh \
  --lengths 256 1024 --query-lengths --timing events \
  --modes forward forward_backward --warmup 3 --repeats 10 \
  --output-dir runs/relu2linear_quick
```

## Matched training trials

Use the same CUDA-enabled PyTorch/Triton installation as the successful v4 run.
The test environment used Triton 3.6.0. Install remaining Python dependencies if
needed; do not replace a working CUDA PyTorch with a CPU wheel:

```bash
python -m pip install "transformers==4.44.2" accelerate datasets pytest
python scripts/compare_hf_relu2linear.py --pack-text --bf16 \
  --thresholds 8 6.5 3.7 --max-steps 1000 \
  --output-dir runs/relu2linear_training
```

This creates separate original-ReLU², fixed8, and configurable-threshold runs,
with matched data, architecture, initialization seed, Trainer sampler seed,
Muon optimizer and learning-rate schedule. It refuses to overwrite a previous
run directory. Resume requires `--resume-from-checkpoint` and compatible saved
architecture/activation settings. Compare loss/perplexity and downstream tasks
alongside speed; repeat across seeds for a quality conclusion.

Offline integration test without a tokenizer/dataset download:

```bash
python scripts/compare_hf_relu2linear.py --synthetic-smoke \
  --block-size 8 --hidden-size 32 --intermediate-size 64 \
  --layers 2 --heads 2 --kv-heads 2 --max-steps 2 --batch-size 2 \
  --dataloader-workers 0 --warmup-steps 0 --thresholds 8 3.7 \
  --output-dir runs/relu2linear_smoke
```

Random-token smoke losses are only integration checks, not quality results.

## Hardware implications

For x>t, the value grows linearly and the derivative is bounded by 2t. At t=8,
the tail can be expressed as a shift by four bits followed by subtracting 64 in
an integer representation with a suitable scale. The quadratic branch is
restricted to inputs up to 8. A power-of-two threshold also gives a power-of-two
slope; a general threshold can require a constant multiplier and coefficient
rounding in fixed-point hardware. Non-power-of-two does not necessarily mean a
costly multiplier: some constants have cheap shift/add implementations.

Example unscaled values (before the attention divisor):

| x | Original ReLU² | Linear tail at 8 |
|---:|---:|---:|
| 8 | 64 | 64 |
| 16 | 256 | 192 |
| 32 | 1024 | 448 |
| 64 | 4096 | 960 |
| 128 | 16384 | 1984 |
| 256 | 65536 | 4032 |

At an input maximum of 256, exact unsigned integer storage of these example
outputs needs 17 bits versus 12 bits. This illustrates **range reduction**, not
an established quantization bit budget. Fractional precision, signed downstream
values, scaling and rounding still matter. The function is unbounded; it does
not cap QK accumulation or the sum over values, and it does not automatically
reduce the precision needed by the rest of the network. Measure trained-model
activation ranges and quantization error before selecting hardware widths.

## Reproduce validation without CUDA

```bash
OMP_NUM_THREADS=1 python -m pytest tests -q
TRITON_INTERPRET=1 python scripts/check_relu2linear_attention_interpreter.py
python scripts/compile_relu2linear_attention.py --output runs/relu2linear_compile.json
```

The interpreter exercises actual forward, dQ/dScale, dK/dV and split-reduction
kernel code against float64 autograd, including masks, signed/zero scales,
rectangular tails and partial gradients. It cannot validate GPU tensor-core
rounding or runtime speed. The CUDA tests remain the required GPU check.

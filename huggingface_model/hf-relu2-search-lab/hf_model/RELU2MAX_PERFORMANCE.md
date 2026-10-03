# Fused ReLU2Max attention

## Why elementwise fusion leaves a large gap

The previous `triton_relu2max` kernel only replaced the elementwise
`relu(scores).square() / divisor`. The model still performed two independent
PyTorch matrix multiplications and wrote/read dense `[B,H,M,N]` scores, masks,
and weights. Faster elementwise arithmetic cannot remove that traffic.

The new `triton_relu2_attention` implementation computes the entire attention
operation in tiles, keeping scores and weights on chip. This follows the
memory-traffic principle behind
[FlashAttention](https://arxiv.org/abs/2205.14135) and the
[Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
ReLU2Max additionally needs no softmax row maximum, exponential, row sum, or
online normalization correction.

The reported 1.72x and 10.09x figures were supplied with the optimization
request; they have not been reproduced here. Neither those ratios nor a
smaller operation count establish a speedup for the new kernel. Measure on
the target GPU, at the original batch/head/sequence shape and dtype.
The subsequent user-supplied RTX 4090 run and proposed next experiments are
analyzed in [RTX 4090 feedback and next steps](RELU2MAX_NEXT_STEPS.md).
The later [complete graph run](RELU2MAX_4090_RESULTS.md) includes the
4096-token results and version-4 benchmark repairs.

## Operation and backward

For scale `s`, divisor `d`, and keep mask `C`, the operation remains:

```text
S = s * (Q @ K.T)
P = where(C, relu(S)**2 / d, 0)
O = P @ V
```

`P` is not normalized to sum to one. With output gradient `G`:

```text
dP     = G @ V.T
dS     = where(C, 2 * relu(S) * dP / d, 0)
dQ     = s * dS @ K
dK     = s * dS.T @ Q
dV     = P.T @ G
dscale = sum(dS * (Q @ K.T))
```

Forward holds a query tile and accumulates its output across key/value tiles.
Launch configurations are autotuned from a bounded set of tile sizes, warp
counts, and pipeline stages, with conservative pruning for FP32/large heads.
Causal loops skip future tiles and apply the per-element causal comparison
only along the diagonal band. Short queries with long caches split the key
sequence across multiple blocks and reduce their FP32 partial outputs, to
improve occupancy during decoding.
Backward recomputes the needed score tiles, with disjoint output ownership
for query gradients and key/value gradients. It does not save a quadratic
attention tensor or atomically accumulate Q/K/V gradients. The learned-scale
gradient uses unscaled dot products, so it works for negative and zero scales.

Causality is `key_index <= query_index + causal_offset`. The default offset
is `key_length - query_length`, matching cached decoding. Padding masks are
two-dimensional `[batch, key_length]` keep-nonzero masks, not additive biases.
Masking applies after scaling, including when the learned scale is negative.

## Model integration and limits

Existing `auto` and `triton` configurations select fusion before creating
scores when the input is supported. `torch` retains the dense reference.

- NVIDIA Ampere or newer, with Triton installed; FP16/BF16/FP32.
- Equal Q/K/V batch/head counts and feature dimensions, head dimension <=256,
  positive sequence lengths, and contiguous feature dimension. Outer strides
  may be noncontiguous, as in the model's reshape/transpose projection path.
- The model expands grouped K/V heads before dispatch, preserving its existing
  approximately-even mapping. Fusion does not yet eliminate that expansion.
- Effective attention dropout must be zero. Evaluation can use fusion even
  when the configured training dropout is nonzero. Training dropout retains
  the existing implementation and its RNG behavior.
- Unsupported full-attention inputs retain elementwise acceleration when
  available. Explicit `triton` still errors when CUDA/Triton is unavailable.
- First-order reverse-mode gradients are supported; higher-order gradients
  are not supported by the fused custom autograd function.

FP32 accumulation and tensor-core operand rounding differ from eager
low-precision intermediate rounding. They implement the same mathematical
operation, not bitwise-identical floating-point steps. In particular, BF16
model comparisons need BF16-appropriate tolerances. Inputs and scale must be
finite and within the selected dtype's useful range; an unnormalized
attention rule can produce large outputs as context or scale grows.

Tuning and JIT compilation are specialized by shape. A previously unseen KV
cache length can incur startup cost during autoregressive generation, even
when its steady-state kernel is fast. These benchmarks warm each exact shape;
they do not include that latency. Static/bucketed cache dispatch and tuning
policy would need a separate generation-throughput evaluation.

The benchmark measures the attention operation after Q/K/V projection and
normalization. End-to-end language-model throughput also includes projection,
RoPE, QK normalization, MLP, vocabulary head, optimizer, and data loading.
The existing pretraining softmax control remains the eager implementation;
the new microbenchmark separately exposes SDPA as a performance reference.

## Validation and benchmarking

The downloadable bundle includes the original elementwise baseline and the
supporting model/test imports. Its `changed-files/` directory is now a complete
working directory for the supplied benchmark and tests; keep `hf_model/`,
`scripts/`, `tests/`, and `train_variations/` beside each other. Do not flatten
these directories. The earlier archive omitted unchanged dependencies.

Check the extracted bundle before starting the GPU run:

```bash
cd changed-files
python scripts/benchmark_hf_relu2max.py --check-install
```

This checks the selected kernel files and imports on CPU or CUDA systems; it
does not measure performance or run GPU kernels.

Run correctness tests before collecting reportable timing results:

```bash
python -m pytest tests/test_hf_nanogpt.py tests/test_triton_relu2_attention.py -q
python scripts/benchmark_hf_relu2max.py --help

# Optional CPU check of the actual Triton kernels' FP32 math and indexing.
TRITON_INTERPRET=1 python scripts/check_relu2_attention_interpreter.py
```

An RTX 4090 sweep (also runnable on A100/H100) needs only PyTorch and Triton;
it downloads no model or dataset:

```bash
bash scripts/benchmark_hf_relu2max_4090.sh --output-dir runs/relu2max_4090

# Include the normalized Q/K and learned-scale workload used by the model.
bash scripts/benchmark_hf_relu2max_4090.sh \
  --qk-norm --scale 20 --learnable-scale \
  --output-dir runs/relu2max_4090_learned_scale

# Quick single-shape check; force FlashAttention instead of automatic SDPA selection.
python scripts/benchmark_hf_relu2max.py \
  --lengths 256 --warmup 3 --repeats 10 --sdpa-backend flash \
  --output-dir runs/relu2max_smoke
```

The wrapper uses BF16, batch 1, 8 heads, head dimension 64, key lengths
256/1024/2048/4096, and extra query lengths 1/16. Override `--batch-size`,
`--heads`, `--head-dim`, `--dtype`, and `--lengths` to match the original run;
for the A100 pretraining preset these are 32/12/64/BF16/1024.

`results.json` and `results.csv` are updated after each measured mode. They
include environment metadata, accuracy errors, forward/backward/combined
latency, and incremental peak allocated memory for eager softmax, SDPA, eager
ReLU2Max, elementwise Triton ReLU2Max, and fused Triton ReLU2Max. Compilation
and autotuning are warmed outside timing. Dense-path causal masks are
precomputed outside timing, favoring those baselines; mask application remains
timed. Forward measures inference with gradients disabled, while backward
reuses a resident forward graph. Combined forward/backward includes graph
creation and saved-state memory. Peak memory excludes CUDA context and
reserved allocator cache. By default, both `events` and `cuda_graph` timing
rows are collected. Event timing can include GPU idle gaps during Python
dispatch. Graph timing captures multiple complete operations and reports
batched replay time per call; compilation, capture, and replay checks are
untimed. `--graph-calls` (default 16) and `--graph-replays` (default 4) control
batching. In version 4, `--graph-calls` is a maximum bounded by an estimated
capture memory budget (default 512 MiB) and reduced on OOM retries. Graph
replay is checked against an identical ungraphed call with dtype-aware
tolerances, finite checks and aggregate error diagnostics.
Speedups compare only matching timing backends. Memory is always measured
with a separate ungraphed call before capture, excluding graph pools.
`--timing events` selects the original method. Capture failures are recorded
explicitly while preserving successful event results.
Graph sweeps isolate each shape/method in a fresh process by default. Inputs
are identically seeded per shape across methods, and process startup is
untimed. Compact bottom-right causal masks enable supported Flash decoding
without constructing a dense mask. See the linked version-4 note for details.
SDPA output and gradients now receive a separate chunked FP32 softmax
reference check. Graph replay checks also record actual elementwise and
aggregate errors against an identical ungraphed call.

Use `--sdpa-backend flash` to require that backend; `auto` may select another
backend, particularly for rectangular/padded shapes. With `--learnable-scale`,
SDPA scales Q explicitly to preserve the scalar gradient. That extra operation
is timed. Failed numerical checks are retained in the output and receive no
speedup ratio. `--skip-accuracy` marks results unverified.

CUDA tests compare outputs and Q/K/V/scale gradients with an independent
float64 dense oracle. They cover causal and noncausal attention, uneven tile
lengths, rectangular caches, arbitrary outer strides, padding/all-masked
rows, negative/zero learned scale, and saved-state size. CPU tests exercise
model dispatch, fallback, dropout behavior, and cache/divisor plumbing.

SDPA is a timing comparison, not a numerical oracle for ReLU2Max: its
softmax weights implement a different attention rule. Compare the fused
output and gradients against the ReLU2Max reference.

### Development validation status

The development environment has no CUDA GPU. The user subsequently supplied
a complete RTX 4090 benchmark log, analyzed in the linked next-steps note;
the newly added CUDA-graph timing path has not yet been GPU-executed here.
CPU tests and offline CUDA
compilation can catch integration and compiler failures, but cannot establish
GPU numerical correctness, race freedom, or throughput. CUDA test and
benchmark results are required before claiming a measured speedup or treating
the optimization as validated on the deployment GPU.

With PyTorch 2.14.0, Triton 3.8.0, and Transformers 4.44.2, the two pytest
modules passed **17 CPU tests**, with **60 CUDA cases skipped**. The separate
Triton-interpreter harness passed **16 FP32 cases** against float64 autograd,
including forward, all gradients, masks, tails, and split-cache reduction.
Interpreter checks do not exercise GPU tensor-core rounding. BF16 validation
is intentionally left to the CUDA suite because this Triton interpreter's
BF16 dot handling is unsuitable as a numerical oracle.

Offline compilation passed **132 variants** across SM80 (A100), SM89
(RTX 4090), and SM90 (H100), covering FP16/BF16/FP32 and head dimensions
1/32/40/64/128/256. This exposed excessive shared-memory use for some large
FP32 tiles on the 4090; smaller stage-one tiles were added. Larger candidates
can still be rejected by Triton's resource-aware autotuner on a specific GPU.
Compilation success is not an execution or performance result.

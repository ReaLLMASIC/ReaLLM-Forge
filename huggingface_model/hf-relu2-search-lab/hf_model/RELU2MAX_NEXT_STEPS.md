# RTX 4090 feedback and next experiments

This note analyzes the first ordinary-event run. See the subsequent
[complete graph run and benchmark repairs](RELU2MAX_4090_RESULTS.md) for
updated 4096-token results, failure analysis and version-4 commands.

## What the supplied run establishes

The supplied terminal log used an RTX 4090, PyTorch 2.11.0+cu128, BF16,
batch 1, 8 heads and head dimension 64. These are attention-only measurements,
with compilation/autotuning warmed and Python-dispatched CUDA-event timing.

| Square sequence length | Previous elementwise Triton F+B, ms | Fused Triton F+B, ms | Softmax SDPA F+B, ms | Fused speedup over previous |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.9776 | 0.6317 | 0.1599 | 1.55x |
| 1024 | 0.6566 | 0.5740 | 0.3058 | 1.14x |
| 2048 | 1.5524 | 0.6942 | 0.2038 | 2.24x |
| 4096 | 7.7056 | 0.6758 | 0.6595 | 11.40x |

At 4096, fused forward was 0.1741 ms versus SDPA's 0.2152 ms; backward
was 0.5171 versus 0.4444 ms. Combined latency was within 2.5% of SDPA.
Incremental peak allocated memory for combined forward/backward fell from
792 MiB to 16 MiB, a 49.5x reduction versus the previous Triton path; SDPA
reported 24.25 MiB. SDPA computes a different attention rule.

The result supports full fusion, but does not establish peak kernel speed.
Fused forward at 256 tokens took 0.1721 ms, almost identical to 4096, while
1024 and 2048 took longer. That pattern suggests host dispatch or measurement
variability contributes materially. Short-query decoding also trails SDPA.
The terminal log contains no printed accuracy failure; it is not a substitute
for the complete CUDA correctness suite or its machine-readable errors.

## Measure before choosing a new implementation

The benchmark now offers both ordinary event timing and batched CUDA-graph
replay. Graph capture contains multiple complete calls, amortizing host launch
cost. Memory is measured separately, without graph pools. Speedups only compare
rows with the same timing backend. A graph failure is recorded explicitly;
successful event results remain available. The new graph path has not been
executed on a GPU in the development environment.

From the self-contained `changed-files/` directory:

```bash
python scripts/benchmark_hf_relu2max.py --check-install
bash scripts/benchmark_hf_relu2max_4090.sh \
  --timing both --sdpa-backend flash \
  --output-dir runs/relu2max_4090_graphs

# Also test the model's QK-normalized, learned-scale workload.
bash scripts/benchmark_hf_relu2max_4090.sh \
  --timing both --qk-norm --scale 20 --learnable-scale \
  --output-dir runs/relu2max_4090_graphs_learned_scale
```

Use `--timing events` to reproduce the original timing method. If a forced
Flash SDPA shape is unsupported, it is reported as an error; `--sdpa-backend
auto` is available for a separate automatic-dispatch comparison. Graph results
are throughput for static inputs and shapes, not end-to-end Python latency or
proof of graph compatibility for the full model.

After this comparison, use a GPU profiler to inspect launch gaps, register
spills, occupancy, tensor-core utilization and memory stalls for the slowest
forward/backward shapes. Target the bottleneck that survives graph replay.
[PyTorch's CUDA-graph documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)
explains why replay removes Python and driver dispatch work.

## Prioritized kernel work

1. **Separate training/prefill from decode.** For one or a few query tokens,
   avoid doing padded matrix-tile work for mostly absent queries. Compare a
   vector/reduction kernel with tensor-core tiles, tune split-KV by actual
   cache length and batch/head parallelism, and include reduction launch cost.
2. **Tune backward independently.** Profile dQ and dK/dV rather than selecting
   all configurations from forward results. Test asynchronous loads and
   double buffering only where register/shared-memory occupancy permits it.
   Variable causal loop lengths make work distribution another candidate.
3. **Remove surrounding costs.** Avoid materializing repeated K/V for grouped
   heads, evaluate QK-normalization/RoPE fusion, and bucket/staticize cache
   lengths to reduce repeated JIT/autotuning. Preserve the model's existing
   nonuniform grouped-head mapping where applicable.
4. **Measure the real model.** Match training batch size, masks, QK norm,
   learned scale, and cache updates. Compare tokens/second and memory through
   the HF training/generation APIs before selecting the default backend.

## Framework options

These are proposed experiments, not implemented or benchmarked backends.

| Option | Role on the RTX 4090 | Hugging Face integration |
| --- | --- | --- |
| Triton / Gluon | Continue the existing kernel; Gluon gives explicit layouts and Ampere+ asynchronous copy control. Lowest migration cost, experimental API. | Existing custom model dispatch; retain explicit backward. |
| TileLang | Best independent Python DSL experiment: tiled GEMMs, software pipelines and shared/register memory control. | Wrap tensor inputs/outputs as a PyTorch operator and implement/register backward. |
| CuTe DSL / CUTLASS | More control over copies, MMA and thread layouts. Language supports SM80+; verify the architecture requirements of each borrowed example. | PyTorch/DLPack adapter, autograd and compiler metadata as needed. |
| FlashInfer custom attention | Focus on inference and KV-cache decoding. Custom variants can disable softmax and replace the weight transformation. | Forward/cache integration for generation; training needs a separate backward. |
| Meta GDPA | Directly relevant ReLU² training design. Current CuTe implementation is SM100-only; Triton path requires experimental Meta Triton/TLX APIs. | Can be adapted behind PyTorch/HF, but is not a stock-Triton drop-in for this 4090 environment. |

Sources: [Gluon](https://triton-lang.org/main/gluon/index.html),
[Gluon async copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html),
[TileLang](https://github.com/tile-ai/tilelang),
[CuTe DSL FAQ](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/faqs.html),
[FlashInfer custom variants](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html),
[FlashInfer paper with softmax-disabled example](https://arxiv.org/pdf/2501.01005),
[GDPA requirements and supported activations](https://github.com/facebookresearch/ads_model_kernel_library/blob/main/gdpa/README.md).

My priority is measured improvements to existing Triton plus one TileLang
comparison, then CuTe/Gluon if the profile justifies their additional control.
FlashInfer is a separate decode experiment. GDPA supplies directly relevant
design ideas, but its published Blackwell numbers do not predict 4090 speed.

## Keeping Hugging Face APIs

Kernel language does not inherently determine HF compatibility. Keep the
model/config classes, output contract, checkpoint format and cache semantics,
and replace the internal attention operator. The current custom NanoGPT
dispatch works with the existing Transformers 4.44.2 integration; it does not
need a newer attention registry simply to call a new kernel.

For newer models that support it, register an `AttentionInterface` function
and matching `AttentionMaskInterface` formatter. Without the mask registration,
HF may skip mask construction and pass `None`; account for causal, padding and
packed-sequence semantics explicitly. Avoid recreating a quadratic mask when
the kernel can accept compact metadata.
[HF attention interface](https://huggingface.co/docs/transformers/en/attention_interface).

For training, supply correct Q/K/V/learned-scale gradients. For generation,
preserve position offsets and KV-cache updates. For `torch.compile`, a
registered `torch.library.triton_op` with registered autograd is a structured
Triton integration route; external CUDA kernels need the corresponding custom
operator, fake/meta and autograd support. These are separate validation gates,
not guarantees from accepting a PyTorch tensor.
[PyTorch Triton integration](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html),
[custom CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

HF checkpoint compatibility alone does not enable the operator in vLLM or
SGLang: their Transformers backends substitute runtime attention. They need a
ReLU²-aware backend/cache adapter.
[vLLM integration](https://docs.vllm.ai/en/latest/models/supported_models/#writing-custom-models),
[SGLang integration](https://lmsysorg.mintlify.app/docs/supported-models/transformers_fallback).

## FlexAttention: possible, with extra work

Setting `score_mod` to ReLU² is insufficient: FlexAttention still applies
softmax. An exact mathematical reconstruction is possible. For positive
scores use log-weights `z = 2*log(score)`, and use `-inf` otherwise. Multiply
the returned normalized attention output by `exp(logsumexp(z)) / divisor`.
This recovers the unnormalized weighted sum.

Guard the logarithm's input before evaluating it. All-zero rows need explicit
handling; an experimental robust construction appends an always-allowed dummy
key with log-weight zero and value zero, then uses the resulting finite LSE.
This adds padding and normalization work.

The user's PyTorch 2.11.0 source provides `return_aux=AuxRequest(lse=True)`
and propagates gradients through returned LSE, so this is a viable experiment
in principle. Actual CUDA output and all gradients still require validation.
The derivation above is an engineering proposal, not a native ReLU² mode or a
measured performance result. Extra log/exp/normalization makes it a lower
priority peak-speed candidate than direct unnormalized fusion.
[PyTorch 2.11 public source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/attention/flex_attention.py),
[PyTorch 2.11 LSE backward](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_higher_order_ops/flex_attention.py).

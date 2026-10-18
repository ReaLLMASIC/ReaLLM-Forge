# RTX 4090 results and benchmark v2

## What the September 14 run establishes

Sources: the uploaded `results.json`, matching `results(1).csv`, and terminal
log `Pasted text(20260914-165032).txt`. The JSON and CSV contain the same 360
method/shape/operation/timing rows. Environment: RTX 4090, PyTorch 2.11.0+cu128,
Triton 3.6.0, BF16, B=1, H=8, D=64, QK-normalized inputs, learned scale 20,
divisor 256, causal attention, configurable threshold 6.5.

The CUDA test suite passed **44 tests** (including its CPU dispatch-rejection
check). All 60 benchmark workers completed, but 39 failed the accuracy gate
before timing. Therefore there are 126 successful measurement rows and 234
error rows. Those errors represent 39 failed accuracy checks, each repeated
across three operation modes and two timing methods; they are not 234 distinct
kernel failures. There were no graph-capture failures or OOMs among the recorded
measurements. All square 1024-, 2048- and 4096-token timing comparisons are missing.

Selected valid medians, **microseconds per call under CUDA graph replay**:

| Method | Q=K=256 forward | Q=K=256 forward+backward | Q=1, K=1024 forward | Q=1, K=1024 forward+backward |
|---|---:|---:|---:|---:|
| SDPA, automatic backend | unavailable | unavailable | 11.296 | 27.648 |
| Original fused ReLU² | 5.440 | 22.016 | 9.008 | 39.216 |
| Fixed8 | 5.216 | 21.120 | 9.216 | 39.168 |
| Configurable8 | 5.216 | 21.120 | 9.216 | 39.184 |
| Configurable6.5 | 5.216 | 20.880 | 9.248 | 42.080 |

At 256 tokens, fixed8 forward+backward used 4.07% less time than original
ReLU² in this run. It and configurable8 were equal at the reported precision.
Across 12 matched graph measurements covering four shapes and three operation
modes, configurable8/fixed8 had a geometric-mean latency ratio of 0.9964, while
configurable6.5/fixed8 was 1.0074. This is a limited, success-filtered set, not an
overall performance estimate. Configurable6.5 was 7.43% slower in the Q=1,
K=1024 forward+backward case; the v1 report did not save selected autotune tiles,
so it cannot identify the cause. V2 records those tiles for investigation.

For Q=1,K=1024, fixed8 forward was 1.226x as fast as SDPA; its forward+backward
was 41.7% slower. These are different comparisons. Ordinary generation uses
forward only, while backward through a KV cache is a separate workload.
SDPA was automatic here; the report does not establish that every SDPA call
used FlashAttention. All these attention functions differ mathematically.

The nonlinear variants used exactly the same measured incremental allocator
memory as original ReLU² in the matched cases, e.g. 1.0083 MiB at Q=K=256
forward+backward and 2.0029 MiB at Q=1,K=1024 forward+backward. This says nothing
about unmeasured long-sequence peaks. Event timings also include idle gaps from
Python dispatch; do not compare an event time with another method's graph time
or equate graph replay throughput with end-to-end HF training speed.

## Why the accuracy checker stopped the sweep

Every failed ReLU²-family worker failed on dQ, sometimes dK too. Their relative
RMS errors were approximately 0.23–0.24%, and output/dV/dScale checks passed.
The original kernel had the same failure pattern as the new kernels.

The v1 pointwise tolerance was `0.03 + 0.05*abs(reference)` for BF16. Dot-product
terms are rounded before they cancel, so a small final gradient entry can have
an absolute error that is large relative to that entry. A CPU arithmetic
emulation of the existing BF16 derivative-tile and output casts reproduced the
pattern at B=1,H=8,N=1024,D=64: 0.2346% RMS error, but 57 rejected entries under
the old check. This supports a tolerance diagnosis; it is not GPU execution or
proof that every failed GPU case is correct.

A separate bug affected one SDPA scale-gradient check: its absolute error was
0.010566, below `atol=0.03`, but the checker additionally required relative RMS
error <=5%. The scalar reference magnitude was only 0.103203, so the relative
error was 10.24%. That aggregate test discarded the absolute tolerance.

SDPA also rounds `Q*scale` to BF16 in the existing learned-scale adapter before
calling SDPA with host scale 1.0. The mathematical FP32 oracle does not make that
intermediate cast. PyTorch's [SDPA API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
accepts a host float scale and documents backend-dependent floating-point
results. Its [numerical accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
also explain why operation order and precision affect mathematically identical
computations. Neither document specifies or endorses this benchmark's tolerance.

## What v2 changes

All existing kernels, HF model/configuration files, and v1 benchmark files are
byte-for-byte preserved. Five added files supply the v2 runner, wrapper,
accuracy helper, regression tests and this document.

The independent chunked FP32 reference is unchanged. Default BF16/FP16
pointwise acceptance adds a rounding allowance based on the RMS of the same
feature vector:

```text
abs(error_i) <= atol + rtol*abs(reference_i) + 2*epsilon*RMS(reference_vector)
RMS(error)   <= atol + rtol*RMS(reference_tensor)
```

The extra pointwise term applies only to FP16/BF16 candidate tensors, not the
FP32 scalar scale gradient. It does not borrow scale from another row, head or
batch. Completely zero reference rows receive no extra allowance. Nonfinite
values fail. This is an explicit engineering screening policy, not a rigorous
forward-error bound or a substitute for high-precision functional tests.

`--accuracy-policy strict` disables the additional pointwise allowance while
retaining the aggregate absolute-tolerance fix. `--accuracy-rounding-factor 0`
also disables the allowance. JSON always records the legacy outcome, original
strict mismatch count, effective budgets, worst error/value/index, and relative
RMS error. The default relative/absolute tolerance values are unchanged.

Each operation gets its own gate: forward checks the output; backward checks
the gradients; forward+backward checks both. A rejected gradient cannot erase a
passing forward measurement. Rejected operations remain `accuracy_failed`, have
no timing or speedup, retain structured accuracy/activation diagnostics, and
cause a nonzero final exit. Graph replay validation and timing code are unchanged.
No old result is silently relabelled as passing, and no missing timing is filled
in from the uploaded files.

V2 additionally records source hashes and selected Triton autotune configurations.
It imports kernels without importing the Hugging Face package initializer, so
the benchmark needs PyTorch/Triton but not Transformers or optional torchao
extensions. The torchao loader warnings in the supplied log did not stop the
successful GPU tests/measurements; they were separate from the accuracy errors.
The model's optional Hugging Face workflow remains separate from this runner.

## Targeted rerun

Use a new output directory to preserve your existing results. From the complete
v2 folder, or after applying the additive patch to your current `with-relu2linear`
folder:

```bash
bash scripts/benchmark_hf_relu2linear_4090_v2.sh --check-install
bash scripts/benchmark_hf_relu2linear_4090_v2.sh \
  --lengths 1024 4096 --query-lengths \
  --modes forward forward_backward --warmup 3 --repeats 20 \
  --threshold 6.5 --output-dir runs/relu2linear_4090_v2_prefill
```

This targets the missing square prefill cases with ten workers. Then run the
full sweep if the checks pass:

```bash
bash scripts/benchmark_hf_relu2linear_4090_v2.sh \
  --threshold 6.5 --output-dir runs/relu2linear_4090_v2_full
```

To isolate the standard fixed-scale SDPA control and require its Flash backend:

```bash
bash scripts/benchmark_hf_relu2linear_4090_v2.sh \
  --lengths 1024 4096 --query-lengths \
  --no-learnable-scale --sdpa-backend flash \
  --output-dir runs/relu2linear_4090_v2_fixed_scale_flash
```

That removes dScale from every method and changes the workload consistently;
do not mix its timings with the learned-scale results. A forced unsupported
Flash backend fails explicitly. Share the new `results.json` and terminal log.

## Hardware interpretation

At Q=K=256, threshold8 affected **135/263168 = 0.0513%** of allowed scores;
threshold6.5 affected **1061/263168 = 0.4032%**. The largest score was 10.8723.
Thus this synthetic run barely exercises the tail, and it does not establish a
trained-model activation bit budget. The code still uses BF16 weight tiles and
FP32 accumulation. Real-data training and activation/quantization calibration
are the next steps after filling the missing timing comparisons.

## Validation of this benchmark-only update

The 13 new CPU regression tests reproduce the cancellation failure, exercise the
scalar tolerance fix, verify per-operation gating and diagnostics, and reject
NaNs, corrupt scalings and isolated outliers. Existing CUDA kernel tests are
unchanged; their 44-pass result above is the user's actual 4090 run. The v2
benchmark itself still needs the targeted GPU rerun.

The complete HF test suite uses the documented Transformers 4.44.2/tokenizers
0.19.1 versions. A separate check with the workspace's Transformers 4.57.6
encountered four generation tests failing on its newer cache interface in the
unchanged models. Use the documented HF version for those generation workflows;
this benchmark-only update does not change that model compatibility contract.

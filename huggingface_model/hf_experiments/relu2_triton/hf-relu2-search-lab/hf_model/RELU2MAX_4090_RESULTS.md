# Complete RTX 4090 run and benchmark repairs

## Supplied results

The September 8 run using PyTorch 2.11.0+cu128 and BF16 completed all twelve
shapes and saved its reports, but reported capture, validation and allocation
failures. Configuration: batch 1, eight heads, head dimension 64, causal
attention, no padding, fixed scale 1/sqrt(64). This is a new run, not merely
the missing end of the earlier partial log.

### Square attention: CUDA-graph forward

| Tokens | Fused ReLU², microseconds | Flash SDPA, microseconds | Reported fused speedup |
| --- | ---: | ---: | ---: |
| 256 | 5.0 | 9.1 | 1.81x |
| 1024 | 15.9 | 22.0 | 1.38x |
| 2048 | 57.8 | 63.9 | 1.11x |
| 4096 | 167.0 | 234.7 | 1.41x |

Speedups use the log's ratios, computed before printed latencies were rounded.
This run's forward measurements favor fused ReLU² at all four lengths. SDPA
implements softmax, so it is a performance comparison, not a numerical oracle.
Graph replay measures repeated static-input GPU execution; these are not
end-to-end Hugging Face training or generation speedups.

At 4096, ordinary-event forward+backward reported:

| Method | Time, ms | Incremental peak allocated memory, MiB |
| --- | ---: | ---: |
| Previous elementwise Triton | 7.6984 | 792.00 |
| Fused Triton | 0.6825 | 16.00 |
| Flash SDPA | 0.6637 | 24.25 |

Thus fused is 11.28x faster than the previous Triton path and about 2.8%
slower than SDPA with ordinary dispatch. Incremental memory is 49.5x smaller
than the previous path and about 34% smaller than SDPA. Memory was measured
outside CUDA capture. It excludes inputs/resident tensors, reserved allocator
memory, CUDA context, and other model components.

The 4096 fused combined graph measurement failed, as did the corresponding
old-Triton graph measurement. SDPA combined graph validation also failed.
There is no valid 4096 graph forward+backward comparison; summing separately
measured forward and backward cannot create one.

Decode forward for Q=1,K=4096 measured 9.6 microseconds fused versus 12.9
for previous Triton (1.34x). Its SDPA comparison is absent. Q=16,K=4096 has
no successful graph measurements. Short-query backward remains a weaker
case at K=1024/2048; this does not add work to inference, which runs forward.

## What failed

- 26 CUDA-graph OOM reports, starting with square-4096 eager ReLU² backward
  and continuing through later methods and decoding. By fused combined
  capture, the process reported about 19.35 GiB reserved but unallocated,
  1.10 GiB allocated in private pools, and only 55 MiB free. This indicates
  a need to bound captures and isolate their allocation lifetimes; it does
  not show that an ordinary fused call intrinsically needs 24 GiB.
- Six SDPA backward/combined replay checks failed at 1024/2048/4096. At
  4096, each comparison flagged 32 out of 2,097,152 elements. Differences
  were consistent in scale with BF16 rounding variation, but are not thereby
  proved harmless. The previous replay assertion used one strict tolerance
  for every dtype and did not retain enough aggregate error diagnostics.
- Twenty-four SDPA mode failures came from forced Flash rejecting the
  explicit masks used for all eight rectangular shapes.
- One initial eager-softmax isolated-backward capture failed following an
  autograd stream-mismatch warning.

Thirty of thirty-six fused graph rows succeeded. Its six failures were OOMs;
the log printed no fused FP32-reference accuracy failure. This is not the full
CUDA correctness suite or validation of other dtypes/model configurations.

PyTorch explains that graph pools retain addresses while graphs and captured
tensors remain alive. Reclaiming ordinary allocator cache alone does not
release live graph storage. The exact allocation cause still needs a GPU
memory trace; the repairs below avoid relying on its complete diagnosis.
[PyTorch 2.11 graph memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#graph-memory-management).

## Version 4 benchmark changes

- Default graph sweeps run each shape/method in a fresh Python process.
  Process startup and compilation remain outside timing, and a failed method
  cannot retain CUDA allocations in the next method. Parent reports retain
  successful rows, worker exit status and missing-result errors.
- Capture batching is limited by a memory estimate and a configurable target
  budget. `--graph-calls 16` is now a maximum, not a promise to capture sixteen
  operations. OOM retries use smaller batches and report their attempts.
- Graph timing uses fresh autograd leaves and constructs forward/backward on
  the capture stream. Cleanup releases tensors and graph references outside
  timing; failures retain diagnostic data rather than live exception frames.
- Replay checks use dtype-aware tolerances, finite checks and aggregate error
  metrics. `--graph-rtol` and `--graph-atol` override this check separately from
  the unchanged FP32 ReLU² reference check. Validation failures still exclude
  rows from speedup comparisons.
- SDPA now also has an independent chunked FP32 softmax output/gradient
  reference check, including learned scale. This checks its mathematics
  separately from graph-versus-ungraphed replay agreement; loosening replay
  tolerances alone is not the validation strategy. `--skip-accuracy` explicitly
  disables the independent checks and marks results unverified.
- Unpadded single-query decode uses no mask and `is_causal=False`, because all
  keys are in its past/current cache. Other rectangular causal shapes use
  symbolic lower-right causal bias, preserving the correct offset while
  allowing supported fused SDPA dispatch. Padding retains explicit masks.

[PyTorch 2.11 lower-right bias](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.bias.causal_lower_right.html)
and its [implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/attention/bias.py)
document this alignment and dispatch. Unsupported forced backends still fail
explicitly; the benchmark does not silently change attention semantics.

Inputs are seeded deterministically per Q/K shape so independently launched
methods receive identical data. This seed policy differs from the older
single-process sweep; compare methods within a new run.

## Rerun

Extract version 4 into a fresh directory, then:

```bash
cd changed-files
python scripts/benchmark_hf_relu2max.py --check-install

# First repeat 4096, including one-query and sixteen-query decoding.
bash scripts/benchmark_hf_relu2max_4090.sh \
  --lengths 4096 --timing both --sdpa-backend flash \
  --output-dir runs/relu2max_4090_v4

# Full sweep once the targeted run is clean.
bash scripts/benchmark_hf_relu2max_4090.sh \
  --timing both --sdpa-backend flash \
  --output-dir runs/relu2max_full_v4
```

Default graph capture target budget is 512 MiB. One operation can exceed this
target when necessary; it is a batching estimate, not a hard GPU-memory limit.
Actual `captured_calls` and `calls_per_sample` are recorded. Reducing capture
batching may expose more launch cost, especially for very small kernels.
The default BF16 replay relative tolerance is two machine epsilons (0.015625),
with absolute tolerance 1e-5; FP32 gradient tensors retain tighter defaults.
`--isolation none` is available for debugging; default isolation is preferable
for this sweep. Captures are serial, not simultaneous GPU workers.
Worker startup and repeated warmup add to total sweep wall time, although
neither contributes to the reported operation latencies.

The repaired graph execution path still needs a real GPU run. Development
validation is CPU integration/semantic tests plus CUDA tests that skip when
no GPU is present. Kernel code and HF model behavior are unchanged by these
benchmark repairs; the measured results above precede them.

# Release validation

**Current update: 92 CPU tests passed; 53 CUDA tests skipped on this CPU-only host.**

The release below describes the original integrated package. The current BF16 checker update adds 37 CPU regression cases and three CUDA FP32 cases. Its current results are in `validation/bf16_fix_pytest.xml` and `validation/bf16_fix_checks.json`; the GPU rerun remains required. All kernel/model/trainer source hashes are unchanged.

This release was checked in a **CPU-only** environment. No GPU throughput, CUDA
execution, or full 100M-token training result is claimed.

## Completed checks

- **55 CPU tests passed; 50 CUDA tests skipped**, using PyTorch 2.9.1+cpu,
  Transformers 4.44.2, tokenizers 0.19.1 and Triton 3.5.1. Machine-readable result:
  `validation/pytest.xml`; terminal summary: `validation/pytest.log`.
- **41 actual Triton-interpreter cases passed**, checking the fused forward and
  dQ/dK/dV/dscale/dbias against the independent PyTorch reference. Cases include
  all three exponents, signed/zero scales, signed per-head biases, masked and
  unmasked keys, cached queries, split-cache reduction, negative causal offsets,
  and noncausal branches. `validation/context_interpreter.json` records each case.
- **108 offline CUDA compilations passed** for SM80 (A100), SM89 (RTX 4090), and
  SM90 (H100): BF16 head size 128, every exponent and bias toggle, forward/query
  backward/key-value backward, masked prefill and unmasked decoding. These checks
  use a fixed 32×32 tile, not every autotuning candidate. Compiled SM89 shared
  memory fits its device limit. See `validation/context_compilation.json`.
- The four original attention kernels match their previous release hashes exactly.
  Frozen baseline outputs/initialization remain covered by regression tests.
- The previous **24-run** synthetic KDA/output-norm sweep passes, including exact
  resumed versus uninterrupted KDA optimizer/model state in both norm modes.
- New bias-plus-scaling CPU training resumes with **exactly matching final weights,
  Muon/AdamW state, and RNG state** in both output-norm conditions.
- Diagnostic reuse/recovery passes: a complete matching result is retained, a
  truncated result generates a new attempt, and the incomplete attempt is preserved.
- The monitor refreshes files with no completed comparison points and while its
  HTTP port is occupied. Empty graphs no longer crash the monitor.
- A fresh archive extraction installed successfully using `bash scripts/setup.sh
  --cpu` into an isolated `venv/`; `pip check` reported no broken requirements.
- The fresh-extraction **32-run** synthetic sweep completed all **64 training
  segments**, using all eight variants, two short contexts, both norm modes, and
  real Muon/AdamW optimizers. Its detailed diagnostic/export check is recorded in
  `validation/release_checks.json`.
- A separate coordinator check automatically probed a completed capped
  `relu2_bias_scale_one` checkpoint. The explicit CUDA gate correctly refused to
  treat this CPU host as a successful GPU test run.

## Bias coverage

`tests/test_context_attention.py` checks zero initialization, one parameter per
layer/head, signed closed-form bias derivatives with independent heads, FP64
finite-difference derivatives for all inputs, masking with negative biases, HF
save/load, checkpoint recomputation, auxiliary AdamW ownership without decay,
and an actual optimizer update. `tests/test_context_pipeline.py` checks trained
bias values in exports and exact checkpoint/optimizer resume.

`tests/test_context_cuda.py` checks actual fused BF16 outputs and all gradients,
bias-only backward at zero scale, fully masked rows, cached chunks, and fused
HF/Muon updates for all new variants in both norm modes. These CUDA tests are
included **but were not executed here**.

## Causal scaling coverage

The CPU suite verifies factors at 0/1/256/512/1024/2048 visible keys, unchanged
short rows, padding-prefix counts, future-key isolation, arbitrary cached chunks,
and combined bias/scaling gradients. Interpreter checks exercise the production
Triton math, including mask prefix loads and split-cache forward reduction.

## Remaining device validation

On the 4090, run `bash run.sh test-cuda`, or use `bash run.sh all`, which verifies
CUDA availability and executes the full test suite before data preparation and
training. A failed accuracy/model-update test stops the launch; a missing GPU
cannot be reported as a successful CUDA gate. The GPU suite evaluates both
pointwise and RMS BF16 error, not merely graph/eager agreement.

Offline compilation establishes compiler acceptance only. Synthetic CPU runs
establish workflow correctness only. They cannot establish faster training,
better long-context validation, numeric precision savings, or device stability.
Keep real results in a fresh run directory with their recorded data, environment,
recipe and source hashes. No prior full-model results are bundled or relabeled.

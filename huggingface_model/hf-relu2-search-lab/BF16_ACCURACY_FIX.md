# BF16 gradient check correction

The reported run passed 104/105 tests. Its single failure was one of 1,152 dQ
entries for B=1, H=3, Q=3, K=1057, D=128, alpha=0, with a learned head bias and
no padding. The absolute difference was 0.017312; the reference component was
about 0.0574. This happened in the correctness gate, before data preparation
or full training in `run.sh all`.

## Diagnosis and limits

The query-gradient kernel computes, for each key j:

```
p_j     = relu(g * dot(q, k_j) - tau_head)
delta_j = (dot(dO, v_j) * 2/C) * p_j * causal_length_factor
dQ      = g * sum_j(round_BF16(delta_j) * k_j)
```

The original reference kept `delta` in FP32. Casting `delta` before the dot
product is part of the fast tensor-core implementation. A sum over 1,057 keys
can contain large terms of opposite signs. A small final component can therefore
have a large relative difference even when the vector's overall error is small.
Rounding only the final FP32 reference to BF16 does not emulate that computation.

On CPU, the same fixture shape/seed with CPU-generated BF16 inputs reproduces
the old checker rejecting two dQ entries. Cast-only emulation has about **0.2321%
relative RMS error**. The RNG streams and exact tensors differ between CPU and
CUDA, so this is a reproduction of the failure mechanism, not a replay of the
user's exact GPU tensors. The updated CUDA check still needs a run on that GPU.

All model, optimizer, trainer and attention kernel files remain byte-identical
to the previous ZIP. There is no runtime precision or throughput tradeoff in
this update. Only correctness tests, diagnostic tools and release notes change.

## How the revised gate works

1. An FP64 reference defines the exact mathematical outputs and gradients for
   the supplied input values. Autograd independently checks the manual formulas.
   FP64 reference matmuls are unaffected by ambient TF32/reduced-reduction flags.
2. A separate prediction explicitly casts the attention weights and score
   derivatives at the same locations as the BF16 kernel.
3. Per-element intervals propagate FP32 arithmetic error, intermediate cast
   boundaries, final output casts, and split-cache reduction. They are computed
   from the inputs; observed kernel outputs never influence the limits.
4. Every fused output and gradient must lie inside its interval **and** satisfy
   the unchanged **1.5% relative RMS limit** against the mathematical reference.
   The old `atol=.015, rtol=.035` mismatch count remains in the JSON diagnostics.
5. Three additional CUDA tests compare the long-cache case in **FP32** against
   FP64 autograd at tight tolerances, eliminating the BF16 intermediate-cast effect.

`diagnostics/context_precision.py` contains the bound calculation. With FP32 unit
roundoff `u=2^-24`, a length-n sum uses `gamma_n=n*u/(1-n*u)` times the absolute
sum of terms. An uncertainty interval [x-e,x+e] is cast at both endpoints so
midpoint crossings are included. Multiplications propagate operand uncertainties
before their own rounding. The length-factor estimate covers FP32 division and
reciprocal/rsqrt rounding. These bounds assume the kernel's FP32 accumulation and
normal finite inputs. This diagnostic deliberately supports only small BF16/FP32
correctness fixtures; it is not a general verifier for overflow/subnormal workloads.

37 CPU regression cases verify the new gate, including independent FP64/autograd
agreement, FP32 arithmetic with explicit casts, the original false-failure pattern,
isolated corruptions too small to fail the aggregate check, wrong signs, omitted
bias, wrong scaling exponents, nonfinite results, and all-masked rows. A deliberately
wide pointwise interval still fails when the aggregate error exceeds 1.5%.

## Install the complete updated ZIP

Extract the updated `relu2-context-lab.zip` over the existing project folder. Its
archive contains source/tests/docs only; it does not overwrite `venv/`, `data/`,
`runs/`, checkpoints or Triton caches. `FILES_SHA256.json` is updated too, so
`run.sh all` can verify the replacement package. No dependency reinstall is needed.

From the existing `relu2-context-lab/` directory:

```bash
# First check the exact previously failing parameter combination.
venv/bin/python -m pytest -q \
  'tests/test_context_cuda.py::test_fused_forward_and_every_gradient[shape1-False-True-0.0]'

# Get per-output/per-gradient metrics and rounding bounds for that GPU fixture.
venv/bin/python -m diagnostics.check_context_numerics \
  --device cuda:0 --output runs/context_numerics.json

# Run the full correctness gate and then the original sweep.
bash run.sh all
```

The diagnostic writes JSON even when a numeric check fails. On a CUDA failure
it also saves the exact input and output/gradient tensors beside the JSON as
`context_numerics.tensors.pt`, allowing the failing values to be examined directly.
CPU-only illustration: add `--cpu-emulation`; this does not count as CUDA validation.
Do not bypass the failing test or remove the GPU gate. If training had already
started in a separate invocation, use its usual `--resume`; this test update does
not change the recorded model/trainer source identity.

Background: [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html)
describes non-associativity, precision of intermediate reductions, and why different
matmul execution paths can differ. The diagnosis above additionally follows the
specific casts in this package's kernel; generic documentation alone is not evidence
that an arbitrary kernel discrepancy is harmless.

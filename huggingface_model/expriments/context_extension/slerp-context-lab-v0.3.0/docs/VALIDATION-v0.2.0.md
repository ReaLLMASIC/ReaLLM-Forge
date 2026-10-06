# Release validation — v0.2.0

Validation date: 2026-09-27. All tests below ran on CPU with PyTorch 2.7.1 and
Transformers 4.57.6. These results establish software correctness for the tested
cases; they do not establish RTX 4090 memory usage or trained long-context quality.

## Completed checks

- **50 CPU tests passed**, zero failures. `validation/prior-v0.2.0/pytest.xml` contains the
  per-test results. `validation/prior-v0.2.0/release-summary.json` records the scope.
- Stable SLERP endpoints/norms and finite gradients near parallel and antipodal
  directions, including a float64 autograd check.
- Native forward parity for tiny Hunyuan, Llama/GQA, and GPT-NeoX. Tests use
  nonuniform Hunyuan Q/K gains and a head dimension that differs from hidden/head
  count, exposing normalization-order and head-shape mistakes.
- Future-token causality, arbitrary prefill splits, checkpointed/non-checkpointed
  gradient agreement, bounded KV-head caches, and single consumption of generated
  tokens for recurrent inference.
- Every tested backbone parameter changes after a full update. Full weights and
  optimizer survive save/load; interrupted training matches an uninterrupted run.
  Tied embeddings remain tied. Saved backbone config permits fresh model creation
  without downloading pretrained weights for reload.
- Gradients reach early input features from a late answer after eviction. A tiny
  full-weight model reduces loss on a small fixed evicted example. This is a
  learning-path test, not evidence of held-out recall or a SLERP advantage.
- Exact short-context KL has the expected zero at initialization, finite student
  gradients after perturbation, and no reference-model gradients.
- CPU BF16 computation retains FP32 parameters and finite gradients.
- Random target index coverage, multi-query order, token/label alignment, and
  separate strict-format versus target-independent code-value scoring.
- Disk guards, retention, explicit optimizer finalization, and latest-only archive.
- Python files parse; shell scripts pass `bash -n`; dependency `pip check` passes.
  CLI native evaluation, assessment, doctor and plot generation were exercised.

## Real pretrained Hunyuan integration

Model: `tencent/Hunyuan-0.5B-Instruct`

Revision: `2359fb220c010e9d6d62c62d466f0eda179c2cf3`

Backbone parameters: **539,010,048**.

A 512-token FP32 forward was compared against Hugging Face's native forward, using
256-token chunks in the custom wrapper and no eviction. Maximum absolute logit
error was **2.0981e-5**, mean absolute error **1.2275e-6**; it passed the declared
absolute/relative tolerance of 2e-4. All backbone parameters were trainable.
See `validation/prior-v0.2.0/hunyuan-pretrained-parity.json`.

Native CPU generation was also checked on two 4K examples with eight facts and
early evidence, using a deliberately small 32-token output budget:

- Single-value recall returned the correct code in an explanatory `<answer>` block:
  strict formatting failed, while the separate code-value score passed.
- Multi-value recall reached the generation budget before producing its answer.

The same multi-value example was rerun with 128 output tokens. It listed the three
correct values, then repeated them in an unfinished explanation and hit the limit.
It still failed both exact metrics: the value metric intentionally rejects extra
or repeated codes. Raw outputs, budgets and metadata are retained in
`validation/prior-v0.2.0/hunyuan-native-4k-smoke*` and `validation/prior-v0.2.0/hunyuan-multi-4k-budget128*`.
These tiny diagnostic samples demonstrate integration and an instruction-formatting
limitation, **not reliable long-context performance**. Do not pool the two output
budgets as one benchmark. The main screening workflow must still be run before
choosing the backbone or attributing an accuracy change to memory.

## Tokenizers and alternative backbone

The real Hunyuan and MiniCPM5 tokenizers load under the pinned stack. Chat formatting
with thinking disabled and exact constructed lengths 2K/4K with 1/8/32 facts were
checked. See `validation/prior-v0.2.0/pretrained-tokenizers.json`.

MiniCPM5 revision used for tokenizer/config checks:
`87179e5c1f455ef22e6223592d2d61351b525bfc`.

MiniCPM's standard Llama/GQA architecture is covered by tiny-model tests. Its real
pretrained weights were not downloaded or exercised during this release validation.

## Required checks on the target machine

- Run native parity and the 2K/4K full-training profiles on the actual RTX 4090.
- Establish native task competence at 4K–32K, then review the small held-out memory
  learning gate. Address output-limit/formatting failures separately from retrieval.
- Profile the exact selected optimizer, KL setting, lengths and gradient policy.
- Run the matched full-weight pilot and inspect all raw outputs and cell metrics.

No real pretrained backbone was finetuned here. CUDA BF16 kernels, the optional
bitsandbytes optimizer, actual GPU peak memory, PG-19 streaming, and full official
RULER/LongBench runs were not validated here. No claim of a SLERP quality gain,
128K/256K full-training fit, or extension beyond native context is made.

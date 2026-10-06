# Release validation — v0.3.0

Validation date: 2026-09-29. Tests used CPU PyTorch 2.7.1 and Transformers 4.57.6.
No CUDA device was available. These are software/integration checks, not evidence
of trained summarization quality, RTX 4090 fit, or edge performance.

## Completed

- **70 CPU tests passed**; see `validation/pytest-v0.3.0.xml`.
- The earlier 50 tests still pass: geometry, native parity, full backbone updates,
  checkpoint/optimizer round-trips, causal recurrence, exact resume, KL, data and
  storage guards. Prior release evidence is in `validation/prior-v0.2.0/`.
- New delta block recurrence matches a sequential implementation in float64,
  including gradients. RMT and delta preserve causality and match segmented versus
  incremental inference through boundaries. Checkpointed gradients match ordinary
  backpropagation and late-answer gradients reach old input features.
- Every tested RMT/delta backbone parameter changes after a full update; tied
  embeddings and all model weights survive offline restore. Summary training
  interrupted/resumed for SLERP, RMT and delta exactly matches continuous training.
- Whole-source/whole-target handling, duplicate rejection, immutable manifests,
  baseline source coverage/call budgets, native skips, all-pass accounting,
  paired-result matching and report generation are exercised. KL is zero for
  identical teacher/student on a fitting input, leaves teacher gradients empty,
  and skips a teacher input beyond native context.
- The two-update CPU profile includes optimizer state. This is a tiny-model shape
  smoke, with `headroom_ok=null`, not a CUDA memory result.
- Python parsing, shell syntax, CLI help and final dependency `pip check` pass.
  The validation environment initially retained a newer fsspec distribution;
  reinstalling the declared 2025.10.0 pin resolved the dependency check. Real HF
  streaming was then repeated successfully under that pinned version.

## Real Hugging Face integration

SmolLM2 revision: `a10cc1512eabd3dde888204e902eca88bddb4951`.
GovReport revision: `4e21184e01ae8017e2c036e180fe5e541fef60a0`.
See `validation/smol-govreport-cpu-smoke.json`.

A 512-token FP32 forward spanning two 256-token chunks matched the original HF
forward within declared tolerances (absolute 5e-5, relative 5e-4). Maximum absolute
logit difference was 2.77e-5; mean difference 2.00e-6. All backbone weights remained
trainable. The earlier 128-token single-chunk comparison had zero difference.

HF streaming prepared a whole validation report with **8,990 source tokens** and
984 reference tokens. Native context is 8,192. Local recurrent processing consumed
the complete source and generated eight tokens; native-full correctly skipped it.
Head-truncated generation also ran. Both outputs/timings and hashes are retained.
The output budget was deliberately too short for quality evaluation; the local
untrained output was poor. This demonstrates integration, not successful memory
learning or a meaningful quality ranking. These CPU timings are not 4090 timings.

Optional semantic scoring was exercised with real MiniLM on a 907-token reference,
longer than its 512-token encoder context. All 907 nonspecial tokens were processed
in blocks. Identical-text precision/recall/F1 were 1.0. See
`validation/semantic-smoke.json`. This verifies coverage and plumbing, not the
metric's correlation with human judgment or any summarizer quality.

## Still required on the target machine

1. Inspect native pretrained and shared-warmup summary quality on validation.
2. Run the actual 4090 two-update maximum-shape profiles, including the exact KL,
   optimizer and source/target settings; require at least 2 GiB GPU headroom.
3. Train the matched arms and evaluate full generated summaries on held-out data.
4. Inspect failure counts, output caps, paired quality/resource results and blind
   human judgments. Expand sample sizes and seeds before making general claims.
5. Only after choosing settings, evaluate untouched test documents above native
   context. Separately benchmark selected methods on Orin/edge hardware.

No pretrained model was finetuned in this packaging environment. CUDA/BF16 speed,
trained compression quality, GPU fit, optional 8-bit optimizers, arXiv preparation
and Jetson runtime/energy were not validated in this release. Alternative model
cards or large advertised context limits are not substitutes for these checks.

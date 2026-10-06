# Changelog

## 0.3.0 — 2026-09-29

- Add a Hugging Face whole-document summarization workflow, pinned SmolLM2-360M
  and GovReport, optional arXiv/Hunyuan studies, and explicit native/working limits.
- Add native/truncation/rolling/map-reduce controls and matched full-weight memory
  training from a shared native summarization warmup.
- Add causal recurrent soft tokens and an exact block delta-rule associative
  memory comparator alongside local/NLERP/SLERP.
- Add all-pass timing, GPU/RSS/state measurements, paired quality/efficiency
  reports, bootstrap intervals, blind review CSV and whole-summary semantic scoring.
- Add exact summary-training resume, optional full-source teacher KL with native
  fit checks, whole-target selection and immutable data/evaluation manifests.
- Add 20 tests (70 total), real SmolLM2/GovReport and long semantic-encoder smoke
  checks. Prior validation is retained under `validation/prior-v0.2.0/`.
- Preserve internal-drive guards and manual sparse external transfers. GPU fit,
  trained summary quality and edge performance still require target-machine runs.

## 0.2.0 — 2026-09-27

- Switch the primary backbone to Hunyuan-0.5B-Instruct; add MiniCPM5-1B/Llama support.
- Replace LoRA adaptation with complete backbone finetuning and full checkpoints.
- Preserve native Hunyuan post-RoPE normalization, NTK-alpha and head dimensions.
- Add unmodified HF baselines, pretrained forward parity, short learning gates,
  per-arm full-training profiling, and a separate optional exact retention-KL config.
- Keep local-history and memory gradients enabled by default; cap vocabulary loss
  allocations to supervised-position chunks.
- Randomize queried records and multi-query order; add fact-count and task controls.
- Add complete/incomplete evaluation markers and explicit optimizer finalization.
- Keep active files internal; provide rate-limited, manual external archiving.
- Preserve Pythia as a full-weight control and keep optional GRPO outside the main workflow.

This release changes the experiment and checkpoint format. v0.1 adapter checkpoints
and evaluation rows are not directly reusable for v0.2 matched comparisons.

## 0.1.0 — previous release

Pythia-410M with attention LoRA and spherical memory; adapter-only checkpoint format.
The earlier design proposal is retained as `docs/original-plan.md` for history and
is superseded by the v0.2 README, method and evaluation documents.

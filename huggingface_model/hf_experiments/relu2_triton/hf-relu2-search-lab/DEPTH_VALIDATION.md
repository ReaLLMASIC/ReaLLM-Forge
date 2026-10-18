# Depth sweep v1.1 disk-budget update — 2026-09-26 UTC

Current update: **41 CPU depth/storage tests passed; 3 CUDA probe tests skipped**.
These cover the 16-layer/three-depth defaults, disk-limited grid selection,
free-space guards, refusal to retire incomplete/corrupt/mismatched exports,
hardlink failure preserving optimizer state, idempotent retirement, and changed
export detection. A real two-run CPU Muon smoke (learned bias + causal length
scaling, both output norms) completed, retired optimizer states, loaded models
through the shared checkpoint paths, and resumed the sweep without retraining
or changing reported best losses. Logs are `validation/disk_update_pytest.*`.

The full original model/trainer/kernel suite was not rerun for this wrapper-only
update. Its source remains unchanged; evidence from v1 is preserved below. No
actual CUDA capacity probe or full-size training was executed here.

## Prior v1 validation (historical)

- Full suite: **115 passed, 56 CUDA skipped**, 2 pre-existing tokenizer warnings,
  202.70 seconds. This run included the first 23 depth-planning tests.
- Expanded depth suite: **27 passed, 3 CUDA skipped**. Four additional CPU
  tests cover cross-depth data identity, frozen common-capacity calibration,
  non-capacity worker failures, and exact decimal 70% rounding.
- In total: **119 distinct CPU tests passed** across these runs. The final
  package contains 56 CUDA-dependent tests; none was executed here.
- Original tests include exact Muon resume, learned subtractive bias, causal
  length scaling, output-norm and checkpoint behavior. The prior BF16 numerical
  accuracy fix is included intact.
- New CUDA tests exercise the real memory worker with SDPA, ReLU² with bias and
  length scaling, and linear16. Those are pending your GPU run, not certified.
- Capacity search tests use simulated workers with different per-condition
  limits. They establish orchestration logic, **not a measured 4090 capacity**.
- CLI planning requires no Torch or CUDA. Report generation was exercised,
  PNG output inspected, and file-only CSV/JSON/Markdown/HTML outputs checked.
- Training/kernel code is unchanged. Package comparison and archive manifest
  checks are recorded in `validation/depth_release_checks.json`.

Raw logs: `validation/depth_full_pytest.log` / `.xml`,
`validation/depth_new_pytest.log` / `.xml`. No GPU throughput or validation-loss
measurements are claimed by this release. Capacity probing and the full sweep
remain to be run on your workstation.

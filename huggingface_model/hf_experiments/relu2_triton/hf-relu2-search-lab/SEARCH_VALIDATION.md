# Validation for HF ReLU² search lab v1

Executed 2026-10-03 against the final frozen source.

- Integrated suite: **191 passed, 65 CUDA-dependent tests skipped**.
- Environment: Python 3.12.14, PyTorch 2.9.1+cpu, Transformers 4.44.2,
  tokenizers 0.19.1, datasets 3.6.0, pytest 8.3.5.
- No CUDA GPU is available here. CUDA correctness, capacity, latency and
  padding overhead must be checked on the 4090; no measured GPU improvement
  or new-model validation-quality result is claimed.

## What was exercised

The new model suite checks independent residual/head dimensions, QK/V widths
in both orders, masking including fully masked rows, all attention gradients,
the learned scale, exact equal-width parity with the old model, actual-width
KV caches, generation, checkpoint recomputation, optimizer routing and a fresh
offline process loading a Hugging Face Auto export.

The new search tests check matched tokens and initialization, bounded discrete
sampling, unsupported configurations, exact interrupted Muon training resume,
repeated absolute stop targets, and missing/failed measurements. The optional
HF Trainer bridge runs real optimizer steps and checks scheduler LR propagation
and optimizer-state restoration. Full Trainer checkpoint resume/distributed
execution is outside the tested scope.

The quick-depth suite checks the specified 512/2048 recipe, six default jobs,
parameter/disk estimates, preflight failure behavior, resume planning and
completed-only graph filtering. The integrated suite also exercises the
inherited context treatments, output norms, monitoring and checkpoint retirement.

The final standalone search CLI smoke completed all three arms at 256 tokens
using d_model=20, H=3, Dq=8 and Dv=6, with live reports and PNG/PDF plots. This is
synthetic **CPU functionality evidence**, not a meaningful model-quality result.

## Evidence files

- `validation/search_release_cpu.txt`: complete final test output.
- `validation/hf_search_focused_cpu.txt`: focused search checks.
- `validation/hf_search_cli_cpu.txt`: final CLI completion log.
- `validation/hf_search_cli_summary.json`: explicit CPU smoke status/backend labels.

All **113 original package files**, excluding the regenerated package
manifest, remain byte-for-byte identical to the recovered depth-lab v1.1 archive.
The new modules are additive. Use a fresh extraction/output to respect the
expanded source fingerprint. `SEARCH_RELEASE.json` records provenance and the
preserved file list. Files under older validation documents describe previous
releases; this file and its log describe the current integrated validation.

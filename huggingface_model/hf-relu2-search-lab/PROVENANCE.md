# Provenance

This integrated release was assembled from the saved KDA/output-norm source package
and the separate context-diagnostics package, then extended with the bias/scaling
experiments, fused kernels, tests, launch workflow, and reports.

Input archive SHA256:

- `relu2-50m-muon-output-norm.zip` (KDA version): `304b08ac877bab70b9e985482c70d2598c5d8f4ae6d0063d81ae412e6d21128e`
- `relu2-context-diagnosis.zip`: `76ae2ab87d479cb41c9f33c50f044d5707e317ca0007aeedfc57b28295518a35`

The four original attention kernel files remain byte-identical to the previous
release; `validation/original_kernel_hashes.json` records their hashes. Frozen
baseline numerical checks remain in `validation/baseline_golden.json` and the test
suite. This ZIP uses a new top-level directory and leaves prior archives unchanged.

New learned-bias/scaling kernels are separate in `hf_model/triton_context_attention.py`.
The original full-attention controls and capped-norm definitions are retained.
Model/trainer files gain new variant dispatch and diagnostic logging. No previous
user run has been resumed, modified, or reinterpreted as a new treatment.

FLA/einops wheels and their licenses/provenance are in `third_party/`. KDA is the
preserved pure recurrent arm in the dense GELU backbone, not the released Kimi hybrid/MoE.
Historical kernel Markdown files remain as background documentation; use the root
README for current setup and commands. Training data, checkpoints, dependency environments,
and earlier synthetic output graphs are not included.

`FILES_SHA256.json` covers all packaged files except itself. Run
`venv/bin/python scripts/check_install.py` after a fresh extraction to verify completeness.

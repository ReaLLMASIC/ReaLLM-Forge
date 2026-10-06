# Pinned KDA runtime

Kimi Linear's authors publish their KDA kernels in Flash Linear Attention (FLA).
This adapter follows the KDA architecture but keeps this project's dense backbone,
common initialization, Muon recipe and optional capped output norms.

- FLA core: **0.5.2**, MIT, [project](https://github.com/fla-org/flash-linear-attention), [release](https://pypi.org/project/fla-core/0.5.2/).
- einops: **0.8.1**, MIT, [project](https://github.com/arogozhnikov/einops), [release](https://pypi.org/project/einops/0.8.1/).
- The wheel files are unmodified PyPI release wheels and contain their license notices.
- `FLA_LICENSE` is the upstream MIT license. The KDA adapter is based on the published neural parameterization and operator APIs; kernels themselves are supplied unmodified by the installed wheel.
- API inspection: `fla/ops/kda/chunk.py`, `fused_recurrent.py`, `fla/modules/conv/causal_conv1d.py`, `fla/modules/fused_norm_gate.py` from **the bundled wheel**, not an unpinned main branch.
- For architecture comparison, upstream main was also inspected at commit `954438d1fcb5e1bb05c22f9908de9c5c2df74ae5` (2026-09-21). That commit is not the dependency pin; the exact release wheel hashes below are.
- No third-party full model/checkpoint is downloaded. The two small Python wheels are included so installation need not change the existing CUDA/HF packages.

```json
{
  "einops-0.8.1-py3-none-any.whl": "919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737",
  "fla_core-0.5.2-py3-none-any.whl": "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761"
}
```

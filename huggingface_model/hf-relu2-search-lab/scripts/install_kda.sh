#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RELU2_PYTHON="${RELU2_PYTHON:-${LAB_PYTHON:-$PWD/venv/bin/python}}"
if [[ ! -x "$RELU2_PYTHON" ]]; then
  echo 'Run bash scripts/setup.sh first.' >&2
  exit 1
fi
"$RELU2_PYTHON" - <<'PY'
import importlib.metadata as m
import torch
for name, required in [('transformers','4.44.2'),('tokenizers','0.19.1')]:
    if m.version(name) != required: raise SystemExit(f'Run bash scripts/setup.sh: {name}=={required} required')
if not hasattr(torch.optim,'Muon'): raise SystemExit('This PyTorch does not provide torch.optim.Muon')
try:
    m.version('flash-linear-attention')
except m.PackageNotFoundError:
    pass
else:
    raise SystemExit('Use an environment containing fla-core only, not the full flash-linear-attention model package. It can require a newer Transformers and conflict with this pinned experiment.')
print('Keeping PyTorch',torch.__version__,'and the installed HF packages.')
PY
# Bundled MIT-licensed Python wheels; do not upgrade PyTorch/Triton/Transformers.
"$RELU2_PYTHON" -m pip install --no-deps \
  third_party/wheels/einops-0.8.1-py3-none-any.whl \
  third_party/wheels/fla_core-0.5.2-py3-none-any.whl
"$RELU2_PYTHON" - <<'PY'
from hf_model.kda_attention import triton_ops
print('KDA backend imports:', ', '.join(op.__name__ for op in triton_ops()))
PY

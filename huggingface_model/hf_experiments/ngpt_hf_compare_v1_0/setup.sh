#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir 'torch==2.10.0' \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install --no-cache-dir -r requirements.txt
python -m pytest -q
python - <<'PY'
import torch, transformers, datasets
print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'datasets:', datasets.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), 'BF16:', torch.cuda.is_bf16_supported())
PY
printf '\nActivate with: source .venv/bin/activate\n'

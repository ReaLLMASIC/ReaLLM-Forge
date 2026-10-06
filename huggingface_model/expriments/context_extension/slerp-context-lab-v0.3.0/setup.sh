#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$SLERP_PACKAGE_ROOT"
python3 - <<'PY'
import sys, shutil
assert (3,10) <= sys.version_info[:2] < (3,13), 'Use Python 3.10, 3.11 or 3.12'
free = shutil.disk_usage('.').free/1e9
assert free >= 20, f'Need at least 20 GB free before installation; available {free:.1f}'
print(f'Free internal disk: {free:.1f} GB')
PY
if [[ "${1:-}" == "--cpu" ]]; then
  SLERP_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  nvidia-smi || { echo 'NVIDIA driver unavailable. Install a suitable driver first.' >&2; exit 1; }
  SLERP_TORCH_INDEX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
fi
python3 -m venv "$SLERP_WORK_ROOT/.venv"
"$SLERP_PYTHON" -m pip install --no-cache-dir 'pip==25.1.1' 'setuptools==80.9.0' wheel
"$SLERP_PYTHON" -m pip install --no-cache-dir 'torch==2.7.1' --index-url "$SLERP_TORCH_INDEX"
"$SLERP_PYTHON" -m pip install --no-cache-dir -r requirements.txt
"$SLERP_PYTHON" -m pip install --no-cache-dir --no-deps --no-build-isolation -e .
mkdir -p reports
"$SLERP_PYTHON" -m pip freeze > reports/installed-packages.txt
bash run.sh doctor
bash run.sh test
echo 'Setup complete. For summarization follow README.md; the older retrieval quickstart is scripts/quickstart_4090.sh.'

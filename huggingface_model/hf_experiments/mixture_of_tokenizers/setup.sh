#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode="cuda"
venv_dir=".venv"
python_bin="${PYTHON_BIN:-python3}"
while (($#)); do
  case "$1" in
    --cpu) mode="cpu"; shift ;;
    --cuda) mode="cuda"; shift ;;
    --venv) venv_dir="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash setup.sh [--cuda|--cpu] [--python python3.12] [--venv .venv]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

"$python_bin" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit('This dependency lock supports Python 3.10–3.12. Pass --python python3.12.')
PY
"$python_bin" -m venv "$venv_dir"
venv_python="$venv_dir/bin/python"
"$venv_python" -m pip install --upgrade pip
if [[ "$mode" == "cpu" ]]; then
  torch_index="https://download.pytorch.org/whl/cpu"
else
  torch_index="https://download.pytorch.org/whl/cu124"
fi
"$venv_python" -m pip install 'torch==2.5.1' --index-url "$torch_index"
"$venv_python" -m pip install -r requirements.txt
"$venv_python" - <<'PY'
import torch, transformers, tokenizers, datasets
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('tokenizers:', tokenizers.__version__)
print('datasets:', datasets.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
printf 'Activate with: source %q/bin/activate\n' "$venv_dir"


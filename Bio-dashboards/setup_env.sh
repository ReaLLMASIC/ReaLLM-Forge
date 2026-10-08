#!/usr/bin/env bash
# Creates an isolated Python environment for Bio-dash in .venv/
#
#   ./setup_env.sh                    all five dashboards
#   ./setup_env.sh polar_dashboard    just one dashboard's requirements
#   ./setup_env.sh --fresh            delete .venv and rebuild from scratch
#
# Then:  source .venv/bin/activate   (run_dashboard.sh scripts use it automatically once active)
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
if [[ "${1:-}" == "--fresh" ]]; then rm -rf "$VENV"; shift; fi

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "❌ Python 3.10+ required (polar-python). Found: $("$PY" --version)"; exit 1
fi
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo "❌ venv module missing. On Debian/Ubuntu/Jetson: sudo apt install python3-venv"; exit 1
fi

REQ=requirements.txt
if [[ -n "${1:-}" ]]; then
  REQ="$1/requirements.txt"
  [[ -f "$REQ" ]] || { echo "❌ no $REQ"; exit 1; }
fi

[[ -d "$VENV" ]] || { echo "🐍 Creating $VENV with $("$PY" --version)..."; "$PY" -m venv "$VENV"; }
"$VENV/bin/python" -m pip install -q --upgrade pip
echo "📦 Installing $REQ..."
"$VENV/bin/pip" install -q -r "$REQ"

echo "🔎 Checking imports..."
"$VENV/bin/python" - << 'PY'
import importlib.util
from importlib.metadata import version, PackageNotFoundError
for mod, dist in [("bleak", "bleak"), ("polar_python", "polar-python"),
                  ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("flask", "Flask")]:
    if importlib.util.find_spec(mod) is None:
        print(f"  ·  {dist:13s} (not in this requirements set)")
        continue
    try:
        print(f"  ✅ {dist:13s} {version(dist)}")
    except PackageNotFoundError:
        print(f"  ✅ {dist:13s}")
PY
echo
echo "Done. Activate with:  source $VENV/bin/activate"

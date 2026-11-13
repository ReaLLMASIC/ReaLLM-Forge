#!/usr/bin/env bash
# Creates an isolated Python environment for Bio-dash in .venv/
#
#   ./setup_env.sh            install / update
#   ./setup_env.sh --fresh    delete .venv and rebuild from scratch
#
# launch.sh and the service (./biodash.py) use .venv automatically.
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

[[ -d "$VENV" ]] || { echo "🐍 Creating $VENV with $("$PY" --version)..."; "$PY" -m venv "$VENV"; }
"$VENV/bin/python" -m pip install -q --upgrade pip
echo "📦 Installing $REQ..."
"$VENV/bin/pip" install -q -r "$REQ"

echo "🔎 Checking imports..."
"$VENV/bin/python" - << 'PY'
import importlib.util
from importlib.metadata import version, PackageNotFoundError
for mod, dist in [("bleak", "bleak"), ("polar_python", "polar-python"),
                  ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("websockets", "websockets")]:
    if importlib.util.find_spec(mod) is None:
        print(f"  ·  {dist:13s} (not in this requirements set)")
        continue
    try:
        print(f"  ✅ {dist:13s} {version(dist)}")
    except PackageNotFoundError:
        print(f"  ✅ {dist:13s}")
PY
echo
echo "Done. Start it with ./launch.sh (or ./biodash.py for a background service)."

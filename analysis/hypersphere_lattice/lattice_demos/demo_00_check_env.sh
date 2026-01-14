#!/usr/bin/env bash
# Demo 00 — Environment check
# ---------------------------
# What this does:
#   - Verifies that `python3` is available.
#   - Checks that this repo has `hypersphere_lattices.py` in the current directory.
#   - Prints Python version, NumPy version, and whether matplotlib and healpy are available.
#
# What to expect:
#   - A short report printed to the terminal. If matplotlib is missing, install with:
#       pip install matplotlib
#   - healpy is optional and only needed for HEALPix demos (S^2).
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.8+." >&2
  exit 1
fi

if [[ ! -f "./hypersphere_lattices.py" ]]; then
  echo "ERROR: ./hypersphere_lattices.py not found in current directory." >&2
  exit 1
fi

python3 - <<'PY'
import sys, json
print("Python:", sys.version.replace("\n"," "))
try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy: MISSING", e)

try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__)
except Exception as e:
    print("matplotlib: MISSING", e)

try:
    import healpy
    print("healpy:", healpy.__version__)
except Exception:
    print("healpy: (optional) not installed")
PY

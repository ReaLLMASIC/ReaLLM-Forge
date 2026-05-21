#!/usr/bin/env python3
"""Check whether the Python packages needed by the Roomba MuJoCo demos import successfully."""
from __future__ import annotations

import importlib
import sys

PACKAGES = [
    ("mujoco", "MuJoCo Python bindings"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("imageio", "imageio"),
]

ok = True
for module_name, label in PACKAGES:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "version unknown")
        print(f"[ok] {label}: {version}")
    except Exception as exc:  # noqa: BLE001 - keep this diagnostic broad.
        ok = False
        print(f"[missing] {label}: could not import {module_name}: {exc}")

print("\nTip: install with `pip install -r requirements.txt` from the package root.")
if not ok:
    sys.exit(1)

"""Check self-contained packaging without CUDA or importing Transformers."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "FILES_SHA256.json").read_text())
errors = [name for name, expected in manifest.items() if not (root / name).is_file()
          or hashlib.sha256((root / name).read_bytes()).hexdigest() != expected]
if errors:
    raise SystemExit("Missing/changed package files: " + ", ".join(errors))
print(f"All {len(manifest)} packaged files match; keep this separate from the prior attention sweep.")

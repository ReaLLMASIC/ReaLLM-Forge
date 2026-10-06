from pathlib import Path
import hashlib
import sys

root=Path(__file__).resolve().parents[1]
errors=[];count=0
for line in (root/"MANIFEST.sha256").read_text().splitlines():
    digest,name=line.split("  ",1)
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        errors.append(name)
    count+=1
if errors:
    raise SystemExit("Changed/missing packaged files: "+", ".join(errors))
print(f"Verified {count} packaged files")

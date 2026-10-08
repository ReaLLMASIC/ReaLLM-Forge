#!/usr/bin/env python3
"""Reconstruct simplified Hanzi from generated multicontext lane text.

Because the `char` lane carries the original character, decoding is a direct
bijection for simplified Hanzi. A timestep where every lane is `⧆` is rendered
as `<NON_HANZI>`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
NON_HANZI="⧆"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="data/simplified_hanzi_mc")
    ap.add_argument("--char_file", default=None, help="Optional generated char-lane text file; defaults to <root>/char/input.txt")
    args=ap.parse_args()
    root=Path(args.root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    char_path=Path(args.char_file) if args.char_file else root/"char"/"input.txt"
    chars=[line.strip() for line in char_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    decoded=["<NON_HANZI>" if ch==NON_HANZI else ch for ch in chars]
    print("".join(decoded))
    print(f"decoded_steps={len(decoded)} lanes={','.join(manifest['lanes'])}")
if __name__ == "__main__": main()

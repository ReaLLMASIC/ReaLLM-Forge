#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from hangul_factorizer import HangulFactorizedTokenizer


def read_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def main() -> None:
    p = argparse.ArgumentParser(description="Recompose sampled Korean multicontext lanes into rendered text.")
    p.add_argument("lanes_dir", help="Directory containing lane output files or lane subdirectories")
    p.add_argument("output", help="Rendered UTF-8 output text")
    p.add_argument("--filename", default="input.txt", help="File name to read from each lane subdirectory")
    p.add_argument("--char-file", default=None, help="Optional original/non-Hangul passthrough character stream")
    p.add_argument("--use-pos", "--pos", action="store_true", help="Include part-of-speech (POS) tag lane")
    args = p.parse_args()

    root = Path(args.lanes_dir)
    tok = HangulFactorizedTokenizer(use_pos=args.use_pos)
    streams = []
    for name in tok.lane_names:
        direct = root / f"{name}.txt"
        nested = root / name / args.filename
        streams.append(read_stream(direct if direct.exists() else nested))
    if not streams or not streams[0]:
        raise SystemExit(f"No lane streams found under {root}")
    char_stream = read_stream(Path(args.char_file)) if args.char_file else read_stream(root / "char" / args.filename)
    n = max(len(s) for s in streams)
    chars = []
    for pos in range(n):
        ids = [tok.id_from_token_or_label(i, streams[i][pos] if pos < len(streams[i]) else "") for i in range(len(tok.lanes))]
        rendered = tok.decode_indices(ids)
        if rendered:
            chars.append(rendered)
        elif pos < len(char_stream):
            chars.append(char_stream[pos])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(chars), encoding="utf-8")

if __name__ == "__main__":
    main()

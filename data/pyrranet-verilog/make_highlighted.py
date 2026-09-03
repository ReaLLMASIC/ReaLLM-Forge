#!/usr/bin/env python3
"""
make_highlighted.py

Batch convert:
  orig/orig_<index>.v  ->  highlighted/ts_<index>.v

Uses verilog_ts_colorize.py to produce a per-byte highlight mask (same length as input).
Verifies lengths match; if not, deletes output and continues.

Reports success/failure totals and success percentage at the end.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


ORIG_RE = re.compile(r"^orig_(\d{7})\.v$")


def run_colorizer(
    colorizer_py: Path,
    src_file: Path,
    highlights_scm: Path,
    tmp_out: Path,
    prefer_longest: bool,
) -> Tuple[int, str]:
    """
    Runs verilog_ts_colorize.py on src_file with an explicit -o tmp_out.
    Returns (returncode, combined_output).
    """
    cmd: List[str] = [
        sys.executable,
        str(colorizer_py),
        str(src_file),
        "--highlights",
        str(highlights_scm),
        "-o",
        str(tmp_out),
    ]
    if prefer_longest:
        cmd.append("--prefer-longest")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert orig_<index>.v -> highlighted/ts_<index>.v with length checks.")
    ap.add_argument("--orig-dir", type=Path, default=Path("orig"), help="Directory containing orig_*.v files.")
    ap.add_argument("--out-dir", type=Path, default=Path("highlighted"), help="Output directory for ts_*.v files.")
    ap.add_argument("--highlights", type=Path, default=Path("highlights.scm"), help="Path to highlights.scm")
    ap.add_argument(
        "--colorizer",
        type=Path,
        default=Path("verilog_ts_colorize.py"),
        help="Path to verilog_ts_colorize.py",
    )
    ap.add_argument("--prefer-longest", action="store_true", help="Pass --prefer-longest to colorizer.")
    ap.add_argument("--max-files", type=int, default=None, help="Optional limit to process first N files.")
    ap.add_argument("--verbose", action="store_true", help="Print per-file results.")
    args = ap.parse_args()

    orig_dir: Path = args.orig_dir
    out_dir: Path = args.out_dir
    highlights: Path = args.highlights
    colorizer: Path = args.colorizer

    if not orig_dir.is_dir():
        raise SystemExit(f"ERROR: orig-dir does not exist or is not a directory: {orig_dir}")
    if not highlights.is_file():
        raise SystemExit(f"ERROR: highlights.scm not found: {highlights}")
    if not colorizer.is_file():
        raise SystemExit(f"ERROR: colorizer script not found: {colorizer}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort inputs by index
    inputs: List[Tuple[int, Path]] = []
    for p in orig_dir.iterdir():
        if not p.is_file():
            continue
        m = ORIG_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        inputs.append((idx, p))
    inputs.sort(key=lambda t: t[0])

    if args.max_files is not None:
        inputs = inputs[: args.max_files]

    total = len(inputs)
    success = 0
    failure = 0

    if total == 0:
        print(f"No files found matching orig_XXXXXXX.v in: {orig_dir}")
        return

    for idx, src in inputs:
        dst = out_dir / f"ts_{idx:07d}.v"
        tmp = out_dir / f".tmp_ts_{idx:07d}.v"

        # Ensure clean temp from prior runs
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

        rc, out = run_colorizer(
            colorizer_py=colorizer,
            src_file=src,
            highlights_scm=highlights,
            tmp_out=tmp,
            prefer_longest=args.prefer_longest,
        )

        if rc != 0:
            failure += 1
            # cleanup temp
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if args.verbose:
                print(f"[FAIL] {src.name}: colorizer exited {rc}")
                print(out.rstrip())
            continue

        # Validate length
        try:
            src_len = src.stat().st_size
            tmp_len = tmp.stat().st_size if tmp.exists() else -1
        except OSError as e:
            failure += 1
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if args.verbose:
                print(f"[FAIL] {src.name}: stat error {e}")
            continue

        if tmp_len != src_len:
            failure += 1
            # Remove bad output and continue
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if args.verbose:
                print(f"[FAIL] {src.name}: length mismatch src={src_len} out={tmp_len} (deleted)")
            continue

        # Move temp to final destination (atomic replace)
        try:
            os.replace(tmp, dst)
        except OSError as e:
            failure += 1
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if args.verbose:
                print(f"[FAIL] {src.name}: could not write {dst.name}: {e}")
            continue

        success += 1
        if args.verbose:
            print(f"[OK]   {src.name} -> {dst.name} ({src_len} bytes)")

    pct = (100.0 * success / total) if total else 0.0
    print()
    print("=== Highlight conversion summary ===")
    print(f"Input files:  {total}")
    print(f"Successes:    {success}")
    print(f"Failures:     {failure}")
    print(f"Success rate: {pct:.2f}%")
    print(f"Output dir:   {out_dir.resolve()}")


if __name__ == "__main__":
    main()


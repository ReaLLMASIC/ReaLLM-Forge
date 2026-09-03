#!/usr/bin/env python3
"""Normalize a UTF-8 document to Simplified or Traditional Chinese."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

try:
    from hanzi_factor.normalization import normalize_chinese
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hanzi_factor.normalization import normalize_chinese


def _read(path: str, encoding: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding=encoding)


def _write(path: str, text: str, encoding: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding=encoding, newline="", dir=destination.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="input file or -")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-o", "--output", default="-", help="output file or -")
    output.add_argument("--in-place", action="store_true", help="replace INPUT atomically")
    parser.add_argument("--to", choices=("simplified", "traditional"), required=True)
    parser.add_argument(
        "--variant",
        choices=("generic", "taiwan", "taiwan-phrases", "hong-kong"),
        default="generic",
        help="regional vocabulary/orthography profile (default: generic)",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.in_place and args.input == "-":
        parser.error("--in-place requires a file INPUT")
    source = _read(args.input, args.encoding)
    normalized = normalize_chinese(source, args.to, variant=args.variant)
    destination = args.input if args.in_place else args.output
    _write(destination, normalized, args.encoding)
    if not args.quiet:
        print(
            "normalize-chinese: "
            f"profile={args.to}/{args.variant} "
            f"input_chars={len(source)} output_chars={len(normalized)} "
            f"changed={source != normalized}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"normalize-chinese: {error}", file=sys.stderr)
        raise SystemExit(2)

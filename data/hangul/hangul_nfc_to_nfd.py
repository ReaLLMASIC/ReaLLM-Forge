#!/usr/bin/env python3
"""
Convert only precomposed Korean Hangul syllables from NFC to NFD.

Everything outside the Hangul syllables block U+AC00..U+D7A3 is left unchanged.
So Latin accents, emoji, punctuation, spaces, line endings, etc. are not normalized.

Usage:
  python3 hangul_nfc_to_nfd.py input.txt output.txt
  python3 hangul_nfc_to_nfd.py --in-place input.txt
  cat input.txt | python3 hangul_nfc_to_nfd.py > output.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unicodedata
from pathlib import Path


HANGUL_SYLLABLES_START = 0xAC00
HANGUL_SYLLABLES_END = 0xD7A3


def is_precomposed_hangul_syllable(ch: str) -> bool:
    codepoint = ord(ch)
    return HANGUL_SYLLABLES_START <= codepoint <= HANGUL_SYLLABLES_END


def hangul_only_nfd(text: str) -> str:
    """
    Apply Unicode NFD only to precomposed Hangul syllables.

    Example:
      한 U+D55C -> ᄒ U+1112 + ᅡ U+1161 + ᆫ U+11AB

    Non-Hangul characters are returned exactly as they are.
    """
    return "".join(
        unicodedata.normalize("NFD", ch)
        if is_precomposed_hangul_syllable(ch)
        else ch
        for ch in text
    )


def convert_bytes(data: bytes, encoding: str) -> bytes:
    # surrogateescape preserves invalid bytes when decoding/re-encoding.
    text = data.decode(encoding, errors="surrogateescape")
    converted = hangul_only_nfd(text)
    return converted.encode(encoding, errors="surrogateescape")


def convert_file(input_path: Path, output_path: Path, encoding: str) -> None:
    data = input_path.read_bytes()
    output_path.write_bytes(convert_bytes(data, encoding))


def convert_file_in_place(path: Path, encoding: str) -> None:
    converted = convert_bytes(path.read_bytes(), encoding)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(converted)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert only Korean Hangul syllables from NFC to NFD."
    )
    parser.add_argument("input", nargs="?", help="Input file. If omitted, reads stdin.")
    parser.add_argument("output", nargs="?", help="Output file. If omitted, writes stdout.")
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Rewrite the input file in place.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding to use. Default: utf-8.",
    )

    args = parser.parse_args()

    if args.in_place:
        if not args.input or args.output:
            parser.error("--in-place requires exactly one input file and no output file.")
        convert_file_in_place(Path(args.input), args.encoding)
        return 0

    if args.input:
        input_path = Path(args.input)
        if args.output:
            convert_file(input_path, Path(args.output), args.encoding)
        else:
            sys.stdout.buffer.write(convert_bytes(input_path.read_bytes(), args.encoding))
        return 0

    if args.output:
        parser.error("An output file requires an input file.")

    sys.stdout.buffer.write(convert_bytes(sys.stdin.buffer.read(), args.encoding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

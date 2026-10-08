#!/usr/bin/env python3
"""Restore a mixed prefix-IDS UTF-8 document to ordinary text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

try:
    from hanzi_factor.datasets import (
        load_ccd_json,
        load_ids_file,
        load_makemeahanzi,
        merge_datasets,
    )
    from hanzi_factor.text import restore_text
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hanzi_factor.datasets import (
        load_ccd_json,
        load_ids_file,
        load_makemeahanzi,
        merge_datasets,
    )
    from hanzi_factor.text import restore_text


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
    sources = parser.add_argument_group("same decomposition catalogue used forward")
    sources.add_argument("--ids", action="append", default=[], metavar="FILE")
    sources.add_argument("--ccd", action="append", default=[], metavar="FILE")
    sources.add_argument("--makemeahanzi", action="append", default=[], metavar="FILE")
    sources.add_argument(
        "--merge-policy", choices=("first", "last", "shortest"), default="first"
    )
    parser.add_argument(
        "--on-unknown",
        choices=("error", "keep"),
        default="error",
        help="policy for structurally valid IDS absent from the catalogue",
    )
    parser.add_argument(
        "--on-ambiguous",
        choices=("error", "first"),
        default="error",
        help="policy for structural collisions (default: error)",
    )
    parser.add_argument(
        "--keep-uplus-escapes",
        action="store_true",
        help="do not turn forward <U+XXXX> fallbacks back into scalars",
    )
    parser.add_argument(
        "--no-wrapped",
        action="store_true",
        help="disable recognition of forward ⟦IDS⟧ wrappers",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--report", metavar="JSON", help="write restoration statistics")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.in_place and args.input == "-":
        parser.error("--in-place requires a file INPUT")

    datasets = []
    datasets.extend(load_ids_file(path, merge_policy=args.merge_policy) for path in args.ids)
    datasets.extend(load_ccd_json(path, merge_policy=args.merge_policy) for path in args.ccd)
    datasets.extend(
        load_makemeahanzi(path, merge_policy=args.merge_policy)
        for path in args.makemeahanzi
    )
    if not datasets:
        parser.error("provide --ids, --ccd, or --makemeahanzi")
    dataset = merge_datasets(
        *datasets,
        merge_policy=args.merge_policy,
        name=" + ".join(item.name for item in datasets),
    )

    source_text = _read(args.input, args.encoding)
    result = restore_text(
        source_text,
        dataset.mapping,
        on_unknown=args.on_unknown,
        on_ambiguous=args.on_ambiguous,
        decode_uplus_escapes=not args.keep_uplus_escapes,
        accept_wrapped=not args.no_wrapped,
    )
    destination = args.input if args.in_place else args.output
    _write(destination, result.text, args.encoding)

    report = {
        "input": args.input,
        "output": destination,
        "catalogue_records": len(dataset),
        "source_issues": len(dataset.issues),
        "input_characters": result.input_characters,
        "output_characters": result.output_characters,
        "decoded_ids_roots": result.decoded_ids_roots,
        "decoded_uplus_escapes": result.decoded_uplus_escapes,
        "passed_through_characters": result.passed_through_characters,
        "unknown_ids_roots": result.unknown_ids_roots,
        "ambiguous_ids_roots": result.ambiguous_ids_roots,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(
            "ids-to-text: "
            f"decoded_ids={result.decoded_ids_roots} "
            f"decoded_escapes={result.decoded_uplus_escapes} "
            f"unknown={result.unknown_ids_roots} "
            f"ambiguous={result.ambiguous_ids_roots}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ids-to-text: {error}", file=sys.stderr)
        raise SystemExit(2)

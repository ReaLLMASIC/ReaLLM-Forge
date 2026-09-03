#!/usr/bin/env python3
"""Replace all covered Han characters in a UTF-8 document with prefix IDS."""

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
    from hanzi_factor.text import factorize_text
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hanzi_factor.datasets import (
        load_ccd_json,
        load_ids_file,
        load_makemeahanzi,
        merge_datasets,
    )
    from hanzi_factor.text import factorize_text


def _read(path: str, encoding: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding=encoding)


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
    sources = parser.add_argument_group("decomposition catalogue")
    sources.add_argument("--ids", action="append", default=[], metavar="FILE")
    sources.add_argument("--ccd", action="append", default=[], metavar="FILE")
    sources.add_argument("--makemeahanzi", action="append", default=[], metavar="FILE")
    sources.add_argument(
        "--merge-policy", choices=("first", "last", "shortest"), default="first"
    )
    parser.add_argument(
        "--format",
        choices=("expanded", "direct"),
        default="expanded",
        help="recursive factorization or canonical source IDS (default: expanded)",
    )
    parser.add_argument(
        "--on-uncovered",
        choices=("keep", "error", "escape"),
        default="keep",
        help="policy for Han characters absent from the catalogue",
    )
    parser.add_argument("--wrap", action="store_true", help="wrap each IDS as ⟦IDS⟧")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--report", metavar="JSON", help="write conversion statistics")
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
    result = factorize_text(
        source_text,
        dataset.mapping,
        expanded=args.format == "expanded",
        uncovered=args.on_uncovered,
        wrap=args.wrap,
    )
    destination = args.input if args.in_place else args.output
    _write(destination, result.text, args.encoding)

    report = {
        "input": args.input,
        "output": destination,
        "format": args.format,
        "catalogue_records": len(dataset),
        "source_issues": len(dataset.issues),
        "input_characters": result.input_characters,
        "matched_characters": result.matched_characters,
        "changed_characters": result.changed_characters,
        "uncovered_han_characters": result.uncovered_han_characters,
        "distinct_uncovered": [
            {"character": char, "codepoint": f"U+{ord(char):04X}"}
            for char in result.distinct_uncovered
        ],
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
            "text-to-ids: "
            f"matched={result.matched_characters} "
            f"changed={result.changed_characters} "
            f"uncovered_han={result.uncovered_han_characters}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"text-to-ids: {error}", file=sys.stderr)
        raise SystemExit(2)

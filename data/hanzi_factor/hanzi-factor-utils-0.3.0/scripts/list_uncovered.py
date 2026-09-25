#!/usr/bin/env python3
"""List characters that fail a selected Hanzi Factor coverage level.

The input is a JSON report produced by ``scripts/coverage_utf.py --json``.
By default a character is covered only when its tree is recursively closed,
binary-roundtrippable, and unique in the report's reverse index.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Literal


Criterion = Literal["direct", "recursive", "unique", "binary", "bijective"]


def is_covered(record: dict[str, Any], criterion: Criterion) -> bool:
    """Return whether one report record reaches ``criterion``."""

    direct = record.get("has_exact_ids") is True
    recursive = record.get("recursively_expandable") is True
    unique = recursive and not record.get("reverse_collision_with")
    binary = record.get("binary_roundtrip") is True
    if criterion == "direct":
        return direct
    if criterion == "recursive":
        return recursive
    if criterion == "unique":
        return unique
    if criterion == "binary":
        return binary
    return unique and binary


def uncovered_reason(record: dict[str, Any], criterion: Criterion) -> str:
    """Explain the first failed requirement in coverage order."""

    if record.get("has_exact_ids") is not True:
        return str(record.get("status") or "no accepted root IDS")
    if criterion == "direct":
        return "no accepted root IDS"
    if record.get("recursively_expandable") is not True:
        unresolved = record.get("unresolved_leaves") or []
        cycle = record.get("cycle") or []
        if unresolved:
            return "unresolved components: " + ",".join(map(str, unresolved))
        if cycle:
            return "cycle: " + " -> ".join(map(str, cycle))
        return str(record.get("error") or "not recursively expandable")
    if criterion in {"unique", "bijective"}:
        collisions = record.get("reverse_collision_with") or []
        if collisions:
            return "structural collision: " + ",".join(map(str, collisions))
    if criterion in {"binary", "bijective"} and record.get("binary_roundtrip") is not True:
        return str(record.get("binary_error") or "binary roundtrip not run/passed")
    return f"does not satisfy {criterion} coverage"


def load_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and minimally validate one coverage report."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error.msg}") from error
    if not isinstance(report, dict) or not isinstance(report.get("records"), list):
        raise ValueError("input is not a Hanzi Factor coverage report")
    records = report["records"]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("coverage report contains a non-object record")
    return report, records


def select_uncovered(
    records: list[dict[str, Any]], criterion: Criterion
) -> list[dict[str, Any]]:
    """Return report records that fail ``criterion`` in original order."""

    return [record for record in records if not is_covered(record, criterion)]


def render_tsv(
    records: list[dict[str, Any]], criterion: Criterion, *, header: bool
) -> str:
    stream = io.StringIO(newline="")
    fields = ("codepoint", "character", "status", "reason", "ids")
    writer = csv.DictWriter(stream, fieldnames=fields, dialect="excel-tab")
    if header:
        writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "codepoint": record.get("codepoint", ""),
                "character": record.get("character", ""),
                "status": record.get("status", ""),
                "reason": uncovered_reason(record, criterion),
                "ids": record.get("ids") or "",
            }
        )
    return stream.getvalue()


def render_json(
    report: dict[str, Any], records: list[dict[str, Any]], criterion: Criterion
) -> str:
    selected = [
        {
            "codepoint": record.get("codepoint"),
            "character": record.get("character"),
            "status": record.get("status"),
            "reason": uncovered_reason(record, criterion),
            "ids": record.get("ids"),
            "unresolved_leaves": record.get("unresolved_leaves") or [],
            "reverse_collision_with": record.get("reverse_collision_with") or [],
        }
        for record in records
    ]
    output = {
        "schema_version": 1,
        "source_dataset": report.get("dataset"),
        "criterion": criterion,
        "total_characters": len(report.get("records", [])),
        "uncovered_characters": len(selected),
        "records": selected,
    }
    return json.dumps(output, ensure_ascii=False, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage report JSON")
    parser.add_argument(
        "--criterion",
        choices=("direct", "recursive", "unique", "binary", "bijective"),
        default="bijective",
        help=(
            "required coverage level (default: bijective = recursive, unique, "
            "and binary-roundtrippable)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "characters", "json"),
        default="tsv",
        help="output representation (default: tsv)",
    )
    parser.add_argument("--output", type=Path, help="write to FILE instead of stdout")
    parser.add_argument(
        "--no-header", action="store_true", help="omit the TSV header row"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, records = load_report(args.report)
        uncovered = select_uncovered(records, args.criterion)
        if args.format == "json":
            rendered = render_json(report, uncovered, args.criterion)
        elif args.format == "characters":
            rendered = "".join(f"{record.get('character', '')}\n" for record in uncovered)
        else:
            rendered = render_tsv(
                uncovered, args.criterion, header=not args.no_header
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8", newline="")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError) as error:
        print(f"list_uncovered: {error}", file=sys.stderr)
        return 2

    print(
        f"{len(uncovered):,} uncovered of {len(records):,} "
        f"for criterion={args.criterion}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit Han decomposition coverage over Unicode ranges or target lists."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from hanzi_factor.coverage import audit_coverage
    from hanzi_factor.datasets import (
        available_presets,
        characters_from_ranges,
        load_ccd_json,
        load_ids_file,
        load_makemeahanzi,
        load_target_file,
        merge_datasets,
        parse_range_spec,
        targets_from_presets,
    )
except ModuleNotFoundError:  # Permit running from an uninstalled source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hanzi_factor.coverage import audit_coverage
    from hanzi_factor.datasets import (
        available_presets,
        characters_from_ranges,
        load_ccd_json,
        load_ids_file,
        load_makemeahanzi,
        load_target_file,
        merge_datasets,
        parse_range_spec,
        targets_from_presets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit direct IDS, recursive expansion, fallbacks, missing/malformed/"
            "cyclic entries, and reverse collisions over Unicode Han targets."
        )
    )
    sources = parser.add_argument_group("local decomposition sources")
    sources.add_argument(
        "--ids",
        action="append",
        default=[],
        metavar="FILE",
        help="CJKVI/CHISE/BabelStone-style tab-separated IDS file (repeatable)",
    )
    sources.add_argument(
        "--makemeahanzi",
        action="append",
        default=[],
        metavar="FILE",
        help="Make Me a Hanzi dictionary JSON-lines file (repeatable)",
    )
    sources.add_argument(
        "--ccd",
        action="append",
        default=[],
        metavar="FILE",
        help="npm chinese-characters-decomposition ccd.json (repeatable)",
    )
    sources.add_argument(
        "--merge-policy",
        choices=("first", "last", "shortest"),
        default="first",
        help="resolve conflicting source entries (default: first)",
    )

    targets = parser.add_argument_group("targets")
    targets.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        metavar="START-END",
        help="inclusive hexadecimal Unicode range, e.g. 4E00-9FFF (repeatable)",
    )
    targets.add_argument(
        "--preset",
        action="append",
        default=[],
        metavar="NAME",
        help="Unicode Han block preset (repeatable; default: unified)",
    )
    targets.add_argument(
        "--target-file",
        action="append",
        default=[],
        metavar="FILE",
        help="literal characters, U+ labels, and/or ranges (repeatable)",
    )
    targets.add_argument(
        "--list-presets",
        action="store_true",
        help="print preset names and exit",
    )
    targets.add_argument(
        "--no-leaf-fallback",
        action="store_true",
        help="require every component operand to have a recursive source entry",
    )
    targets.add_argument(
        "--no-binary-roundtrip",
        action="store_true",
        help="skip expanded IDS encode/decode verification",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", metavar="FILE", help="write complete JSON report")
    output.add_argument("--csv", metavar="FILE", help="write per-target CSV report")
    output.add_argument("--text", metavar="FILE", help="write human-readable report")
    output.add_argument(
        "--show-records",
        action="store_true",
        help="include every target row in stdout/text (summary only by default)",
    )
    output.add_argument(
        "--fail-on",
        action="append",
        choices=("missing", "malformed", "cyclic", "collision", "binary"),
        default=[],
        help="return exit status 2 when the selected condition is present",
    )
    output.add_argument(
        "--fail-under",
        type=float,
        metavar="PERCENT",
        help="return exit status 2 when recursive coverage is below PERCENT",
    )
    return parser


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _ensure_parent(path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fail_under is not None and not 0 <= args.fail_under <= 100:
        parser.error("--fail-under must be between 0 and 100")
    if args.list_presets:
        print("\n".join(available_presets()))
        return 0

    datasets = []
    for path in args.ids:
        datasets.append(load_ids_file(path, merge_policy=args.merge_policy))
    for path in args.makemeahanzi:
        datasets.append(load_makemeahanzi(path, merge_policy=args.merge_policy))
    for path in args.ccd:
        datasets.append(load_ccd_json(path, merge_policy=args.merge_policy))
    if not datasets:
        parser.error("provide at least one local source with --ids, --makemeahanzi, or --ccd")
    dataset = merge_datasets(
        *datasets,
        merge_policy=args.merge_policy,
        name=" + ".join(item.name for item in datasets),
    )

    selected: list[str] = []
    preset_names = args.preset
    if not (preset_names or args.ranges or args.target_file):
        preset_names = ["unified"]
    if preset_names:
        try:
            selected.extend(targets_from_presets(preset_names))
        except ValueError as exc:
            parser.error(str(exc))
    for specification in args.ranges:
        try:
            selected.extend(characters_from_ranges((parse_range_spec(specification),)))
        except ValueError as exc:
            parser.error(str(exc))
    for path in args.target_file:
        selected.extend(load_target_file(path))
    selected = _ordered_unique(selected)

    report = audit_coverage(
        dataset,
        selected,
        allow_leaf_fallback=not args.no_leaf_fallback,
        verify_binary_roundtrip=not args.no_binary_roundtrip,
    )
    rendered = report.to_text(include_records=args.show_records)
    print(rendered, end="")

    for path in (args.json, args.csv, args.text):
        _ensure_parent(path)
    if args.json:
        report.write_json(args.json)
    if args.csv:
        report.write_csv(args.csv)
    if args.text:
        report.write_text(args.text, include_records=args.show_records)

    summary = report.summary
    failed = (
        "missing" in args.fail_on
        and summary.missing > 0
        or "malformed" in args.fail_on
        and summary.malformed > 0
        or "cyclic" in args.fail_on
        and summary.cyclic > 0
        or "collision" in args.fail_on
        and summary.reverse_collision_groups > 0
        or "binary" in args.fail_on
        and summary.binary_roundtrip_failures > 0
        or args.fail_under is not None
        and summary.decodable_coverage_ratio * 100 < args.fail_under
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

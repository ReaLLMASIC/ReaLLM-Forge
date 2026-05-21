#!/usr/bin/env python3
"""
Transform a UTF text file into a self-contained reversible token map enriched
with Han main-block decomposition metadata.

Typical flow:
  1. Build the character dataset with han_main_block_decomp.py.
  2. Run this script on a UTF text file.
  3. Optionally edit token[...]["text_current"] or consume token-level
     decomposition features downstream.
  4. Use han_file_decomp_map_reverse.py to recover, diff, and validate.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from han_file_decomp_map_common import (
    SCHEMA_VERSION,
    DEFAULT_COMPONENT_SEPARATOR,
    DEFAULT_PAYLOAD_CODEC,
    HanFileMapError,
    build_token_records,
    count_newline_kinds,
    decode_text,
    dedupe_preserve_order,
    detect_utf_plan,
    encode_payload,
    encode_text,
    iso_utc_now,
    is_main_block_char,
    load_decomposition_subset,
    sha256_bytes,
    sha256_file,
    sha256_text_utf8,
    stderr,
    write_map_document,
    load_map_document,
)

SCRIPT_NAME = "han_file_decomp_map_transform.py"
SCRIPT_VERSION = "1.0.0"


def infer_output_format(output_path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    suffix = output_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return suffix[1:]
    return "json"


def build_mapped_document(
    *,
    input_path: Path,
    dataset_path: Path,
    encoding: str,
    errors: str,
    component_separator: str,
    payload_codec: str,
    embed_original_bytes: bool,
) -> dict:
    raw_bytes = input_path.read_bytes()
    plan = detect_utf_plan(raw_bytes, encoding, errors)
    text = decode_text(raw_bytes, plan)

    # Fail early if the decoded text cannot be losslessly re-encoded under the
    # chosen UTF plan. Exact byte recovery is part of the design contract.
    roundtrip = encode_text(text, plan, preserve_bom=True)
    if roundtrip != raw_bytes:
        raise HanFileMapError(
            "Input file did not round-trip exactly under the chosen UTF encoding plan. "
            "Use the correct UTF encoding and error handler so the reversible map "
            "can preserve exact bytes."
        )

    wanted_han_chars = dedupe_preserve_order(ch for ch in text if is_main_block_char(ch))
    record_lookup = load_decomposition_subset(dataset_path, wanted_han_chars)
    tokens, inventory, _inventory_index = build_token_records(text, plan, record_lookup, component_separator)

    expected_final_byte = len(raw_bytes)
    actual_final_byte = tokens[-1]["positions"]["byte_end"] if tokens else len(plan.bom_bytes)
    if actual_final_byte != expected_final_byte:
        raise HanFileMapError(
            f"Token byte spans do not cover the original file exactly: {actual_final_byte} != {expected_final_byte}"
        )

    missing_han = [
        {
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
        }
        for ch in wanted_han_chars
        if ch not in record_lookup
    ]

    line_count = text.count("\n") + text.count("\r") - count_newline_kinds(text)["CRLF"] + 1 if text else 1
    document = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "script": SCRIPT_NAME,
            "version": SCRIPT_VERSION,
            "created_at_utc": iso_utc_now(),
        },
        "source_document": {
            "path": str(input_path),
            "filename": input_path.name,
            "byte_length": len(raw_bytes),
            "sha256_bytes": sha256_bytes(raw_bytes),
            "text_length_codepoints": len(text),
            "text_sha256_utf8": sha256_text_utf8(text),
            "line_count": line_count,
            "newline_counts": count_newline_kinds(text),
            "encoding_plan": plan.to_dict(),
        },
        "original_payload": encode_payload(raw_bytes, payload_codec) if embed_original_bytes else None,
        "decomposition_dataset": {
            "path": str(dataset_path),
            "filename": dataset_path.name,
            "sha256_bytes": sha256_file(dataset_path),
            "matched_char_count": len(record_lookup),
            "requested_han_char_count": len(wanted_han_chars),
            "missing_han_characters": missing_han,
        },
        "tokenization": {
            "mode": "codepoint",
            "char_offsets_are_half_open": True,
            "byte_offsets_are_half_open": True,
            "byte_offsets_are_absolute": True,
            "component_separator_default": component_separator,
            "token_separator_default": "",
            "non_han_tokens_are_preserved": True,
        },
        "inventory": inventory,
        "tokens": tokens,
    }
    return document


def cmd_build(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_format = infer_output_format(output_path, args.output_format)

    document = build_mapped_document(
        input_path=input_path,
        dataset_path=dataset_path,
        encoding=args.encoding,
        errors=args.errors,
        component_separator=args.component_separator,
        payload_codec=args.payload_codec,
        embed_original_bytes=not args.no_embed_original_bytes,
    )
    write_map_document(document, output_path, output_format)
    stderr("Mapped file written.")
    stderr(
        json.dumps(
            {
                "output": str(output_path),
                "output_format": output_format,
                "token_count": len(document["tokens"]),
                "inventory_count": len(document["inventory"]),
                "matched_han_char_count": document["decomposition_dataset"]["matched_char_count"],
                "missing_han_char_count": len(document["decomposition_dataset"]["missing_han_characters"]),
                "embedded_original_payload": document["original_payload"] is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_self_tests() -> None:
    sample_dataset = [
        {
            "char": "河",
            "codepoint": "U+6CB3",
            "unihan": {
                "kRSUnicode": [{"raw": "85.5", "radical_number": 85}],
                "kTotalStrokes": [8],
                "kDefinition": "river",
            },
            "decomposition": {
                "primary_raw": "⿰氵可",
                "immediate_components": ["氵", "可"],
                "leaf_components_raw": ["氵", "丁", "口"],
                "leaf_components_raw_unique": ["氵", "丁", "口"],
                "leaf_radicals_normalized": ["水", "丁", "口"],
                "leaf_radicals_normalized_unique": ["水", "丁", "口"],
                "unresolved_tokens": [],
                "expanded_tree": {
                    "kind": "op",
                    "value": "⿰",
                    "scheme": "ids",
                    "children": [
                        {"kind": "leaf", "value": "氵", "scheme": "ids"},
                        {
                            "kind": "op",
                            "value": "⿱",
                            "scheme": "ids",
                            "children": [
                                {"kind": "leaf", "value": "丁", "scheme": "ids"},
                                {"kind": "leaf", "value": "口", "scheme": "ids"},
                            ],
                        },
                    ],
                },
            },
        },
        {
            "char": "語",
            "codepoint": "U+8A9E",
            "unihan": {
                "kRSUnicode": [{"raw": "149.7", "radical_number": 149}],
                "kTotalStrokes": [14],
                "kDefinition": "language",
            },
            "decomposition": {
                "primary_raw": "⿰言吾",
                "immediate_components": ["言", "吾"],
                "leaf_components_raw": ["言", "五", "口"],
                "leaf_components_raw_unique": ["言", "五", "口"],
                "leaf_radicals_normalized": ["言", "五", "口"],
                "leaf_radicals_normalized_unique": ["言", "五", "口"],
                "unresolved_tokens": [],
                "expanded_tree": {
                    "kind": "op",
                    "value": "⿰",
                    "scheme": "ids",
                    "children": [
                        {"kind": "leaf", "value": "言", "scheme": "ids"},
                        {
                            "kind": "op",
                            "value": "⿱",
                            "scheme": "ids",
                            "children": [
                                {"kind": "leaf", "value": "五", "scheme": "ids"},
                                {"kind": "leaf", "value": "口", "scheme": "ids"},
                            ],
                        },
                    ],
                },
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dataset_path = tmp / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as fh:
            for record in sample_dataset:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")

        input_path = tmp / "sample.txt"
        raw = ("\ufeff河A語\n").encode("utf-8")
        input_path.write_bytes(raw)

        doc = build_mapped_document(
            input_path=input_path,
            dataset_path=dataset_path,
            encoding="utf-8-sig",
            errors="strict",
            component_separator=" ",
            payload_codec="zlib+base64",
            embed_original_bytes=True,
        )
        assert doc["source_document"]["sha256_bytes"] == sha256_bytes(raw)
        assert len(doc["tokens"]) == 4
        assert doc["tokens"][0]["text_original"] == "河"
        assert doc["tokens"][0]["inventory_index"] == 0
        assert doc["tokens"][0]["decomposition"]["render_normalized_default"] == "水 丁 口"
        assert doc["tokens"][1]["text_original"] == "A"
        assert doc["tokens"][1]["decomposition"] is None
        assert doc["tokens"][2]["text_original"] == "語"
        assert doc["tokens"][3]["kind"] == "newline"
        assert doc["tokens"][0]["positions"]["byte_start"] == 3  # after UTF-8 BOM

        map_path = tmp / "mapped.json"
        write_map_document(doc, map_path, "json")
        loaded = load_map_document(map_path)
        assert loaded["source_document"]["encoding_plan"]["bom_length"] == 3
        assert loaded["inventory"][0]["char"] == "河"

    stderr("All self-tests passed.")


def cmd_self_test(args: argparse.Namespace) -> int:
    run_self_tests()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform a UTF text file into a reversible Han decomposition token map."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="Build a reversible mapped file from a UTF text input.")
    p_build.add_argument("--input", required=True, help="Path to the input UTF text file.")
    p_build.add_argument("--dataset", required=True, help="Path to the decomposition dataset from han_main_block_decomp.py.")
    p_build.add_argument("--output", required=True, help="Output mapped file (.json or .jsonl).")
    p_build.add_argument(
        "--output-format",
        default=None,
        choices=("json", "jsonl"),
        help="Serialization format. If omitted, inferred from the output suffix, otherwise defaults to json.",
    )
    p_build.add_argument("--encoding", default="utf-8", help="Input file encoding. Exact reversible mapping is supported for UTF-8/16/32 variants.")
    p_build.add_argument("--errors", default="strict", help="Unicode decode/encode error handler. Use strict unless you know you need another UTF-safe mode.")
    p_build.add_argument(
        "--component-separator",
        default=DEFAULT_COMPONENT_SEPARATOR,
        help="Separator used for default token-level decomposition renderings.",
    )
    p_build.add_argument(
        "--payload-codec",
        default=DEFAULT_PAYLOAD_CODEC,
        choices=("base64", "zlib+base64"),
        help="How to embed the exact original file bytes in the mapped file.",
    )
    p_build.add_argument(
        "--no-embed-original-bytes",
        action="store_true",
        help="Do not embed the original raw bytes. The map will still contain token spans, but exact original-byte recovery will then require the source file.",
    )
    p_build.set_defaults(func=cmd_build)

    p_self = subparsers.add_parser("self-test", help="Run internal mapper self-tests.")
    p_self.set_defaults(func=cmd_self_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HanFileMapError as exc:
        stderr(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        stderr("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

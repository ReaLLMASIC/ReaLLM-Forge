#!/usr/bin/env python3
"""
Create a readable symbolic compact serialization of a transformed Han file map.

Two entry paths are supported:
  * build     : start from input.txt + the Han decomposition dataset
  * from-map  : start from an existing mapped JSON/JSONL document

The resulting file stays human-readable:
  * a legend maps short symbol IDs back to original tokens and their serialized
    decomposition/current-text view;
  * the body is a wrapped symbol stream with visible escapes for control chars;
  * the original UTF file can be recovered exactly from the legend + body.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from han_file_decomp_map_common import (
    DEFAULT_COMPONENT_SEPARATOR,
    HanFileMapError,
    load_map_document,
    sha256_bytes,
    stderr,
)
from han_file_decomp_map_transform import build_mapped_document
from han_file_symbolic_common import (
    DEFAULT_SYMBOLIC_SENTINEL,
    DEFAULT_SYMBOLIC_WRAP_WIDTH,
    DEFAULT_SYMBOL_ALPHABET_PROFILE,
    HanReadableSymbolicError,
    VALID_SERIALIZATION_MODES,
    build_readable_symbolic_document_from_map,
    load_readable_symbolic_document,
    readable_symbolic_compare_original_hash,
    reconstruct_original_bytes,
    reconstruct_original_text,
    reconstruct_serialized_text,
    write_readable_symbolic_document,
)

SCRIPT_NAME = "han_file_symbolic_serialize.py"
SCRIPT_VERSION = "1.0.0"


def _summarize_document(document: dict) -> dict:
    stats = document.get("stats") or {}
    source = document.get("source_document") or {}
    serialization = document.get("serialization") or {}
    return {
        "source_filename": source.get("filename"),
        "source_byte_length": source.get("byte_length"),
        "source_sha256_bytes": source.get("sha256_bytes"),
        "token_count": stats.get("token_count"),
        "symbol_entry_count": stats.get("symbol_entry_count"),
        "symbolized_token_count": stats.get("symbolized_token_count"),
        "han_symbol_entry_count": stats.get("han_symbol_entry_count"),
        "han_symbolized_token_count": stats.get("han_symbolized_token_count"),
        "body_length_codepoints": stats.get("body_length_codepoints"),
        "body_sha256_utf8": stats.get("body_sha256_utf8"),
        "mode": serialization.get("mode"),
        "sentinel": serialization.get("sentinel"),
        "symbol_id_width": serialization.get("symbol_id_width"),
        "symbolize_non_han": serialization.get("symbolize_non_han"),
    }



def build_symbolic_from_map_document(args: argparse.Namespace, mapped_document: dict) -> dict:
    return build_readable_symbolic_document_from_map(
        mapped_document,
        mode=args.mode,
        component_separator=args.component_separator,
        preserve_non_han=not args.drop_non_han,
        missing_han_fallback=args.missing_han_fallback,
        include_han_metadata=args.include_han_metadata,
        symbolize_non_han=args.symbolize_non_han,
        sentinel=args.sentinel,
        symbol_alphabet_profile=args.alphabet_profile,
        symbol_id_width=args.symbol_id_width,
    )



def cmd_build(args: argparse.Namespace) -> int:
    mapped_document = build_mapped_document(
        input_path=Path(args.input),
        dataset_path=Path(args.dataset),
        encoding=args.encoding,
        errors=args.errors,
        component_separator=args.component_separator,
        payload_codec="zlib+base64",
        embed_original_bytes=False,
    )
    symbolic_document = build_symbolic_from_map_document(args, mapped_document)
    output_path = Path(args.output)
    write_readable_symbolic_document(symbolic_document, output_path, wrap_width=args.wrap_width)

    recovered = reconstruct_original_bytes(symbolic_document, errors=args.errors)
    source_hash = (symbolic_document.get("source_document") or {}).get("sha256_bytes")
    summary = _summarize_document(symbolic_document)
    summary.update(
        {
            "output": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "recovered_sha256_bytes": sha256_bytes(recovered),
            "matches_stored_source_hash": sha256_bytes(recovered) == source_hash,
        }
    )
    stderr("Readable symbolic file written.")
    stderr(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0



def cmd_from_map(args: argparse.Namespace) -> int:
    map_path = Path(args.map)
    mapped_document = load_map_document(map_path)
    symbolic_document = build_symbolic_from_map_document(args, mapped_document)
    output_path = Path(args.output)
    write_readable_symbolic_document(symbolic_document, output_path, wrap_width=args.wrap_width)

    summary = _summarize_document(symbolic_document)
    summary.update(
        {
            "input_map": str(map_path),
            "input_map_size_bytes": map_path.stat().st_size,
            "output": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "symbolic_vs_map_size_ratio": (
                output_path.stat().st_size / map_path.stat().st_size if map_path.stat().st_size else None
            ),
        }
    )
    stderr("Readable symbolic file written from map.")
    stderr(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0



def cmd_inspect(args: argparse.Namespace) -> int:
    document = load_readable_symbolic_document(Path(args.symbolic))
    report = _summarize_document(document)
    report["compare_to_stored_hash"] = readable_symbolic_compare_original_hash(document, errors=args.errors)
    if args.include_previews:
        report["preview_original_text"] = reconstruct_original_text(document)[: args.preview_chars]
        report["preview_serialized_text"] = reconstruct_serialized_text(document)[: args.preview_chars]
        report["preview_body"] = (document.get("body") or "")[: args.preview_chars]
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
                "expanded_tree": None,
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
                "expanded_tree": None,
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

        input_path = tmp / "input.txt"
        raw_text = "河語A 河\n語¤\t\r"
        input_path.write_text(raw_text, encoding="utf-8")

        mapped = build_mapped_document(
            input_path=input_path,
            dataset_path=dataset_path,
            encoding="utf-8",
            errors="strict",
            component_separator=" ",
            payload_codec="zlib+base64",
            embed_original_bytes=False,
        )
        symbolic = build_readable_symbolic_document_from_map(
            mapped,
            mode="decomp-normalized",
            component_separator=" ",
            preserve_non_han=True,
            missing_han_fallback="current",
            include_han_metadata=True,
            symbolize_non_han=False,
            sentinel="¤",
            symbol_id_width=1,
        )

        original_text = reconstruct_original_text(symbolic)
        serialized_text = reconstruct_serialized_text(symbolic)
        original_bytes = reconstruct_original_bytes(symbolic, errors="strict")
        assert original_text == raw_text
        assert original_bytes == input_path.read_bytes()
        assert serialized_text.startswith("水 丁 口言 五 口A ")
        assert "¤~n" in symbolic["body"]
        assert "¤~t" in symbolic["body"]
        assert "¤~r" in symbolic["body"]
        assert "¤~s" in symbolic["body"]

        path = tmp / "sample.hsym"
        write_readable_symbolic_document(symbolic, path, wrap_width=10)
        loaded = load_readable_symbolic_document(path)
        assert reconstruct_original_text(loaded) == raw_text
        assert reconstruct_original_bytes(loaded, errors="strict") == input_path.read_bytes()
        compare = readable_symbolic_compare_original_hash(loaded, errors="strict")
        assert compare["bytes_equal_to_stored_hash"] is True



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_symbolic_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=sorted(VALID_SERIALIZATION_MODES), default="decomp-normalized")
        p.add_argument("--component-separator", default=DEFAULT_COMPONENT_SEPARATOR)
        p.add_argument(
            "--missing-han-fallback",
            choices=["current", "original", "empty"],
            default="current",
            help="How to serialize Han characters that are missing decomposition leaves.",
        )
        p.add_argument("--drop-non-han", action="store_true", help="Serialize non-Han tokens as empty strings.")
        p.add_argument(
            "--symbolize-non-han",
            action="store_true",
            help="Put non-Han tokens into the symbol legend too, instead of preserving safe literals inline.",
        )
        p.add_argument("--include-han-metadata", action="store_true")
        p.add_argument("--sentinel", default=DEFAULT_SYMBOLIC_SENTINEL)
        p.add_argument("--alphabet-profile", default=DEFAULT_SYMBOL_ALPHABET_PROFILE)
        p.add_argument("--symbol-id-width", type=int, default=None)
        p.add_argument("--wrap-width", type=int, default=DEFAULT_SYMBOLIC_WRAP_WIDTH)

    p_build = sub.add_parser("build", help="Build a readable symbolic file from input text + dataset")
    p_build.add_argument("--input", required=True)
    p_build.add_argument("--dataset", required=True)
    p_build.add_argument("--output", required=True)
    p_build.add_argument("--encoding", default="utf-8")
    p_build.add_argument("--errors", default="strict")
    add_symbolic_options(p_build)
    p_build.set_defaults(func=cmd_build)

    p_from_map = sub.add_parser("from-map", help="Build a readable symbolic file from an existing mapped file")
    p_from_map.add_argument("--map", required=True)
    p_from_map.add_argument("--output", required=True)
    add_symbolic_options(p_from_map)
    p_from_map.set_defaults(func=cmd_from_map)

    p_inspect = sub.add_parser("inspect", help="Inspect a readable symbolic file")
    p_inspect.add_argument("--symbolic", required=True)
    p_inspect.add_argument("--errors", default="strict")
    p_inspect.add_argument("--include-previews", action="store_true")
    p_inspect.add_argument("--preview-chars", type=int, default=240)
    p_inspect.set_defaults(func=cmd_inspect)

    p_self = sub.add_parser("self-test", help="Run internal self-tests")
    p_self.set_defaults(func=lambda args: (run_self_tests(), print("self-test: ok"), 0)[-1])

    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (HanReadableSymbolicError, HanFileMapError) as exc:
        stderr(f"error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

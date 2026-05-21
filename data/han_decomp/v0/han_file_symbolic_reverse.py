#!/usr/bin/env python3
"""
Recover and compare outputs from the readable symbolic Han serialization format.

Primary use:
  * recover the exact original input.txt bytes from the symbolic text file
  * optionally render the original text, serialized/transformed text, or the
    symbolic body itself
  * compare recovered output against the original file or stored hashes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from han_file_decomp_map_common import (
    diff_lines,
    first_difference,
    sha256_bytes,
    sha256_text_utf8,
    stderr,
)
from han_file_symbolic_common import (
    HanReadableSymbolicError,
    load_readable_symbolic_document,
    readable_symbolic_compare_original_hash,
    reconstruct_original_bytes,
    reconstruct_original_text,
    reconstruct_serialized_text,
)
from han_file_symbolic_serialize import run_self_tests as run_serialize_self_tests

SCRIPT_NAME = "han_file_symbolic_reverse.py"
SCRIPT_VERSION = "1.0.0"
TEXT_MODES = {"original-text", "serialized-text", "symbolic-body"}
ALL_MODES = {"original-bytes", *TEXT_MODES}



def recover_output(
    document: dict,
    *,
    mode: str,
    text_output_encoding: str,
    text_output_errors: str,
) -> bytes:
    if mode == "original-bytes":
        return reconstruct_original_bytes(document, errors=text_output_errors)
    if mode == "original-text":
        return reconstruct_original_text(document).encode(text_output_encoding, text_output_errors)
    if mode == "serialized-text":
        return reconstruct_serialized_text(document).encode(text_output_encoding, text_output_errors)
    if mode == "symbolic-body":
        return (document.get("body") or "").encode(text_output_encoding, text_output_errors)
    raise HanReadableSymbolicError(f"Unsupported recover mode: {mode}")



def _decode_for_diff(data: bytes, encoding: str, errors: str) -> str:
    return data.decode(encoding, errors)



def cmd_recover(args: argparse.Namespace) -> int:
    document = load_readable_symbolic_document(Path(args.symbolic))
    recovered = recover_output(
        document,
        mode=args.mode,
        text_output_encoding=args.text_output_encoding,
        text_output_errors=args.text_output_errors,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(recovered)

    report = {
        "output": str(output_path),
        "mode": args.mode,
        "byte_length": len(recovered),
        "sha256": sha256_bytes(recovered),
    }
    if args.mode == "original-bytes":
        report["stored_hash_check"] = readable_symbolic_compare_original_hash(
            document,
            errors=args.text_output_errors,
        )
    stderr(json.dumps(report, ensure_ascii=False, indent=2))
    return 0



def cmd_compare(args: argparse.Namespace) -> int:
    document = load_readable_symbolic_document(Path(args.symbolic))
    recovered = recover_output(
        document,
        mode=args.mode,
        text_output_encoding=args.text_output_encoding,
        text_output_errors=args.text_output_errors,
    )

    if args.compare_to:
        other_bytes = Path(args.compare_to).read_bytes()
        compare_target_label = str(Path(args.compare_to))
    else:
        if args.mode != "original-bytes":
            raise HanReadableSymbolicError("--compare-to is required for non-original-bytes compare mode")
        stored = readable_symbolic_compare_original_hash(document, errors=args.text_output_errors)
        report = {
            "mode": args.mode,
            "compare_to": "stored_source_hash",
            **stored,
        }
        if args.report_json:
            path = Path(args.report_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if stored["bytes_equal_to_stored_hash"] else 1

    equal = recovered == other_bytes
    report = {
        "mode": args.mode,
        "compare_to": compare_target_label,
        "equal": equal,
        "recovered_byte_length": len(recovered),
        "compare_to_byte_length": len(other_bytes),
        "recovered_sha256": sha256_bytes(recovered),
        "compare_to_sha256": sha256_bytes(other_bytes),
        "first_difference": None if equal else first_difference(recovered, other_bytes),
    }

    if args.mode in TEXT_MODES or args.decode_compare_as_text:
        recovered_text = _decode_for_diff(recovered, args.text_output_encoding, args.text_output_errors)
        other_text = _decode_for_diff(other_bytes, args.text_output_encoding, args.text_output_errors)
        report["recovered_text_sha256_utf8"] = sha256_text_utf8(recovered_text)
        report["compare_to_text_sha256_utf8"] = sha256_text_utf8(other_text)
        if not equal:
            diff = diff_lines(
                other_text,
                recovered_text,
                fromfile=compare_target_label,
                tofile=f"recovered:{args.mode}",
                context=args.diff_context,
            )
            report["unified_diff_sample"] = diff[: args.diff_lines_limit]

    if args.report_json:
        path = Path(args.report_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if equal else 1



def cmd_inspect(args: argparse.Namespace) -> int:
    document = load_readable_symbolic_document(Path(args.symbolic))
    report = {
        "schema_version": document.get("schema_version"),
        "generator": document.get("generator"),
        "source_document": document.get("source_document"),
        "source_map": document.get("source_map"),
        "serialization": document.get("serialization"),
        "stats": document.get("stats"),
        "stored_hash_check": readable_symbolic_compare_original_hash(document, errors=args.text_output_errors),
    }
    if args.include_previews:
        report["preview_original_text"] = reconstruct_original_text(document)[: args.preview_chars]
        report["preview_serialized_text"] = reconstruct_serialized_text(document)[: args.preview_chars]
        report["preview_body"] = (document.get("body") or "")[: args.preview_chars]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_recover = sub.add_parser("recover", help="Recover output from a readable symbolic file")
    p_recover.add_argument("--symbolic", required=True)
    p_recover.add_argument("--mode", choices=sorted(ALL_MODES), default="original-bytes")
    p_recover.add_argument("--output", required=True)
    p_recover.add_argument("--text-output-encoding", default="utf-8")
    p_recover.add_argument("--text-output-errors", default="strict")
    p_recover.set_defaults(func=cmd_recover)

    p_compare = sub.add_parser("compare", help="Compare recovered output to a file or stored hash")
    p_compare.add_argument("--symbolic", required=True)
    p_compare.add_argument("--mode", choices=sorted(ALL_MODES), default="original-bytes")
    p_compare.add_argument("--compare-to")
    p_compare.add_argument("--text-output-encoding", default="utf-8")
    p_compare.add_argument("--text-output-errors", default="strict")
    p_compare.add_argument("--decode-compare-as-text", action="store_true")
    p_compare.add_argument("--diff-context", type=int, default=3)
    p_compare.add_argument("--diff-lines-limit", type=int, default=60)
    p_compare.add_argument("--report-json")
    p_compare.set_defaults(func=cmd_compare)

    p_inspect = sub.add_parser("inspect", help="Inspect a readable symbolic file")
    p_inspect.add_argument("--symbolic", required=True)
    p_inspect.add_argument("--text-output-errors", default="strict")
    p_inspect.add_argument("--include-previews", action="store_true")
    p_inspect.add_argument("--preview-chars", type=int, default=240)
    p_inspect.set_defaults(func=cmd_inspect)

    p_self = sub.add_parser("self-test", help="Run internal self-tests")
    p_self.set_defaults(func=lambda args: (run_serialize_self_tests(), print("self-test: ok"), 0)[-1])

    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HanReadableSymbolicError as exc:
        stderr(f"error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

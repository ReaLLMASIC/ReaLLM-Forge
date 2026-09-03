#!/usr/bin/env python3
"""
tokenize_and_annotate_sizes.py

Reads the *filtered* JSON produced by filter_files_by_script.py (one entry per file),
runs prepare.py (assumed symlinked as ./prepare.py) to tokenize each text file with
100% train split (no val), writes:

  text_<lang>_<script>_<tokenization_type>.bin

and then appends/updates:

  entry["tokenized_sizes"][<tokenization_type>] = <size_kb_of_bin>

Example output entry:
{
  "language": "ace",
  "script": "Latn",
  "lang_script": "ace_Latn",
  "size_kb": 277.0,
  "tokenized_sizes": {"tiktoken": 300.0},
  "filename": "text_ace_Latn.txt"
}

Notes:
- prepare.py writes meta.pkl in the current directory and overwrites it each run.
- This script runs tokenization sequentially to avoid meta.pkl races.
- Uses --percentage_train 1.0 to ensure 100% of the file is tokenized and no val is written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def run_prepare(
    prepare_path: Path,
    input_txt: Path,
    out_bin: Path,
    method: str,
    tiktoken_encoding: str,
    additional_tokens_file: str | None,
    extra_args: List[str],
) -> None:
    cmd = [
        "python3",
        str(prepare_path),
        "--method",
        method,
        "-t",
        str(input_txt),
        "--train_output",
        str(out_bin),
        "--percentage_train",
        "1.0",
    ]

    # Only pass tiktoken args when relevant
    if method == "tiktoken":
        cmd += ["--tiktoken_encoding", tiktoken_encoding]
        if additional_tokens_file:
            cmd += ["--additional_tokens_file", additional_tokens_file]

    # Allow power-users to append any extra prepare.py args
    cmd += extra_args

    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--in-json",
        default="filtered_files.json",
        help="Input JSON from filter_files_by_script.py",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Output JSON (default: overwrite --in-json)",
    )
    ap.add_argument(
        "--prepare",
        default="./prepare.py",
        help="Path to prepare.py (symlink in cwd is fine)",
    )
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Directory where the text_*.txt files live (default: cwd)",
    )

    # Tokenization selection (start with tiktoken, but allow switching)
    ap.add_argument(
        "--method",
        choices=[
            "tiktoken",
            "sentencepiece",
            "char",
            "custom",
            "byte",
            "custom_char_byte_fallback",
            "json_byte_fallback",
            "python_programming",
            "sinewave",
        ],
        default="tiktoken",
        help="Tokenizer method to run via prepare.py",
    )

    # tiktoken-specific knobs (ignored for other methods)
    ap.add_argument(
        "--tiktoken-encoding",
        choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base"],
        default="gpt2",
        help="tiktoken encoding (only used if --method tiktoken)",
    )
    ap.add_argument(
        "--additional-tokens-file",
        default=None,
        help="JSON file of additional special tokens for tiktoken (only used if --method tiktoken)",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run tokenization even if the output .bin already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run, but do not execute prepare.py or write json",
    )

    # Pass-through to prepare.py (optional)
    ap.add_argument(
        "--prepare-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Everything after this flag is passed to prepare.py verbatim. "
             "Example: --prepare-extra-args -T",
    )

    args = ap.parse_args()

    in_json = Path(args.in_json)
    out_json = Path(args.out_json) if args.out_json else in_json
    prepare_path = Path(args.prepare)
    base_dir = Path(args.base_dir)

    if not prepare_path.exists():
        raise SystemExit(f"prepare.py not found at: {prepare_path}")
    if not in_json.exists():
        raise SystemExit(f"Input JSON not found: {in_json}")

    rows: List[Dict[str, Any]]
    with in_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit("Expected input JSON to be a list of objects")

    method = args.method

    for entry in rows:
        # expected from your filtered file:
        # language, script, lang_script, size_kb, filename (optional)
        filename = entry.get("filename")
        if not filename:
            # If filename was dropped, reconstruct from lang/script
            lang = entry["language"]
            script = entry["script"]
            filename = f"text_{lang}_{script}.txt"

        input_txt = base_dir / filename
        if not input_txt.exists():
            # skip missing files (common if base_dir wrong)
            print(f"[skip] missing input: {input_txt}")
            continue

        # Output: text_<lang>_<script>_<method>.bin
        lang = entry["language"]
        script = entry["script"]
        out_bin = base_dir / f"text_{lang}_{script}_{method}.bin"

        # Ensure tokenized_sizes map exists
        tok_sizes = entry.get("tokenized_sizes")
        if not isinstance(tok_sizes, dict):
            tok_sizes = {}
            entry["tokenized_sizes"] = tok_sizes

        if out_bin.exists() and not args.force:
            # already computed? still record size in json
            kb = file_size_kb(out_bin)
            tok_sizes[method] = kb
            print(f"[reuse] {out_bin.name} {kb:.1f} KB")
            continue

        cmd_preview = f"python3 {prepare_path} --method {method} -t {input_txt} --train_output {out_bin} --percentage_train 1.0"
        if method == "tiktoken":
            cmd_preview += f" --tiktoken_encoding {args.tiktoken_encoding}"
            if args.additional_tokens_file:
                cmd_preview += f" --additional_tokens_file {args.additional_tokens_file}"
        if args.prepare_extra_args:
            cmd_preview += " " + " ".join(args.prepare_extra_args)

        if args.dry_run:
            print(f"[dry-run] {cmd_preview}")
            continue

        print(f"[run] {cmd_preview}")
        try:
            run_prepare(
                prepare_path=prepare_path,
                input_txt=input_txt,
                out_bin=out_bin,
                method=method,
                tiktoken_encoding=args.tiktoken_encoding,
                additional_tokens_file=args.additional_tokens_file,
                extra_args=args.prepare_extra_args,
            )
        except subprocess.CalledProcessError as e:
            print(f"[error] tokenization failed for {input_txt.name} ({method}): {e}")
            continue

        if not out_bin.exists():
            print(f"[error] expected output missing: {out_bin}")
            continue

        kb = file_size_kb(out_bin)
        tok_sizes[method] = kb
        print(f"[ok] {out_bin.name} {kb:.1f} KB")

    if args.dry_run:
        return

    # Write updated JSON
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote updated json: {out_json}")


if __name__ == "__main__":
    main()


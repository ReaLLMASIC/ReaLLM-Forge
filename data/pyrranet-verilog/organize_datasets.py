#!/usr/bin/env python3
"""
Download PyraNet-Verilog from Hugging Face and extract code samples.

Adds local CSV caching:
- By default, writes a CSV snapshot alongside extracted files:
    <output-dir>/pyrranet_verilog_train.csv
- If the CSV exists, we read from it and skip Hugging Face download.
- Use --refresh-csv to force redownload/regenerate the CSV.

Notes:
- We stream from Hugging Face by default (unless --no-streaming).
- CSV writing is done incrementally (no need to load full dataset in memory).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset

# Allow very large Verilog sources in CSV (code field can exceed 128KB)
try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, AttributeError):
    # Some platforms require an int smaller than sys.maxsize
    csv.field_size_limit(2**31 - 1)


DATASET_NAME = "bnadimi/PyraNet-Verilog"
DATASET_SPLIT = "train"
DEFAULT_CSV_NAME = "pyrranet_verilog_train.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download PyraNet-Verilog and split each code sample into its own file "
            "named orig_<index>.v. Also caches a local CSV snapshot for reuse."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "orig",
        help="Directory to write orig_<index>.v files.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for number of rows to extract.",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (downloads the entire dataset).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing orig_<index>.v files if they already exist.",
    )

    # CSV caching controls
    p.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help=(
            "Path to local CSV snapshot. Default: <output-dir>/pyrranet_verilog_train.csv. "
            "If it exists, we'll read from it instead of downloading."
        ),
    )
    p.add_argument(
        "--no-csv-cache",
        action="store_true",
        help="Do not create/read the CSV cache; always use Hugging Face dataset.",
    )
    p.add_argument(
        "--refresh-csv",
        action="store_true",
        help="Force redownload/regenerate CSV even if it already exists.",
    )
    return p.parse_args()


def iter_rows_from_csv(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def write_csv_snapshot(
    csv_path: Path,
    rows: Iterable[Dict],
    max_rows: Optional[int] = None,
) -> int:
    """
    Write rows (dicts) to csv_path incrementally.
    Returns count written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    writer = None
    f = csv_path.open("w", encoding="utf-8", newline="")
    try:
        for row in rows:
            if writer is None:
                # Stabilize column order based on first row
                fieldnames = list(row.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()

            writer.writerow(row)
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    finally:
        f.close()

    return count


def hf_dataset_iter(streaming: bool) -> Iterable[Dict]:
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=streaming)
    for row in ds:
        yield row


def write_verilog_samples(
    output_dir: Path,
    rows: Iterable[Dict],
    max_rows: Optional[int],
    overwrite: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for row in rows:
        filename = output_dir / f"orig_{count:07d}.v"

        # allow skipping existing files
        if filename.exists() and not overwrite:
            count += 1
            if max_rows is not None and count >= max_rows:
                break
            continue

        code = row.get("code")
        if code is None:
            count += 1
            if max_rows is not None and count >= max_rows:
                break
            continue

        # Ensure trailing newline; strip only outer whitespace
        filename.write_text(str(code).strip() + "\n", encoding="utf-8")

        count += 1
        if max_rows is not None and count >= max_rows:
            break

    return count


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    max_rows: Optional[int] = args.max_rows
    streaming = not args.no_streaming
    overwrite: bool = args.overwrite

    csv_path = args.csv_path or (output_dir / DEFAULT_CSV_NAME)

    # Decide data source:
    # 1) If CSV cache allowed and exists and not refresh -> read CSV
    # 2) Else -> read from Hugging Face; optionally write CSV snapshot
    use_csv_cache = (not args.no_csv_cache)

    if use_csv_cache and csv_path.exists() and not args.refresh_csv:
        print(f"[organize_datasets] Using existing CSV cache: {csv_path}")
        rows_iter = iter_rows_from_csv(csv_path)
        total = write_verilog_samples(output_dir, rows_iter, max_rows=max_rows, overwrite=overwrite)
        print(f"[organize_datasets] Wrote {total} files to {output_dir}")
        return

    # Otherwise download from HF (streaming by default)
    print(f"[organize_datasets] Loading dataset {DATASET_NAME} split={DATASET_SPLIT} streaming={streaming}")
    hf_iter = hf_dataset_iter(streaming=streaming)

    if use_csv_cache:
        # We want both: CSV snapshot + verilog files
        # To avoid double-iteration, we buffer rows for the verilog writer by streaming once:
        # We'll write CSV first while also collecting rows into a small list if max_rows is small;
        # but for general streaming, we re-open by reading CSV after writing it.
        #
        # Approach:
        #   - Write CSV snapshot (up to max_rows if set)
        #   - Then read CSV back and write verilog files from it (deterministic)
        print(f"[organize_datasets] Writing CSV snapshot: {csv_path}")
        wrote_csv = write_csv_snapshot(csv_path, hf_iter, max_rows=max_rows)
        print(f"[organize_datasets] CSV rows written: {wrote_csv}")

        print(f"[organize_datasets] Writing Verilog files from CSV cache...")
        rows_iter = iter_rows_from_csv(csv_path)
        total = write_verilog_samples(output_dir, rows_iter, max_rows=max_rows, overwrite=overwrite)
        print(f"[organize_datasets] Wrote {total} files to {output_dir}")
    else:
        # No CSV cache: write verilog files directly from HF iterator
        total = write_verilog_samples(output_dir, hf_iter, max_rows=max_rows, overwrite=overwrite)
        print(f"[organize_datasets] Wrote {total} files to {output_dir}")


if __name__ == "__main__":
    main()


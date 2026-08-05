#!/usr/bin/env python3
"""Build regular multicontext integer datasets from CSV columns.

Every column is range-checked against an inclusive [int_min, int_max] and then
stored as token ids shifted to zero by subtracting int_min. Each column folder
gets an independent vocab_size of int_max - int_min + 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence



@dataclass(frozen=True)
class IntRange:
    int_min: int
    int_max: int

    @property
    def vocab_size(self) -> int:
        return self.int_max - self.int_min + 1


def parse_int_range(text: str) -> IntRange:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected <int_min>:<int_max>")
    int_min, int_max = int(parts[0]), int(parts[1])
    if int_min > int_max:
        raise argparse.ArgumentTypeError("int_min must be <= int_max")
    return IntRange(int_min, int_max)


def parse_column_range(text: str) -> tuple[str, IntRange]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected <column>:<int_min>:<int_max>")
    column = parts[0].strip()
    if not column:
        raise argparse.ArgumentTypeError("Column name/index cannot be empty")
    int_min, int_max = int(parts[1]), int(parts[2])
    if int_min > int_max:
        raise argparse.ArgumentTypeError("int_min must be <= int_max")
    return column, IntRange(int_min, int_max)


def clean_context_name(name: str, fallback_idx: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return cleaned or f"col_{fallback_idx}"


def ensure_unique(values: Sequence[str], label: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    if dupes:
        raise ValueError(f"Duplicate {label}: {sorted(dupes)}")


def column_aliases(raw_name: str, context_name: str, idx: int) -> set[str]:
    return {raw_name, context_name, str(idx), f"col_{idx}", f"column_{idx}", f"column_{idx + 1}"}


def read_rows(input_csv: Path, has_header: bool) -> tuple[list[str], list[list[int]]]:
    rows: list[list[int]] = []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc

        if has_header:
            headers = [cell.strip() for cell in first]
            if any(not h for h in headers):
                raise ValueError("Header cells must be non-empty")
            ensure_unique(headers, "CSV headers")
        else:
            headers = [f"col_{i}" for i in range(len(first))]
            rows.append(parse_int_row(first, 1, len(headers)))

        for row_num, row in enumerate(reader, start=2 if has_header else 2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            rows.append(parse_int_row(row, row_num, len(headers)))

    if len(headers) == 0:
        raise ValueError("CSV must contain at least one column")
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two data rows")
    return headers, rows


def parse_int_row(row: Sequence[str], row_num: int, expected_cols: int) -> list[int]:
    if len(row) != expected_cols:
        raise ValueError(f"Row {row_num} has {len(row)} columns; expected {expected_cols}")
    values: list[int] = []
    for col_idx, cell in enumerate(row):
        stripped = cell.strip()
        if stripped == "":
            raise ValueError(f"Empty cell at row {row_num}, column {col_idx}")
        try:
            values.append(int(stripped, 10))
        except ValueError as exc:
            raise ValueError(f"Non-integer cell at row {row_num}, column {col_idx}: {cell!r}") from exc
    return values


def resolve_ranges(
    *,
    headers: Sequence[str],
    context_names: Sequence[str],
    range_specs: Iterable[tuple[str, IntRange]],
    default_range: IntRange | None,
) -> list[IntRange]:
    ranges: list[IntRange | None] = [None] * len(headers)
    alias_to_idx: dict[str, int] = {}
    for idx, (header, context_name) in enumerate(zip(headers, context_names)):
        for alias in column_aliases(header, context_name, idx):
            alias_to_idx[alias] = idx

    for column_key, int_range in range_specs:
        if column_key not in alias_to_idx:
            raise ValueError(
                f"Range specified for unknown column {column_key!r}. "
                f"Known columns/aliases include: {sorted(alias_to_idx)[:20]}"
            )
        ranges[alias_to_idx[column_key]] = int_range

    resolved: list[IntRange] = []
    missing: list[str] = []
    for idx, maybe_range in enumerate(ranges):
        if maybe_range is None:
            if default_range is None:
                missing.append(headers[idx])
            else:
                resolved.append(default_range)
        else:
            resolved.append(maybe_range)

    if missing:
        raise ValueError(
            "Missing integer ranges for columns: "
            + ", ".join(missing)
            + ". Provide --range <column>:<int_min>:<int_max> or --default_range <int_min>:<int_max>."
        )
    return resolved


def storage_type_for_vocab(vocab_size: int) -> tuple[str, str]:
    return ("I", "uint32") if vocab_size > 65536 else ("H", "uint16")


def write_token_file(path: Path, token_ids: Sequence[int], typecode: str) -> None:
    values = array(typecode, token_ids)
    with path.open("wb") as f:
        values.tofile(f)


def write_values_csv(path: Path, raw_values: Sequence[int], token_ids: Sequence[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "raw_value", "token_id"])
        for idx, (raw, token) in enumerate(zip(raw_values, token_ids)):
            writer.writerow([idx, int(raw), int(token)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert integer CSV columns into regular multicontext datasets.")
    parser.add_argument("--input_csv", default="input.csv", help="Input CSV path.")
    parser.add_argument("--output_root", default="csv_mc_int", help="Output folder under data/.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in (0, 1).")
    parser.add_argument("--has_header", dest="has_header", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--no_header", dest="has_header", action="store_false", help="Treat first row as data and name folders col_0, col_1, ...")
    parser.add_argument("--range", dest="ranges", action="append", default=[], type=parse_column_range, help="Repeatable <column>:<int_min>:<int_max> range. Columns can be names, col_N, or N.")
    parser.add_argument("--default_range", type=parse_int_range, default=None, help="Default <int_min>:<int_max> for columns without --range.")
    parser.add_argument("--allow_out_of_range", action="store_true", help="Clip out-of-range values instead of failing after printing the fit report.")
    parser.add_argument("--save_values_csv", action="store_true", help="Save raw/token pairs in each column folder.")
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    headers, rows = read_rows(input_csv, args.has_header)
    context_names = [clean_context_name(h, i) for i, h in enumerate(headers)]
    ensure_unique(context_names, "sanitized output folder names")
    ranges = resolve_ranges(headers=headers, context_names=context_names, range_specs=args.ranges, default_range=args.default_range)

    total_rows = len(rows)
    total_cols = len(headers)
    columns = [[row[idx] for row in rows] for idx in range(total_cols)]
    actual_mins = [min(column) for column in columns]
    actual_maxs = [max(column) for column in columns]
    fits = [(actual_mins[i] >= r.int_min and actual_maxs[i] <= r.int_max) for i, r in enumerate(ranges)]

    print(f"Input: {input_csv}")
    print(f"Header row: {args.has_header}")
    print(f"Rows: {total_rows}; columns: {total_cols}")
    print("Column range fit report:")
    for idx, (header, context_name, int_range, fits_range) in enumerate(zip(headers, context_names, ranges, fits)):
        print(
            f"  [{idx}] folder={context_name!r} column={header!r} "
            f"declared=[{int_range.int_min},{int_range.int_max}] "
            f"actual=[{int(actual_mins[idx])},{int(actual_maxs[idx])}] "
            f"vocab_size={int_range.vocab_size} fits={fits_range}"
        )

    if not all(fits) and not args.allow_out_of_range:
        raise ValueError("At least one column has values outside its declared range. Re-run with corrected ranges or --allow_out_of_range.")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "data" / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    train_n = int(total_rows * args.train_ratio)
    if train_n <= 0 or train_n >= total_rows:
        raise ValueError(f"Invalid split: train={train_n}, total={total_rows}")

    manifest = {
        "tokenizer": "csv_integer_range_multicontext_manifest",
        "source_csv": str(input_csv),
        "has_header": bool(args.has_header),
        "rows": int(total_rows),
        "train_rows": int(train_n),
        "val_rows": int(total_rows - train_n),
        "output_root": args.output_root,
        "multicontext_datasets": [],
        "columns": [],
    }

    for idx, (header, context_name, int_range) in enumerate(zip(headers, context_names, ranges)):
        context_dir = output_root / context_name
        context_dir.mkdir(parents=True, exist_ok=True)
        raw_values = columns[idx]
        clipped = [min(max(value, int_range.int_min), int_range.int_max) for value in raw_values]
        token_ids = [value - int_range.int_min for value in clipped]
        typecode, dtype_name = storage_type_for_vocab(int_range.vocab_size)
        write_token_file(context_dir / "train.bin", token_ids[:train_n], typecode)
        write_token_file(context_dir / "val.bin", token_ids[train_n:], typecode)
        if args.save_values_csv:
            write_values_csv(context_dir / "values.csv", raw_values, token_ids)

        dataset_name = f"{args.output_root}/{context_name}"
        meta = {
            "tokenizer": "csv_integer_range",
            "vocab_size": int(int_range.vocab_size),
            "source_csv": str(input_csv),
            "source_column": header,
            "context_name": context_name,
            "column_index": int(idx),
            "has_header": bool(args.has_header),
            "int_min": int(int_range.int_min),
            "int_max": int(int_range.int_max),
            "value_encoding": "token_id = raw_integer_value - int_min",
            "value_decoding": "raw_integer_value = token_id + int_min",
            "actual_int_min": int(actual_mins[idx]),
            "actual_int_max": int(actual_maxs[idx]),
            "fits_declared_range": bool(fits[idx]),
            "dtype": dtype_name,
            "samples": int(total_rows),
            "train_ratio": float(args.train_ratio),
        }
        with (context_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        manifest["multicontext_datasets"].append(dataset_name)
        manifest["columns"].append(meta)

    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Output root: {output_root}")
    print("Multicontext datasets:")
    for dataset in manifest["multicontext_datasets"]:
        print(f"  {dataset}")


if __name__ == "__main__":
    main()

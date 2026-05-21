#!/usr/bin/env python3
"""Create integer-quantized numerical multicontext datasets from CSV columns.

Manual pipeline per value:
    transformed = (raw + shift) * scale
    quantized_int = round(transformed)

Auto-range pipeline per value:
    transformed = out_min + ((raw - raw_min) / (raw_max - raw_min)) * (out_max - out_min)
    quantized_int = round(transformed)

Each column becomes one context directory under data/<output_root>/.
Optionally, each context directory can also contain a CSV export of the quantized values.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


UINT16_MIN = 0
UINT16_MAX = 65535


def parse_column_transform(spec: str) -> Tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --column-transform '{spec}'. Expected <column>:<shift>:<scale>")
    column, shift_str, scale_str = parts
    if not column:
        raise ValueError(f"Invalid --column-transform '{spec}': empty column")
    return column, float(shift_str), float(scale_str)


def clean_context_name(header: str, fallback_idx: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in header.strip()).strip("_")
    return cleaned or f"column_{fallback_idx + 1}"


def get_context_name(header: str, idx: int, use_col_index_names: bool) -> str:
    if use_col_index_names:
        return f"col_{idx}"
    return clean_context_name(header, idx)


def read_csv_columns(input_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError("CSV must contain a header row")

        data: Dict[str, List[float]] = {h: [] for h in headers}
        for row_idx, row in enumerate(reader, start=2):
            for header in headers:
                cell = (row.get(header) or "").strip()
                if cell == "":
                    raise ValueError(f"Empty cell at row {row_idx}, column '{header}'")
                data[header].append(float(cell))

    total = len(data[headers[0]])
    if total < 2:
        raise ValueError("CSV must contain at least 2 rows")
    if any(len(col) != total for col in data.values()):
        raise ValueError("All columns must contain the same number of rows")
    return headers, data


def auto_range_transform(raw: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    raw_min = float(raw.min())
    raw_max = float(raw.max())

    if raw_max == raw_min:
        midpoint = (out_min + out_max) / 2.0
        return np.full_like(raw, midpoint, dtype=np.float32)

    scaled = (raw - raw_min) / (raw_max - raw_min)
    return out_min + scaled * (out_max - out_min)


def write_quantized_csv(
    out_path: Path,
    source_column_name: str,
    original_header: str,
    quantized: np.ndarray,
    raw: np.ndarray,
    transformed: np.ndarray,
) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "source_column",
                "original_csv_header",
                "raw_value",
                "transformed_value",
                "quantized_uint16",
            ]
        )
        for i, (r, t, q) in enumerate(zip(raw, transformed, quantized)):
            writer.writerow([i, source_column_name, original_header, float(r), float(t), int(q)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build int-quantized multicontext datasets from CSV numeric columns.")
    parser.add_argument("--input_csv", default="input.csv", help="Path to input CSV (header required).")
    parser.add_argument("--output_root", default="csv_num_mc_int", help="Output folder under data/.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio in (0,1).")

    parser.add_argument(
        "--column-transform",
        action="append",
        default=[],
        help="Repeatable '<column>:<shift>:<scale>' for manual mode.",
    )

    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Automatically map each column's raw min/max into [--auto_range_min, --auto_range_max].",
    )
    parser.add_argument(
        "--auto_range_min",
        type=float,
        default=5000.0,
        help="Lower bound for automatic per-column range mapping.",
    )
    parser.add_argument(
        "--auto_range_max",
        type=float,
        default=25000.0,
        help="Upper bound for automatic per-column range mapping.",
    )

    parser.add_argument("--clip_min", type=int, default=UINT16_MIN, help="Minimum integer value after quantization.")
    parser.add_argument("--clip_max", type=int, default=UINT16_MAX, help="Maximum integer value after quantization.")

    parser.add_argument(
        "--save_output_csv",
        action="store_true",
        help="Also save a CSV of quantized values into each column folder.",
    )
    parser.add_argument(
        "--use_col_index_folder_names",
        action="store_true",
        help="Use folder names like col_<column_index> instead of sanitized header names. Default is false.",
    )

    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")
    if args.clip_min < UINT16_MIN or args.clip_max > UINT16_MAX or args.clip_min > args.clip_max:
        raise ValueError(f"clip range must be within [0,{UINT16_MAX}] and clip_min<=clip_max")
    if args.auto_range_min > args.auto_range_max:
        raise ValueError("--auto_range_min must be <= --auto_range_max")
    if args.column_transform and args.auto_range:
        raise ValueError("Use either --column-transform or --auto-range, not both")

    transforms: Dict[str, Tuple[float, float]] = {}
    for item in args.column_transform:
        column, shift, scale = parse_column_transform(item)
        transforms[column] = (shift, scale)

    input_path = Path(args.input_csv).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    headers, data = read_csv_columns(input_path)
    total_samples = len(data[headers[0]])
    train_n = int(total_samples * args.train_ratio)
    if train_n <= 0 or train_n >= total_samples:
        raise ValueError(f"Invalid split: train={train_n}, total={total_samples}")

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / "data" / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output root: {output_root}")
    print(f"Rows/column: total={total_samples}, train={train_n}, val={total_samples - train_n}")
    print(f"Save output CSVs: {args.save_output_csv}")
    print(f"Use col_<index> folder names: {args.use_col_index_folder_names}")

    if args.auto_range:
        print(f"Mode: auto-range [{args.auto_range_min}, {args.auto_range_max}]")
    else:
        print("Mode: manual shift/scale")

    for idx, header in enumerate(headers):
        raw = np.asarray(data[header], dtype=np.float32)

        if args.auto_range:
            transformed = auto_range_transform(raw, args.auto_range_min, args.auto_range_max)
            shift = None
            scale = None
            raw_min = float(raw.min())
            raw_max = float(raw.max())
        else:
            shift, scale = transforms.get(header, (0.0, 1.0))
            transformed = (raw + shift) * scale
            raw_min = None
            raw_max = None

        rounded = np.rint(transformed)
        quantized = np.clip(rounded, args.clip_min, args.clip_max).astype(np.uint16)

        context_name = get_context_name(header, idx, args.use_col_index_folder_names)
        source_column_name = context_name if args.use_col_index_folder_names else header

        context_dir = output_root / context_name
        context_dir.mkdir(parents=True, exist_ok=True)

        quantized[:train_n].tofile(context_dir / "train.bin")
        quantized[train_n:].tofile(context_dir / "val.bin")

        if args.save_output_csv:
            write_quantized_csv(
                context_dir / "quantized_output.csv",
                source_column_name=source_column_name,
                original_header=header,
                quantized=quantized,
                raw=raw,
                transformed=transformed,
            )

        if args.auto_range:
            quantization_meta = {
                "type": "auto_minmax_to_range_round_clip_uint16",
                "raw_min": raw_min,
                "raw_max": raw_max,
                "target_min": float(args.auto_range_min),
                "target_max": float(args.auto_range_max),
                "clip_min": int(args.clip_min),
                "clip_max": int(args.clip_max),
                "inverse": "raw ~= raw_min + ((quantized - target_min) / (target_max - target_min)) * (raw_max - raw_min)",
            }
        else:
            quantization_meta = {
                "type": "shift_scale_round_clip_uint16",
                "shift": float(shift),
                "scale": float(scale),
                "clip_min": int(args.clip_min),
                "clip_max": int(args.clip_max),
                "inverse": "raw ~= (quantized / scale) - shift (for scale != 0)",
            }

        meta = {
            "tokenizer": "csv_quantized_int",
            "vocab_size": UINT16_MAX + 1,
            "numerical_multicontext_input_format": "scalar",
            "source_csv": str(input_path),
            "source_column": source_column_name,
            "original_csv_header": header,
            "context_name": context_name,
            "column_index": idx,
            "samples": int(total_samples),
            "train_ratio": float(args.train_ratio),
            "quantization": quantization_meta,
            "float_range_raw_min": float(raw.min()),
            "float_range_raw_max": float(raw.max()),
            "float_range_transformed_min": float(transformed.min()),
            "float_range_transformed_max": float(transformed.max()),
            "int_range_quantized_min": int(quantized.min()),
            "int_range_quantized_max": int(quantized.max()),
            "saved_quantized_csv": bool(args.save_output_csv),
        }
        with (context_dir / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

        if args.auto_range:
            print(
                f"[{context_name}] column='{header}', source_column='{source_column_name}', "
                f"auto_range=({args.auto_range_min},{args.auto_range_max}), "
                f"raw[min,max]=({raw.min():.6g},{raw.max():.6g}), "
                f"tx[min,max]=({transformed.min():.6g},{transformed.max():.6g}), "
                f"q[min,max]=({quantized.min()},{quantized.max()})"
            )
        else:
            print(
                f"[{context_name}] column='{header}', source_column='{source_column_name}', "
                f"shift={shift}, scale={scale}, "
                f"raw[min,max]=({raw.min():.6g},{raw.max():.6g}), "
                f"tx[min,max]=({transformed.min():.6g},{transformed.max():.6g}), "
                f"q[min,max]=({quantized.min()},{quantized.max()})"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Utilities for downloading and preprocessing the MIDI detailed note dataset.

This script downloads the Lakh detailed MIDI note parquet shards hosted on
Hugging Face and converts them into two derived artefacts:

* JSON files grouped by MIDI note number (0-127). Each JSON file stores
  chronologically ordered messages for a specific note.
* A combined CSV file with all records for quick inspection.

While emitting rows, a rolling window keyed on ``floor(start_time)`` is kept.
Whenever the integer portion of ``start_time`` changes the buffered rows are
sorted by ``start_time`` before being written out. This guards against the
observed bug where the source data briefly regresses inside the same second
(e.g. ``285`` → ``285.25`` → ``285``).

Examples
--------

```
python3 get_dataset.py
```

```
python3 get_dataset.py --output-dir ./processed --skip-download
```
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional


PARQUET_SOURCES = {
    "p=e": "https://huggingface.co/datasets/nintorac/midi_etl/resolve/main/lakh/detailed_notes/p%3De/data_0.parquet?download=true",
    "p=f": "https://huggingface.co/datasets/nintorac/midi_etl/resolve/main/lakh/detailed_notes/p%3Df/data_0.parquet?download=true",
}

OUTPUT_FIELDS = [
    "midi_id",
    "track_id",
    "message_id",
    "start_time",
    "duration",
    "velocity",
    "note",
    "set_type",
    "p",
]

DEFAULT_BATCH_SIZE = 50_000


class JsonArrayWriter:
    """Incrementally write dictionaries into a JSON array."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        self._file.write("[")
        self._first = True

    def write(self, obj: Dict[str, object]) -> None:
        if self._first:
            self._file.write("\n")
            self._first = False
        else:
            self._file.write(",\n")
        json.dump(obj, self._file, ensure_ascii=False)

    def close(self) -> None:
        if self._first:
            # No entries were written; emit an empty array.
            self._file.write("]")
        else:
            self._file.write("\n]")
        self._file.close()


def download_file(url: str, destination: str) -> None:
    """Download *url* to *destination* with a progress bar."""

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    requests = _import_requests()
    tqdm = _import_tqdm()
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MiB chunks

    with open(destination, "wb") as fh, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=os.path.basename(destination),
    ) as progress:
        for chunk in response.iter_content(block_size):
            fh.write(chunk)
            progress.update(len(chunk))


def _import_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise SystemExit(
            "requests is required to download the parquet files. Install it via 'pip install requests'."
        ) from exc
    return requests


def _import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise SystemExit("tqdm is required for download progress bars. Install it via 'pip install tqdm'.") from exc
    return tqdm


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise SystemExit(
            "pyarrow is required to process the parquet files. Install it via 'pip install pyarrow'."
        ) from exc
    return pq


def is_nan(value: object) -> bool:
    try:
        return math.isnan(value)  # type: ignore[arg-type]
    except TypeError:
        return False


def normalize_value(value: object) -> Optional[object]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, bool)):
        return int(value)
    if isinstance(value, float):
        return None if math.isnan(value) else float(value)
    if is_nan(value):
        return None
    return value


def sanitize_row(row: Dict[str, object]) -> Dict[str, Optional[object]]:
    sanitized: Dict[str, Optional[object]] = {}
    for key in OUTPUT_FIELDS:
        sanitized[key] = normalize_value(row.get(key))
    return sanitized


def compute_floor(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return math.floor(numeric)


def batch_to_rows(batch) -> Iterator[Dict[str, object]]:
    columns = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
    names = list(batch.schema.names)
    for values in zip(*columns):
        yield dict(zip(names, values))


def start_time_sort_key(row: Dict[str, Optional[object]]) -> tuple:
    start = row.get("start_time")
    if start is None:
        return (1, 0.0)
    return (0, float(start))


@dataclass
class ProcessingContext:
    csv_writer: csv.DictWriter
    note_writers: Dict[int, JsonArrayWriter]
    counts_by_note: Counter


def ensure_note_writer(note: int, context: ProcessingContext, output_dir: str) -> JsonArrayWriter:
    if note not in context.note_writers:
        file_path = os.path.join(output_dir, f"note_{note:03d}.json")
        context.note_writers[note] = JsonArrayWriter(file_path)
    return context.note_writers[note]


def write_rows(rows: Iterable[Dict[str, Optional[object]]], context: ProcessingContext, note_dir: str) -> int:
    emitted = 0
    for row in rows:
        csv_row = {key: ("" if row.get(key) is None else row.get(key)) for key in OUTPUT_FIELDS}
        context.csv_writer.writerow(csv_row)

        note_val = row.get("note")
        if note_val is None:
            continue
        try:
            note = int(note_val)
        except (TypeError, ValueError):
            continue
        writer = ensure_note_writer(note, context, note_dir)
        writer.write(row)
        context.counts_by_note[note] += 1
        emitted += 1
    return emitted


def flush_window(window: List[Dict[str, Optional[object]]], context: ProcessingContext, note_dir: str) -> int:
    if not window:
        return 0
    ordered = sorted(window, key=start_time_sort_key)
    emitted = write_rows(ordered, context, note_dir)
    window.clear()
    return emitted


def process_parquet(
    parquet_path: str,
    context: ProcessingContext,
    note_dir: str,
    batch_size: int,
) -> int:
    pq = _import_pyarrow()
    parquet_file = pq.ParquetFile(parquet_path)
    window: List[Dict[str, Optional[object]]] = []
    current_floor: Optional[int] = None
    processed = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for raw_row in batch_to_rows(batch):
            sanitized = sanitize_row(raw_row)
            floor_value = compute_floor(sanitized.get("start_time"))

            if floor_value is not None:
                if current_floor is None:
                    current_floor = floor_value
                elif floor_value != current_floor:
                    processed += flush_window(window, context, note_dir)
                    current_floor = floor_value
            window.append(sanitized)

    processed += flush_window(window, context, note_dir)
    return processed


def run(output_dir: str, skip_download: bool, batch_size: int) -> None:
    download_dir = os.path.join(output_dir, "downloaded_parquets")
    note_dir = os.path.join(output_dir, "notes_by_number")
    csv_path = os.path.join(output_dir, "all_notes.csv")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(note_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        csv_writer.writeheader()
        context = ProcessingContext(csv_writer=csv_writer, note_writers={}, counts_by_note=Counter())

        for shard, url in PARQUET_SOURCES.items():
            file_name = f"{shard.replace('=', '_')}_{os.path.basename(url.split('?', 1)[0])}"
            parquet_path = os.path.join(download_dir, file_name)
            if not skip_download or not os.path.exists(parquet_path):
                print(f"Downloading {url} → {parquet_path}")
                download_file(url, parquet_path)
            else:
                print(f"Skipping download for {parquet_path} (already exists)")

            print(f"Processing {parquet_path}")
            processed = process_parquet(parquet_path, context, note_dir, batch_size)
            print(f"Processed {processed} rows from {parquet_path}")

    # Finalise JSON arrays
    for writer in context.note_writers.values():
        writer.close()

    summary_path = os.path.join(output_dir, "note_counts.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {str(note): count for note, count in sorted(context.counts_by_note.items())},
            summary_file,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and split the MIDI detailed note dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Destination directory for the downloaded and processed artefacts.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing parquet files instead of downloading them again.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of parquet rows to load per batch while streaming (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir, args.skip_download, args.batch_size)


if __name__ == "__main__":
    main()

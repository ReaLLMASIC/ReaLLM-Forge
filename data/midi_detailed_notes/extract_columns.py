#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path

from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TextColumn,
    SpinnerColumn,
    TransferSpeedColumn,
)

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB

def count_lines_with_progress(path: Path) -> int:
    """Count newline characters using chunked reads with a progress bar by bytes."""
    total_bytes = path.stat().st_size if path.exists() else 0
    if total_bytes == 0:
        # Empty or missing file; we'll let open() raise later if missing
        return 0

    with Progress(
        TextColumn("[bold blue]Stage 1/2: Counting lines"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TransferSpeedColumn(),  # bytes/s here
    ) as progress:
        task = progress.add_task("Counting...", total=total_bytes)
        n_lines = 0
        with open(path, "rb") as fb:
            while True:
                chunk = fb.read(CHUNK_SIZE)
                if not chunk:
                    break
                n_lines += chunk.count(b"\n")
                progress.update(task, advance=len(chunk))
    return n_lines

def format_and_save(input_path: Path, output_path: Path, skip_header: bool = False):
    total_rows = count_lines_with_progress(input_path)

    # If the file has a header and user wants to skip it, subtract 1 from the progress total (not below 0)
    progress_total = max(0, total_rows - (1 if skip_header and total_rows > 0 else 0))

    written = 0
    skipped = 0

    with open(input_path, newline='') as fin, open(output_path, 'w', newline='') as fout, Progress(
        TextColumn("[bold green]Stage 2/2: Processing rows"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TransferSpeedColumn(),  # rows/s here
    ) as progress:
        reader = csv.reader(fin)
        # Space-delimited output
        writer = csv.writer(fout, delimiter=" ", quoting=csv.QUOTE_MINIMAL)

        task = progress.add_task("Processing...", total=progress_total)

        # Optionally skip a header row
        if skip_header:
            try:
                next(reader, None)
            except StopIteration:
                pass  # empty file

        for row in reader:
            # Need at least 7 columns (0..6)
            if len(row) < 7:
                skipped += 1
                progress.update(task, advance=1)
                continue
            try:
                col4 = float(row[3])         # two decimals
                col5 = float(row[4])         # three decimals
                col6 = int(float(row[5]))    # ints; tolerate "6.0"
                col7 = int(float(row[6]))

                writer.writerow([f"{col4:.2f}", f"{col5:.3f}", str(col6), str(col7)])
                written += 1
            except ValueError:
                skipped += 1
            progress.update(task, advance=1)

    # Final summary to stdout
    print(
        f"[Summary] Input: {input_path}  →  Output: {output_path}\n"
        f"          Rows written: {written}   Skipped: {skipped}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Stream a CSV and save columns 4–7 with formatting "
                    "(4: 2dp, 5: 3dp, 6–7 ints) to a space-delimited text file."
    )
    parser.add_argument("csv_file", nargs="?", default="all_notes.csv",
                        help="Path to the input CSV (default: all_notes.csv)")
    parser.add_argument("-o", "--out", default="input.txt",
                        help='Output file path (default: "input.txt")')
    parser.add_argument("--skip-header", action="store_true",
                        help="Skip the first row of the CSV (treat as header)")
    args = parser.parse_args()

    in_path = Path(args.csv_file)
    out_path = Path(args.out)

    format_and_save(in_path, out_path, skip_header=args.skip_header)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# data/template/utils/filter_csv_by_regex.py
import argparse
import csv
import re
import sys
import textwrap


def resolve_target_field(fieldnames, column_name: str | None, col_index: int | None) -> str:
    if column_name:
        if column_name in fieldnames:
            return column_name
        lowered = {fn.lower(): fn for fn in fieldnames}
        if column_name.lower() in lowered:
            return lowered[column_name.lower()]
        raise SystemExit(f"Error: column '{column_name}' not found. Available: {fieldnames}")

    if col_index is not None:
        if col_index < 1 or col_index > len(fieldnames):
            raise SystemExit(
                f"Error: --col-index {col_index} out of range. "
                f"CSV has {len(fieldnames)} columns (1..{len(fieldnames)})."
            )
        return fieldnames[col_index - 1]

    if "tag" in fieldnames:
        return "tag"
    return fieldnames[0]


def main():
    epilog_text = textwrap.dedent(
        """\
        Examples:
          # Exclude any row where tag column is exactly 'rock'
          ./filter_csv_by_regex.py songs.csv no_rock.csv exclude '^rock$' --column tag

          # Include only rows where tag column contains 'rock' (case-insensitive)
          ./filter_csv_by_regex.py songs.csv only_rock.csv include 'rock' --column tag --ignore-case

          # Include rows where the 3rd column ends with 'rock'
          ./filter_csv_by_regex.py songs.csv col3_rock.csv include 'rock$' --col-index 3

          # Exclude rows where title column starts with a digit
          ./filter_csv_by_regex.py songs.csv no_number_titles.csv exclude '^[0-9]' --column title
        """
    )

    parser = argparse.ArgumentParser(
        description="Filter CSV rows by regex on a chosen column (by name or 1-based index).",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("output_csv", help="Path to output CSV")
    parser.add_argument(
        "mode",
        choices=["exclude", "include"],
        help="exclude: remove rows that match; include: keep only rows that match",
    )
    parser.add_argument("regex", help="Regex to test against the target column")
    parser.add_argument(
        "--ignore-case", action="store_true", help="Case-insensitive regex"
    )
    parser.add_argument(
        "--column",
        help="Column name to match against (wins over --col-index if both provided)",
    )
    parser.add_argument(
        "--col-index",
        type=int,
        help="1-based column index to match against (e.g., 1 for first column)",
    )

    args = parser.parse_args()

    flags = re.IGNORECASE if args.ignore_case else 0
    try:
        pattern = re.compile(args.regex, flags)
    except re.error as e:
        raise SystemExit(f"Invalid regex: {e}")

    with open(args.input_csv, newline="", encoding="utf-8") as infile, \
         open(args.output_csv, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise SystemExit("Error: CSV appears to have no header row.")
        target_field = resolve_target_field(reader.fieldnames, args.column, args.col_index)

        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            value = row.get(target_field, "")
            match = bool(pattern.search(value))
            if (args.mode == "exclude" and not match) or (args.mode == "include" and match):
                writer.writerow(row)


if __name__ == "__main__":
    main()


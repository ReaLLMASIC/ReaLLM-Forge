#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Extract keys from a JSON object into a JSON array"
    )
    parser.add_argument(
        "input",
        help="Path to input JSON file containing a key→value mapping"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (default: <input>_tokens.json)"
    )
    parser.add_argument(
        "--no-length-sort",
        action="store_true",
        help="Disable sorting tokens by length (longest → shortest)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Default output: <input>_tokens.json
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_tokens.json")
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = list(data.keys())

    # Default behavior: longest → shortest
    if not args.no_length_sort:
        tokens.sort(key=len, reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

    sort_mode = "length-desc" if not args.no_length_sort else "original order"
    print(f"Wrote {len(tokens)} tokens to {output_path} ({sort_mode})")


if __name__ == "__main__":
    main()


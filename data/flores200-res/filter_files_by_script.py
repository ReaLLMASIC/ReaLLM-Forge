
#!/usr/bin/env python3
"""
filter_files_by_script.py

Read files.json and emit a simplified JSON with only fields
relevant to script/language analysis.

Keeps:
  - language (ISO 639-3)
  - script (ISO 15924)
  - lang_script (language_script)
  - size_kb (float)
  - filename (optional but useful)
"""

import json
import re
import argparse

FNAME_RE = re.compile(r"^text_([a-z]{3})_([A-Za-z]{4})\.txt$")


def parse_size_to_kb(size_str: str) -> float:
    """
    Convert ls -h style sizes to KB.
    """
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]?)(B?)\s*$", size_str, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized size string: {size_str!r}")

    val = float(m.group(1))
    unit = m.group(2).upper()

    mult = {
        "": 1.0 / 1024.0,  # bytes -> KB
        "K": 1.0,
        "M": 1024.0,
        "G": 1024.0**2,
        "T": 1024.0**3,
        "P": 1024.0**4,
    }[unit]

    return val * mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="files.json", help="Input files.json")
    ap.add_argument("--out", default="filtered_scripts.json", help="Output JSON")
    ap.add_argument("--drop-filename", action="store_true",
                    help="Do not include original filename in output")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    filtered = []

    for r in rows:
        name = r.get("name", "")
        m = FNAME_RE.match(name)
        if not m:
            continue

        lang, script = m.groups()
        size_kb = parse_size_to_kb(str(r["size"]))

        entry = {
            "language": lang,
            "script": script,
            "lang_script": f"{lang}_{script}",
            "size_kb": size_kb,
        }

        if not args.drop_filename:
            entry["filename"] = name

        filtered.append(entry)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(filtered)} entries to {args.out}")


if __name__ == "__main__":
    main()


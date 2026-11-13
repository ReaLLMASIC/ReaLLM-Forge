#!/usr/bin/env python3
"""Turn an integer CSV (an export_mcint.py file, or what ReaLLM-Forge's sample.py wrote from
one) back into real units, using the .columns.json that came with the export.

    ./tools/mcint_decode.py out/csv_mc_int/csv_samples/sample.csv bio-dash_2026-10-07_1s_mcint.columns.json
    ./tools/mcint_decode.py sample.csv cols.json --out sample_real.csv

temp_c_x10 214 -> temp_c 21.4; a "no reading" value -> an empty cell. Columns are matched by
name, so a CSV with fewer columns (or in another order) works. Standard library only.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_dashboard"))
import mcint  # noqa: E402


def real_name(c):
    n = c["column"]
    return n[: -len(f"_x{c['scale']}")] if c["scale"] != 1 and n.endswith(f"_x{c['scale']}") else n


def main():
    ap = argparse.ArgumentParser(description="Integer CSV -> real units.")
    ap.add_argument("csv")
    ap.add_argument("columns_json")
    ap.add_argument("--out", help="default: <csv>_real.csv beside the input")
    a = ap.parse_args()
    spec = {c["column"]: c for c in json.load(open(a.columns_json))["columns"]}
    out = a.out or os.path.splitext(a.csv)[0] + "_real.csv"
    with open(a.csv, newline="") as f, open(out, "w", newline="") as g:
        r, w = csv.reader(f), csv.writer(g)
        head = [h.strip() for h in next(r)]
        unknown = [h for h in head if h not in spec]
        if unknown:
            sys.exit(f"Not in {a.columns_json}: {', '.join(unknown)}")
        cols = [spec[h] for h in head]
        w.writerow([real_name(c) for c in cols])
        n = bad = 0
        for row in r:
            if not row:
                continue
            try:
                vals = mcint.decode_row(cols, row)
            except ValueError:
                bad += 1
                continue
            w.writerow(["" if v is None else (f"{v:g}" if isinstance(v, float) else v) for v in vals])
            n += 1
    print(f"Wrote {n:,} rows to {out}" + (f" ({bad} rows skipped: not whole numbers)" if bad else ""))


if __name__ == "__main__":
    main()

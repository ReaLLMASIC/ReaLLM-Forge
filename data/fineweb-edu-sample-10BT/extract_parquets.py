"""Extract text from downloaded parquet files into input.txt."""

import os
import sys
import pandas as pd

parquet_dir = sys.argv[1] if len(sys.argv) > 1 else "downloaded_parquets"
output_file = sys.argv[2] if len(sys.argv) > 2 else "input.txt"

files = sorted(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
print(f"Found {len(files)} parquet files in {parquet_dir}")

total = 0
with open(output_file, "w", encoding="utf-8") as out:
    for fname in files:
        path = os.path.join(parquet_dir, fname)
        df = pd.read_parquet(path, columns=["text"])
        for text in df["text"]:
            if text:
                out.write(text.strip() + "\n")
        total += len(df)
        print(f"  {fname}: {len(df):,} rows (total: {total:,})")

print(f"Done. Wrote {total:,} examples to {output_file}")

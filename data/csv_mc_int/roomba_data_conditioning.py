#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="input.csv")
    parser.add_argument("--output_csv", default="roomba_integer.csv")
    parser.add_argument("--mapping_csv", default="action_mapping.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # 1. Encode action column as integers
    df["action"], action_categories = pd.factorize(df["action"])

    mapping = pd.DataFrame({
        "id": range(len(action_categories)),
        "action": action_categories,
    })
    mapping.to_csv(args.mapping_csv, index=False)

    # 2. Battery percent: multiply by 10
    df["battery_percent"] = (df["battery_percent"] * 10).round().astype(int)

    # 3. total_distance_mm modulo 1000
    df["total_distance_mm"] = df["total_distance_mm"].astype(int) % 1000

    # 4. timestamp modulo 10
    df["timestamp"] = df["timestamp"].astype(int) % 10

    # 5. Make all remaining numeric columns integer-safe
    for col in df.columns:
        if col != "action":
            df[col] = pd.to_numeric(df[col], errors="raise")
            df[col] = df[col].round().astype(int)

    df.to_csv(args.output_csv, index=False)

    print(f"Wrote conditioned CSV: {args.output_csv}")
    print(f"Wrote action mapping: {args.mapping_csv}")


if __name__ == "__main__":
    main()

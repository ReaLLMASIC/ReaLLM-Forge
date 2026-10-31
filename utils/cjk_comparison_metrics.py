"""Read validation metrics and cumulative training counts from token snapshots."""

import argparse
import csv
import json
import math


FIELDS = ("avg_target_probability", "avg_target_rank", "avg_left_probability")


def training_token_stats(path):
    """Count cumulative training targets from only the latest exported snapshot."""
    latest_iteration = -1
    counts = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"iteration", "token_id", "training_seen_count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing training token counts")
        for row in reader:
            iteration = int(row["iteration"])
            if iteration > latest_iteration:
                latest_iteration = iteration
                counts = {}
            if iteration == latest_iteration:
                count = int(row["training_seen_count"])
                if count < 0:
                    raise ValueError(f"{path} has a negative training token count")
                counts[int(row["token_id"])] = count
    if not counts:
        raise ValueError(f"{path} contains no training token rows")
    return {
        "vocab_size": len(counts),
        "training_tokens_iteration": latest_iteration,
        "training_tokens_processed": sum(counts.values()),
        "training_unique_token_ids": sum(count > 0 for count in counts.values()),
    }


def snapshot_metrics(path, iteration):
    totals = {field: 0.0 for field in FIELDS}
    counts = {field: 0 for field in FIELDS}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing probability/rank metrics")
        for row in reader:
            if int(row["iteration"]) != iteration:
                continue
            count = int(row["val_eval_count"])
            for field in FIELDS:
                value = float(row[field])
                if count > 0 and math.isfinite(value):
                    totals[field] += value * count
                    counts[field] += count
    if not all(counts.values()):
        raise ValueError(f"{path} has no evaluated probability/rank metrics at {iteration}")
    return [totals[field] / counts[field] for field in FIELDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("iteration", type=int, nargs="?")
    parser.add_argument("--training-stats", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.training_stats:
        stats = training_token_stats(args.csv_path)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)
                handle.write("\n")
        print("\t".join(str(stats[key]) for key in (
            "training_tokens_iteration", "training_tokens_processed", "training_unique_token_ids",
        )))
        return
    if args.iteration is None:
        parser.error("iteration is required unless --training-stats is used")
    if args.output_json:
        parser.error("--output-json requires --training-stats")
    print("\t".join(f"{value:.8g}" for value in snapshot_metrics(args.csv_path, args.iteration)))


if __name__ == "__main__":
    main()

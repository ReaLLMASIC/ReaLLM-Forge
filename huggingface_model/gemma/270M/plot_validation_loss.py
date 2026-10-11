#!/usr/bin/env python3
"""
Plot validation loss vs global step from one or more Hugging Face Trainer runs.

Each run directory should contain `trainer_state.json` (saved by Trainer checkpoints).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_eval_curve(run_dir: Path):
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}")

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    eval_records = [
        entry for entry in state.get("log_history", [])
        if "eval_loss" in entry and "step" in entry
    ]
    steps = [record["step"] for record in eval_records]
    losses = [record["eval_loss"] for record in eval_records]
    return steps, losses


def parse_run(value: str):
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = Path(path).name
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description="Plot validation loss curves for Gemma finetuning runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in format LABEL=PATH (or just PATH). Repeat for multiple runs.",
    )
    parser.add_argument("--title", type=str, default="Validation loss per iteration")
    parser.add_argument("--output", type=str, default="validation_loss_comparison.png")
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))
    for run_spec in args.run:
        label, run_dir = parse_run(run_spec)
        steps, losses = load_eval_curve(run_dir)
        plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.title(args.title)
    plt.xlabel("Global step")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()

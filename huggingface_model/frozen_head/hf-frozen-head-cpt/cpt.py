#!/usr/bin/env python3
"""Hugging Face frozen-head continual-pretraining experiment CLI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import load_config, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "run", "anchor", "train", "report", "evaluate"))
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", default="full")
    p.add_argument("--lr", type=float)
    p.add_argument("--stop-after", type=int, help="Checkpoint and interrupt after this step; preserves the original schedule")
    p.add_argument("--gains", type=float, nargs="+", default=[0.025, 0.05, 0.1, 0.2])
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    c = load_config(args.config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.command == "prepare":
        from prepare_data import prepare
        prepare(c)
    elif args.command == "preflight":
        from experiment import preflight
        preflight(c)
    elif args.command in ("anchor", "train"):
        from experiment import train
        seed = c["seeds"][0] if args.seed is None else args.seed
        lr = c["lrs"][0] if args.lr is None else args.lr
        return train(c, seed, "a" if args.command == "anchor" else "b", args.mode, lr, args.stop_after)
    elif args.command == "run":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Run with python on one GPU, not torchrun; arms are sequential")
        from prepare_data import prepare
        prepare(c)
        # Spawn a fresh process per run to release every model, optimizer, and CUDA allocation.
        base = [sys.executable, str(Path(__file__).resolve())]
        config = str(Path(args.config).resolve())
        for seed in c["seeds"]:
            commands = [["anchor", "--config", config, "--seed", str(seed)]]
            commands += [["train", "--config", config, "--seed", str(seed), "--mode", mode, "--lr", str(lr)]
                         for lr in c["lrs"] for mode in c["modes"]]
            for command in commands:
                result = subprocess.run(base + command)
                if result.returncode:
                    return result.returncode
        from analyze import report
        report(c, args.gains, not args.no_plots)
    elif args.command == "report":
        from analyze import report
        report(c, args.gains, not args.no_plots)
    elif args.command == "evaluate":
        # Re-evaluate a completed run's saved HF checkpoint without restoring optimizer state.
        from common import run_dir
        from experiment import latest_checkpoint
        from model_utils import load_model, setup_device
        from diagnostics import data_blocks, evaluate, rare_tokens
        seed = c["seeds"][0] if args.seed is None else args.seed
        lr = c["lrs"][0] if args.lr is None else args.lr
        root = run_dir(c, seed, args.mode, lr)
        device, _ = setup_device(c)
        checkpoint = latest_checkpoint(root)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint for {root}")
        model = load_model(str(checkpoint), c).to(device)
        rare = rare_tokens(Path(c["output"]) / "data", model.get_output_embeddings().out_features, device)
        scores = {d: evaluate(model, data_blocks(c, d, "test"), c, device, rare) for d in ("a", "b")}
        write_json(root / "reevaluated_test.json", scores)
        print(scores)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume committed checkpoints.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            print("GPU memory budget exceeded. Keep this run's config unchanged for resume. "
                  "For a new experiment, reduce micro_batch/seq_len/head_chunk, or choose a smaller model, "
                  "and use a NEW output directory. The runner never silently changes precision or optimizer.", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Evaluate validation loss vs. block size for all ckpt.pt files under a directory.

This script mirrors the checkpoint loading and eval flow used in sample.py, including
support for extending block size at inference using GPT.update_block_size().
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# Ensure repository root is importable regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_conf import GPTConfig
from model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find ckpt.pt files, evaluate validation loss for each over "
            "multiple block sizes, and write a Plotly HTML report."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to recursively search for ckpt.pt files.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="minipile",
        help="Dataset under data/<dataset>/val.bin to evaluate on (default: minipile).",
    )
    parser.add_argument(
        "--block_sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of block sizes to evaluate, e.g. --block_sizes 256 512 1024.",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=250,
        help="Number of random validation batches for each point (default: 250).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per eval step (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="AMP dtype when using CUDA (default: bfloat16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic eval sampling.",
    )
    parser.add_argument(
        "--output_html",
        type=Path,
        default=Path("demos/ckpt_block_size_eval_report.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--fail_fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, stop on first checkpoint evaluation error.",
    )
    parser.add_argument(
        "--dark_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Plotly dark theme for the output chart.",
    )
    return parser.parse_args()


def find_checkpoints(search_dir: Path) -> List[Path]:
    ckpts = sorted(p for p in search_dir.rglob("ckpt.pt") if p.is_file())
    return ckpts


def load_validation_data(dataset: str) -> np.memmap:
    val_path = REPO_ROOT / "data" / dataset / "val.bin"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data file not found: {val_path}")
    return np.memmap(val_path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Dataset is too short for block_size={block_size}. len(data)={len(data)}"
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def calculate_validation_loss(
    model: GPT,
    val_data: np.memmap,
    block_size: int,
    eval_iters: int,
    batch_size: int,
    device: str,
    ptdtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    device_type = "cuda" if "cuda" in device else "cpu"

    start = time.perf_counter()
    amp_context = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else torch.no_grad()
    )

    with torch.no_grad():
        with amp_context:
            for _ in range(eval_iters):
                x, y = get_batch(val_data, block_size, batch_size, device)
                _, loss = model(x, y)
                losses.append(float(loss.item()))

    elapsed = time.perf_counter() - start
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    std_loss = float(np.std(losses)) if losses else float("nan")
    return {
        "val": mean_loss,
        "val_std": std_loss,
        "eval_iters": float(eval_iters),
        "elapsed_time_s": elapsed,
    }


def load_model_from_ckpt(ckpt_path: Path, device: str) -> GPT:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args")
    if model_args is None:
        raise KeyError(f"Checkpoint missing model_args: {ckpt_path}")

    model_args["dropout"] = 0.0
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_ckpt_over_block_sizes(
    ckpt_path: Path,
    block_sizes: Iterable[int],
    val_data: np.memmap,
    eval_iters: int,
    batch_size: int,
    device: str,
    ptdtype: torch.dtype,
) -> List[Dict[str, float]]:
    model = load_model_from_ckpt(ckpt_path, device)
    base_block_size = int(model.config.block_size)

    results: List[Dict[str, float]] = []
    for requested_block_size in block_sizes:
        if requested_block_size <= 0:
            raise ValueError(f"Invalid block size: {requested_block_size}")

        if requested_block_size > model.config.block_size:
            model.update_block_size(requested_block_size)
            # Some position-embedding variants create new modules on CPU in
            # update_block_size; ensure all parameters stay on the requested device.
            model.to(device)

        metrics = calculate_validation_loss(
            model=model,
            val_data=val_data,
            block_size=requested_block_size,
            eval_iters=eval_iters,
            batch_size=batch_size,
            device=device,
            ptdtype=ptdtype,
        )
        results.append(
            {
                "block_size": float(requested_block_size),
                "val_loss": float(metrics["val"]),
                "val_std": float(metrics["val_std"]),
                "elapsed_time_s": float(metrics["elapsed_time_s"]),
                "base_block_size": float(base_block_size),
            }
        )

    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    return results


def build_plotly_report(
    results_by_ckpt: Dict[str, List[Dict[str, float]]],
    dataset: str,
    output_html: Path,
    dark_mode: bool,
) -> None:
    fig = make_subplots(rows=1, cols=1)

    for ckpt_label, rows_data in results_by_ckpt.items():
        x = [r["block_size"] for r in rows_data]
        y = [r["val_loss"] for r in rows_data]
        err = [r["val_std"] for r in rows_data]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=ckpt_label,
                showlegend=True,
                error_y=dict(type="data", array=err, visible=True),
                hovertemplate=(
                    "ckpt=%{text}<br>"
                    "block_size=%{x}<br>"
                    "val_loss=%{y:.6f}<extra></extra>"
                ),
                text=[ckpt_label] * len(x),
            ),
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Validation loss", row=1, col=1)
    fig.update_xaxes(title_text="Block size", row=1, col=1)

    fig.update_layout(
        title=f"Validation Loss vs Block Size (dataset={dataset})",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=700,
        legend_title_text="Checkpoint",
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    ptdtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    ckpts = find_checkpoints(args.search_dir)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt.pt files found under: {args.search_dir}")

    block_sizes = list(dict.fromkeys(args.block_sizes))
    val_data = load_validation_data(args.eval_dataset)

    results_by_ckpt: Dict[str, List[Dict[str, float]]] = {}
    failed_ckpts: List[Tuple[str, str]] = []

    print(f"Found {len(ckpts)} checkpoint(s).")
    search_dir_resolved = args.search_dir.resolve()
    for ckpt_path in ckpts:
        ckpt_abs = ckpt_path.resolve()
        try:
            ckpt_label = str(ckpt_abs.relative_to(search_dir_resolved))
        except ValueError:
            ckpt_label = str(ckpt_abs)
        print(f"Evaluating: {ckpt_label}")
        try:
            ckpt_results = evaluate_ckpt_over_block_sizes(
                ckpt_path=ckpt_path,
                block_sizes=block_sizes,
                val_data=val_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                device=args.device,
                ptdtype=ptdtype,
            )
            results_by_ckpt[ckpt_label] = ckpt_results
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failed_ckpts.append((ckpt_label, err))
            print(f"[ERROR] {ckpt_label}: {err}")
            if args.fail_fast:
                raise
            traceback.print_exc()
            continue

    if not results_by_ckpt:
        raise RuntimeError(
            "All checkpoint evaluations failed. "
            "Rerun with --fail_fast for immediate debugging."
        )

    build_plotly_report(
        results_by_ckpt,
        args.eval_dataset,
        args.output_html,
        dark_mode=args.dark_mode,
    )

    print(f"Wrote Plotly report: {args.output_html}")
    if failed_ckpts:
        print(f"Completed with {len(failed_ckpts)} failed checkpoint(s):")
        for ckpt_label, err in failed_ckpts:
            print(f"  - {ckpt_label}: {err}")


if __name__ == "__main__":
    main()


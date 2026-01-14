# sample.py
import argparse
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

# from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tiktoken
from collections import OrderedDict
from rich import print
from rich.console import Console
from rich.table import Table
from rich.text import Text
from torch.nn import functional as F

from model import GPT, GPTConfig
from utils.model_info import print_summary, print_module_structure, print_model_blocks
from variations.model_variations import model_variation_dictionary

import lm_eval
from benchmarks.gpt_lm_eval_wrapper import NanoGPTLM
from benchmarks import run_all

def parse_args():
    parser = argparse.ArgumentParser(description="Inference from trained models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to load checkpoint from")
    parser.add_argument("--quantization_data_file", type=str, default=None, help="File name to export the quantized weights/activations, scale factor, and zero point")
    parser.add_argument("--init_from", type=str, default="resume", help="Either 'resume' (from an out_dir) or a GPT-2 variant (e.g., 'gpt2-xl')")
    parser.add_argument("--start", type=str, default="\n", help="Start text for generation. Can specify a file using 'FILE:prompt.txt'")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of inference streams to draw")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of tokens to generate in each sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for predictions (1.0 = no change, < 1.0 = less random, > 1.0 = more random)")
    parser.add_argument("--top_k", type=int, nargs='+', default=[1, 200], help="Retain only the top_k most likely tokens")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for pseudorandom number generator")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch data type for inference")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help="Compile the model (requires PyTorch 2.0)")
    parser.add_argument('--sample_file', type=str, default=None, help="Output file for inference")
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, help="Enable interactive generation")
    parser.add_argument('--stop_strings', nargs='+', type=str, default=['~W'], help="One or more strings to stop generation and allow user input. ""E.g. --stop_strings \"\n\n\" \".\"")
    parser.add_argument('--last_k_tokens', type=int, default=10, help="Number of last tokens to display in heatmaps")
    parser.add_argument('--chart_type', type=str, default='heatmap', choices=['heatmap', 'barchart'], help="Type of chart to display: 'heatmap' or 'barchart'")
    parser.add_argument('--block_size', type=int, default=None, help="Block size for context length, default is model's block size")
    parser.add_argument('--sym_rot_num_angles', type=int, default=None, help="Number of angles for symmetrical rotary embedding")
    parser.add_argument('--rope_length', type=int, default=None, help="Number of embeddings to rotate (must be an even number <= total embedding size)")
    parser.add_argument('--token_boundary', type=str, default=None, help="optional separator between emitted tokens")
    parser.add_argument('--print_model_info', default=True, action=argparse.BooleanOptionalAction, help="print info about model before infernece")

    parser.add_argument(
        '--cosine_penalty',
        type=float,
        nargs='*',
        default=None,
        help="Apply a penalty to logits based on cosine similarity to recent tokens. "
            "Use alone for defaults (N=5, alpha=1.0). "
             "Optionally provide lookback window N and penalty strength alpha. Ex: --cosine_penalty 5 1.5"
    )


    # Output Confidence
    parser.add_argument('--colorize_mode', type=str, default='minmax',
                        choices=['minmax', 'softmax', 'softmax_top_k', 'rank', 'dot_product', 'topk', 'all'],
                        help="Mode to colorize text: 'minmax' (default), 'softmax', 'softmax_top_k' for softmax over top-k values,"
                             " 'rank', 'dot_product', or 'topk' to display a prediction table. "
                             "Requires --colorize_output (enabled by default).")
    parser.add_argument('--colorize_topk', type=int, default=10,
                        help="Number of top predictions to display when colorize_mode='topk'")
    parser.add_argument('--colorize_output', default=False, action=argparse.BooleanOptionalAction,
                    help="Colorize tokens based on their predicted probabilities. Default = True. "
                    "Disable with --no-colorize-output.")

    # Visualizations
    parser.add_argument('--show_heatmaps', default=False, action=argparse.BooleanOptionalAction, help="Show heatmaps of top-k choices for each token")
    parser.add_argument('--show_minmax_chart', default=False, action=argparse.BooleanOptionalAction, help="Output a line chart of the chosen-token logits used for minmax colorization")
    parser.add_argument(
        '--softmax_threshold',
        type=float,
        nargs='?',
        const=0.5, # default value if flag is present without a value
        default=None, # default value if flag is not present
        help="Enable softmax threshold sampling. Only considers tokens with a probability within this percentage of the top probability. "
             "Use without a value for default 50%% (0.5), or provide one e.g. '--softmax_threshold 0.2'. Overrides --top_k.")




    # Steering Vector Related
    parser.add_argument('--save_avg_vector', type=str, default=None, help="Path to save the average vector of the start text to an .npy file")
    parser.add_argument('--apply_vector_file1', type=str, default=None, help="First .npy file to load the vector for subtraction")
    parser.add_argument('--apply_vector_file2', type=str, default=None, help="Second .npy file to load the vector for subtraction")
    parser.add_argument('--steering_vector_scaling_factor', type=float, default=1.0, help="Scaling factor to apply after subtracting vectors")
    parser.add_argument('--apply_to_layer_idx', type=int, default=None, help="Layer index at which to apply the resulting vector")

    # Leanred Steering Vector Related
    parser.add_argument('--use_lsv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lsv_size',  type=int, default=1, help="Number of vectors to test")
    parser.add_argument('--lsv_scaling_factor',  type=float, default=None, help="scaling factor")
    parser.add_argument('--lsv_mixture',  type=float, nargs='+', default=None, help="scaling factor mixture")

    # Multicontext Related
    parser.add_argument('--multicontext', action=argparse.BooleanOptionalAction, help="multicontext mode inference")
    parser.add_argument('--multicontext_datasets',  type=str, nargs='+', default=None, help="list of dataset names")
    parser.add_argument('--multicontext_start', type=str, nargs='+', default=None,
                        help="List of start strings, one for each context, if using --multicontext. "
                        "Must match the number/order of --multicontext_datasets.")

    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction, help="Enable evaluation only mode to calculate and print validation loss")
    parser.add_argument("--eval_iters", type=int, default=250, help="iterations for evaluation")
    parser.add_argument("--eval_dataset", type=str, default=None, help="dataset for evaluation")

    # lm_eval Benchmarking Related
    parser.add_argument('--lm_eval_tasks', type=str, default=None,
                    help="Comma-separated list of tasks for lm-eval (e.g. 'arc_easy,hellaswag')")
    parser.add_argument(
        '--lm_eval_results_output',
        type=str,
        default=None,
        help="Where to save the lm-eval results (JSON). "
             "If not set, defaults to out_dir/<timestamp>_lm_eval_results.json"
    )
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size to use for evaluation")

    return parser.parse_args()



def convert_rich_renderable_to_ansi(renderable) -> str:
    """Convert any Rich renderable (Text, Table, etc.) into an ANSI string."""
    buffer = io.StringIO()
    temp_console = Console(
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
    )
    temp_console.print(renderable)
    return buffer.getvalue()

def append_to_sample_file(sample_file, output_line, start_token, k_tag, iter_num=None, best_val_loss=None, run_name=None):
    to_print = {
        "run_name":   run_name,
        "iter_num":   iter_num,
        "best_val_loss": best_val_loss,
        "top_k": k_tag,
    }
    with open(sample_file, "a", encoding="utf-8", errors="replace") as file:
        header = '\n---------------'

        # Print remaining available statistics
        for name, value in to_print.items():
            if value is not None:
                header += f"\n {name}: {value} \n"

        # Handle start token as special case due to special chars
        if start_token is not None:
            header += f"\n start_token: {repr(start_token)} \n"

        header += '---------------\n'

        # If it's a Rich renderable, convert it to an ANSI string
        if not isinstance(output_line, str):
            output_line = convert_rich_renderable_to_ansi(output_line)

        file.write(header + output_line + '\n\n')

def colorize_text(tokens, data_for_color, decode, colorize_mode='minmax'):

    """
    Colorizes each token according to one of two modes:
      - 'minmax': data_for_color is a 1D list/array of chosen-token logits.
                  We min-max normalize them across time, then map to R->G colors.
      - 'softmax': data_for_color is a 2D list/array (T, vocab_size) containing
                   the *full* distribution at each step. We extract the chosen
                   token's probability for each step, then min-max normalize.
    """
    text = Text()

    norm_values = None

    if colorize_mode == 'softmax' or colorize_mode == 'softmax_top_k':
        # data_for_color is shape (T, vocab_size) per step
        # gather the chosen token's probability each step
        # then apply min–max to those probabilities
        dist_tensor = torch.stack(data_for_color, dim=0)  # shape (T, vocab_size)

        chosen_probs = []
        for i, dist_row in enumerate(dist_tensor):
            # print(dist_row)
            prob_dist = F.softmax(dist_row, dim=-1)
            # print(prob_dist)
            # input()
            chosen_probs.append(prob_dist[tokens[i]])
        values = torch.stack(chosen_probs)

        norm_values = values

    if colorize_mode == 'minmax' or colorize_mode == 'dot_product':
        # data_for_color is shape (T,) with each chosen-token score (logit or dot product)
        values = torch.tensor(data_for_color, dtype=torch.float32)

        # Normalize the chosen values (probabilities or logits) to [0..1]
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-6)

    for i, token_id in enumerate(tokens):
        token_str = decode([token_id])
        color_val = norm_values[i].item()  # 0..1
        r = int((1 - color_val) * 255)
        g = int(color_val * 255)
        text.append(token_str, style=f"bold #{r:02x}{g:02x}00")
    return text


def _escape_ws(text: str) -> str:
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _topk_table(
    token_ids: List[int],
    rows: List[torch.Tensor],
    decode: Callable[[Sequence[int]], str],
    k: int,
    max_token_chars: int = 20,
    escape_ws: bool = True,
) -> Table:
    """Return a Rich table showing top-k predictions for each token."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("target", no_wrap=True)
    table.add_column("xent", justify="right", no_wrap=True)
    table.add_column("rank", justify="right", no_wrap=True)
    table.add_column("p_tgt", justify="right", no_wrap=True)
    table.add_column("p_left", justify="right", no_wrap=True)
    for _ in range(k):
        table.add_column(justify="center", no_wrap=True)

    for tid, row in zip(token_ids, rows):
        probs = F.softmax(row, dim=-1)
        tgt_prob = probs[tid].item()
        rank = int((row > row[tid]).sum().item()) + 1
        prob_left = probs[row > row[tid]].sum().item()
        ce = -math.log(tgt_prob + 1e-12)

        topv, topi = row.topk(k)
        norm = (topv - topv.min()) / (topv.max() - topv.min() + 1e-6)
        words: List[Text] = []
        for idx, v in zip(topi.tolist(), norm.tolist()):
            r = int((1 - v) * 255); g = int(v * 255)
            style = f"#{r:02x}{g:02x}00"
            token = decode([idx])
            if max_token_chars >= 0:
                token = token[:max_token_chars]
            if escape_ws:
                token = _escape_ws(token)
            if idx == tid:
                words.append(Text(token, style="bold cyan"))
            else:
                words.append(Text(token, style=style))

        rank_norm = 1 - (min(rank, 100) - 1) / 99
        r = int((1 - rank_norm) * 255); g = int(rank_norm * 255)
        rank_text = Text(str(rank), style=f"bold #{r:02x}{g:02x}00")

        v = tgt_prob
        r = int((1 - v) * 255); g = int(v * 255)
        p_tgt_text = Text(f"{tgt_prob:.4f}", style=f"bold #{r:02x}{g:02x}00")

        v = 1 - prob_left
        r = int((1 - v) * 255); g = int(v * 255)
        p_left_text = Text(f"{prob_left:.4f}", style=f"bold #{r:02x}{g:02x}00")

        target_word = decode([tid])
        if max_token_chars >= 0:
            target_word = target_word[:max_token_chars]
        if escape_ws:
            target_word = _escape_ws(target_word)

        table.add_row(Text(target_word, style="bold cyan"), f"{ce:.4f}", rank_text, p_tgt_text, p_left_text, *words)

    return table

def save_chart(probs, idx, decode, step, out_dir, last_k_tokens, chart_type, selected_token, top_k_value, args):
    """
    Generates and saves a chart of token probabilities for a single generation step.

    This function adapts its visualization based on the sampling method specified in `args`.
    - If `softmax_threshold` is used, it visualizes the actual pool of candidate tokens.
    - If `top_k` is used, it visualizes the top `k` most likely tokens.

    Args:
        probs (torch.Tensor): The final probability distribution tensor (shape: 1, vocab_size)
                              used for sampling the next token.
        idx (torch.Tensor): The tensor of all generated token IDs so far.
        decode (function): A function to decode a list of token IDs into a string.
        step (int): The current generation step number.
        out_dir (str): The base output directory to save charts into.
        last_k_tokens (int): The number of recent tokens to show in the chart's context label.
        chart_type (str): The type of chart to generate ('heatmap' or 'barchart').
        selected_token (str): The string representation of the token that was actually chosen.
        top_k_value (int or None): The `k` value used for top-k sampling.
        args (argparse.Namespace): The command-line arguments, used to check the sampling mode.
    """
    # --- 1. Determine Visualization Parameters based on Sampling Mode ---
    vocab_size = probs.size(-1)
    chart_title = ""
    num_candidates = 0

    if args.softmax_threshold is not None:
        # Mode: Softmax Threshold Sampling
        # Visualize the actual candidate pool (tokens with non-zero probability).
        num_candidates = torch.count_nonzero(probs).item()
        # Cap the number of plotted tokens for readability.
        k_to_plot = min(num_candidates, 60)
        chart_title = f"Top {k_to_plot} of {num_candidates} Candidates (Softmax Threshold)"
    else:
        # Mode: Top-K or No Sampling Truncation
        # Use the provided top_k_value to determine how many tokens to show.
        k_to_plot = top_k_value
        if k_to_plot is None:
            # If no top_k was specified, use a reasonable default for visualization.
            k_to_plot = 40
        k_to_plot = min(k_to_plot, vocab_size)
        chart_title = f"Top-{k_to_plot} Probabilities (Top-K Setting: {top_k_value})"

    # --- 2. Prepare Data for Plotting ---
    # Get the top k probabilities and their corresponding indices from the final distribution.
    # This works for both modes because we want to see the most likely candidates.
    top_probs, top_indices = torch.topk(probs.flatten(), k=k_to_plot)
    top_tokens = [decode([i.item()]) for i in top_indices]

    # --- 3. Generate and Save the Chart ---
    plt.figure(figsize=(16, 9))

    if chart_type == 'heatmap':
        annot_data = np.array(top_tokens).reshape(1, -1)
        sns.heatmap(
            top_probs.cpu().numpy().reshape(1, -1),
            annot=annot_data,
            fmt='',
            cmap='viridis',
            cbar_kws={'label': 'Probability'}
        )
        plt.yticks([])  # Hide y-axis ticks as they are not meaningful here.
        plt.title(f"Step {step}: {chart_title} (Heatmap)")

    elif chart_type == 'barchart':
        colors = sns.color_palette('viridis', n_colors=k_to_plot)
        bars = plt.bar(top_tokens, top_probs.cpu().numpy(), color=colors)
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.0)  # Ensure a consistent y-axis scale for probabilities.
        plt.xticks(rotation=45, ha="right")  # Prevent x-axis label overlap.

        # Highlight the bar for the token that was actually selected.
        try:
            selected_token_index = top_tokens.index(selected_token)
            bars[selected_token_index].set_edgecolor('red')
            bars[selected_token_index].set_linewidth(2)
        except ValueError:
            # This can happen if the selected token is not in the top k_to_plot,
            # which is unlikely but possible with unusual settings.
            print(f"Note: Selected token '{selected_token}' not in top {k_to_plot} for visualization at step {step}.")
        plt.title(f"Step {step}: {chart_title} (Bar Chart)")

    # Add a descriptive x-axis label showing the recent generation context.
    last_tokens_decoded = decode(idx[0, -last_k_tokens:].tolist())
    plt.xlabel(f"Token Candidates (Context: ...{last_tokens_decoded})")

    # --- 4. Save to File ---
    # Ensure the 'charts' subdirectory exists.
    charts_dir = os.path.join(out_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Use a high-resolution timestamp to prevent filename collisions.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(charts_dir, f"step_{step}_{timestamp}.png")

    plt.tight_layout()  # Adjust layout to prevent labels from being cut off.
    plt.savefig(out_path)
    plt.close()  # Close the plot to free up memory.


def save_raw_logits_chart(raw_logit_values, out_dir, k_tag, sample_idx):
    """
    Generates and saves a line chart of the raw, pre-temperature chosen-token logits over time.
    """
    # Ensure there's data to plot
    if not raw_logit_values:
        return

    # Convert list of single-item tensors to a numpy array
    logits_np = torch.tensor(raw_logit_values).cpu().numpy()

    steps = np.arange(len(logits_np))

    plt.figure(figsize=(16, 9))

    plt.plot(steps, logits_np, marker='o', linestyle='-', label=f'Sample {sample_idx+1}, K-Setting: {k_tag}')

    # The Y-axis is automatically scaled by matplotlib to the min and max of the data
    plt.ylabel("Raw Model Logit (Pre-Temperature)")
    plt.xlabel("Generation Step")
    plt.title(f"Raw Chosen-Token Model Logits Over Time")
    plt.grid(True)
    plt.legend()

    # Save the figure to the 'charts' subdirectory
    charts_dir = os.path.join(out_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(charts_dir, f"raw_logits_k{k_tag}_sample{sample_idx+1}_{timestamp}.png")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _colorize_rank(
    token_ids: List[int],
    ranks: List[int],
    decode: Callable[[Sequence[int]], str],
    k: Optional[int],
) -> Text:
    """
    Return a Rich Text object whose colours encode rank:

    • rank == 1  → no colour (default terminal fg)
    • rank  2..k → gradient green -> yellow -> red
    • rank > k   → no colour
    """
    text = Text()
    max_rank = max(k or 0, 2)      # guarantees divisor ≥ 1

    for tid, rnk in zip(token_ids, ranks):
        token_str = decode([tid])

        if rnk == 1:
            # best-rank token: leave unstyled
            text.append(token_str)
        elif 2 <= rnk <= max_rank:
            ratio = (rnk - 2) / (max_rank - 2) if max_rank > 2 else 1.0
            r = int(255 * ratio)          # 0 → green, 1 → red
            g = int(255 * (1 - ratio))
            # style string identical to your colorize_text template
            text.append(token_str, style=f"bold #{r:02x}{g:02x}00")
        else:
            text.append(token_str)        # ranks outside 1..k

    return text


def sample_with_existing_model(
    model: torch.nn.Module,
    start_ids: torch.Tensor,
    decode: Callable[[Sequence[int]], str],
    device: str = "cuda",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Union[int, Sequence[int], None] = 200,  # list allowed
    start_tokens: Optional[Sequence[int]] = None,
    num_samples: int = 1,
    # Additional Args
    args=None,
    # ── visual / logging flags ────────────────────────────────────────────
    colorize_output: bool = False,
    colorize_mode: str = "minmax",         # "rank", "topk" & "all" supported
    token_boundary: Optional[str] = None,
    show_heatmaps: bool = False,
    chart_type: str = "heatmap",
    last_k_tokens: int = 10,
    out_dir: Union[str, Path] = "out",
    sample_file: Optional[str] = None,
    iter_num: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    run_name: Optional[str] = None,
    writer: Optional[object] = None,
    dataset_idx: Optional[int] = None,
    console: Console | None = None,
):
    """
    Generate text from an already-loaded GPT model.

    Parameters
    ----------
    top_k : int | list[int] | None
        • int   – sample from top-k.
        • None  – no truncation.
        • list  – run once per k in the list (duplicates filtered).
    colorize_mode :
        "minmax" | "softmax" | "softmax_top_k" | "dot_product" | "rank" | "topk" | "all"
    writer : torch.utils.tensorboard.SummaryWriter | None
        When provided, dataset metrics for each top-k sample will be logged to TensorBoard.
    """

    console = console or Console()

    # Determine sampling strategy. Softmax threshold overrides top_k.
    if args.softmax_threshold is not None:
        console.print(f"[yellow]Info:[/yellow] Using softmax threshold sampling ({args.softmax_threshold:.2f}). --top_k will be ignored.")
        # Force the loop to run once with a null k-value
        k_values: List[Optional[int]] = [None]
    else:
    # Use the standard top_k logic
        if top_k is None or isinstance(top_k, int):
            k_values: List[Optional[int]] = [top_k]
        else:
            k_values = list(dict.fromkeys(top_k)) # Deduplicate

    console = Console()
    model.eval()

    valid_modes = ["minmax", "softmax", "softmax_top_k", "dot_product", "rank", "topk"]
    modes_to_apply = valid_modes if colorize_mode == "all" else [colorize_mode]


    for current_k in k_values:
        # Set a tag for logging/filenames based on the active sampling mode
        if args.softmax_threshold is not None:
            k_tag = f"sm_thresh_{args.softmax_threshold:.2f}"
        else:
            k_tag = "no_topk" if current_k is None else f"top_k_{current_k}"


        for sample_idx in range(num_samples):
            # ------------- LSV per-sample section -------------------
            kl_divergences = [] # To store the impact of the cosine penalty

            if args is not None:
                # This block handles LSV for standalone sampling. When called from train.py,
                # lsv_size is not an arg, so we skip this to avoid an AttributeError and
                # to respect the index already set by the trainer.
                if args.use_lsv and hasattr(args, 'lsv_size'):
                    model.set_lsv_index(sample_idx % args.lsv_size)
                    if args.lsv_scaling_factor is not None:
                        model.set_lsv_scaling_factor(args.lsv_scaling_factor)
                    if args.lsv_mixture is not None:
                        model.set_lsv_mode(2)
                        model.set_lsv_mixture(args.lsv_mixture)
                    else:
                        model.set_lsv_mode(1)


                    console.print(f"[green]LSV[/green]  idx={sample_idx % args.lsv_size} "
                          f"scale={args.lsv_scaling_factor} "
                          f"mixture={args.lsv_mixture}")
            # ------------- END LSV per-sample section -------------------

            x = start_ids.clone()

            # storage for colouring
            tokens_for_color: List[int] = []
            full_rows: List[torch.Tensor] = []
            topk_rows: List[torch.Tensor] = []
            pre_temp_scalar_rows: List[torch.Tensor] = []
            scalar_rows: List[torch.Tensor] = []
            ranks_list: List[int] = []  # NEW

            with torch.no_grad():
                for _step in range(max_new_tokens):
                    idx_cond = (
                        x
                        if x.size(1) <= model.config.block_size
                        else x[:, -model.config.block_size :]
                    )

                    model_logits, _ = model(idx_cond, dataset_idx=dataset_idx)
                    raw_logits_row = model_logits[:, -1, :]      # Raw logits from model

                    # --- Apply Cosine Similarity Penalty (if enabled) ---
                    if args.cosine_penalty is not None:
                        N = 5 if len(args.cosine_penalty) < 1 else int(args.cosine_penalty[0])
                        alpha = 1.0 if len(args.cosine_penalty) < 2 else args.cosine_penalty[1]

                        # Calculate original probabilities for comparison
                        probs_before = F.softmax(raw_logits_row / temperature, dim=-1)


                        # Apply penalty as long as there are tokens in the context and N > 0
                        if x.size(1) > 0 and N > 0:
                            # Python's negative slicing gracefully handles cases where x.size(1) < N
                            last_n_tokens = x[0, -N:]

                            embedding_matrix = model.transformer.wte.weight

                            # Normalize embeddings
                            last_n_embeds = F.normalize(embedding_matrix[last_n_tokens], p=2, dim=1)
                            all_embeds = F.normalize(embedding_matrix, p=2, dim=1)

                            # Calculate max cosine similarity for each candidate against the last N tokens
                            sim_matrix = torch.matmul(all_embeds, last_n_embeds.T)
                            max_sim_per_candidate, _ = torch.max(sim_matrix, dim=1)
                            penalty = alpha * max_sim_per_candidate
                            raw_logits_row = raw_logits_row - penalty

                            # Calculate KL divergence to measure the change
                            probs_after = F.softmax(raw_logits_row / temperature, dim=-1)
                            # Add a small epsilon to avoid log(0)
                            kl_div = F.kl_div(torch.log(probs_after + 1e-9), probs_before, reduction='sum')
                            kl_divergences.append(kl_div.item())


                    logits = raw_logits_row / temperature        # Scaled logits for sampling
                    full_row = logits[0].clone()               # pre-mask


                    # Apply the selected truncation logic
                    if args.softmax_threshold is not None:
                        # Calculate probabilities and find the threshold
                        probs = F.softmax(logits, dim=-1)
                        max_prob = torch.max(probs)
                        prob_threshold = max_prob * args.softmax_threshold
                        # Set probabilities of tokens below the threshold to 0
                        probs[probs < prob_threshold] = 0


                    topk_row = logits[0].clone()               # post-mask

                    if args.softmax_threshold is not None:
                        # Calculate probabilities and find the threshold
                        probs = F.softmax(logits, dim=-1)
                        max_prob = torch.max(probs)
                        prob_threshold = max_prob * args.softmax_threshold
                        # Set probabilities of tokens below the threshold to 0
                        probs[probs < prob_threshold] = 0
                        # Sample from the modified, unnormalized distribution of probabilities
                        idx_next = torch.multinomial(probs, num_samples=1)
                        # For colorization, we can still use the unmasked logits
                        topk_row = logits[0].clone()
                    elif current_k is not None:
                        v, _ = torch.topk(logits, min(current_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("inf")
                        topk_row = logits[0].clone()               # post-mask
                        probs = F.softmax(logits, dim=-1) # Re-softmax after masking
                        idx_next = torch.multinomial(probs, num_samples=1)
                    else: # No truncation / default case
                        topk_row = logits[0].clone()
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)

                    x = torch.cat((x, idx_next), dim=1)

                    if colorize_output:
                        chosen = idx_next.item()
                        # rank: 1 = best
                        rank = (full_row > full_row[chosen]).sum().item() + 1

                        tokens_for_color.append(chosen)
                        full_rows.append(full_row)
                        topk_rows.append(topk_row)
                        scalar_rows.append(full_row[chosen])
                        if args.show_minmax_chart:
                            pre_temp_scalar_rows.append(raw_logits_row[0, chosen])
                        ranks_list.append(rank)

                    if show_heatmaps:
                        sel_txt = decode([idx_next.item()])
                        save_chart(                             # type: ignore
                            probs,
                            x,
                            decode,
                            _step,
                            out_dir,
                            last_k_tokens,
                            chart_type,
                            sel_txt,
                            current_k,
                            args,

                        )

            # ---------- Print summary statistics for this sample ------------------
            if kl_divergences:
                avg_kl = np.mean(kl_divergences)
                console.print(f"\n[bold yellow]Cosine Penalty Impact (Avg KL Divergence):[/bold yellow] [bold cyan]{avg_kl:.4f}[/bold cyan]")

            # ---------- save minmax chart if requested ----------------------
            if args.show_minmax_chart and pre_temp_scalar_rows:
                save_raw_logits_chart(
                    pre_temp_scalar_rows, out_dir, k_tag, sample_idx
                 )


            # ---------- decode plain text -----------------------------------
            plain_text = decode(x[0].tolist())
            if token_boundary is not None:
                plain_text = plain_text.replace(token_boundary, " ")

            if args and getattr(args, "sample_metrics", False):
                metrics = run_all(plain_text)
                metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                console.print(
                    f"\n[bold magenta]Metrics ({k_tag}, sample {sample_idx+1}):[/bold magenta] {metric_str}"
                )
                if writer is not None and getattr(args, "tensorboard_log", False):
                    for mk, mv in metrics.items():
                        # group top-k runs on a single chart per metric
                        writer.add_scalars(f"sample_metrics/{mk}", {k_tag: mv}, iter_num or 0)

            # ---------- colourised outputs ----------------------------------
            if colorize_output:
                # --- Pre-calculate any special data sources for colorization ---
                dot_product_values = None
                if 'dot_product' in modes_to_apply and len(tokens_for_color) > 1:
                    dot_product_values = [0.0] # First token has no prior, assign neutral value.
                    embedding_matrix = model.transformer.wte.weight
                    for i in range(1, len(tokens_for_color)):
                        prev_vec = F.normalize(embedding_matrix[tokens_for_color[i-1]], p=2, dim=0)
                        current_vec = F.normalize(embedding_matrix[tokens_for_color[i]], p=2, dim=0)
                        # The dot product of two unit vectors is their cosine similarity.
                        dot_product_values.append(torch.dot(prev_vec, current_vec).item())

                for cm in modes_to_apply:
                    # Select the appropriate data source for the current colorization mode
                    data_for_color = None
                    if cm == "minmax":
                        data_for_color = scalar_rows
                    elif cm == "softmax":
                        data_for_color = full_rows
                    elif cm == "softmax_top_k":
                        data_for_color = topk_rows
                    elif cm == "dot_product":
                        data_for_color = dot_product_values

                    if data_for_color is not None:
                        coloured = colorize_text(              # type: ignore
                            tokens_for_color,
                            data_for_color,
                            decode,
                            colorize_mode=cm,
                        )
                    elif cm == "rank":
                        coloured = _colorize_rank(
                            tokens_for_color, ranks_list, decode, current_k
                        )
                    elif cm == "topk":
                        coloured = _topk_table(
                            tokens_for_color, full_rows, decode, args.colorize_topk
                        )
                    else:
                        continue  # Should not happen if data_for_color is None


                    fgcolor="bold light_slate_blue"
                    bgcolor="bold cyan"
                    console.print(f"\n\n[{bgcolor}]--- tokens=[/{bgcolor}][{fgcolor}]{max_new_tokens}[/{fgcolor}][{bgcolor}], top_k=[/{bgcolor}][{fgcolor}]{k_tag}[/{fgcolor}][{bgcolor}], colorization=[/{bgcolor}][{fgcolor}]{cm}[/{fgcolor}][{bgcolor}] ---[/{bgcolor}]\n")
                    console.print(coloured)

                    if sample_file:
                        append_to_sample_file(                 # type: ignore
                            sample_file,
                            coloured,
                            start_tokens,
                            k_tag,
                            iter_num,
                            best_val_loss,
                            f"{run_name}_{k_tag}_{cm}" if run_name else f"{k_tag}_{cm}",
                        )
            else:
                console.print(f"[bold cyan]--- {k_tag} ---[/bold cyan]")
                console.print("[bold green]" + plain_text + "[/bold green]")

            # ---------- always store plain text once ------------------------
            if sample_file:
                append_to_sample_file(                         # type: ignore
                    sample_file,
                    plain_text,
                    start_tokens,
                    k_tag,
                    iter_num,
                    best_val_loss,
                    f"{run_name}_{k_tag}" if run_name else k_tag,
                )


def interactive_generation(model, start_ids, device, max_new_tokens, temperature, top_k, stop_string, decode, encode):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    while True:
        x, generated_text = model.generate_with_stop(x, max_new_tokens, stop_string, decode, temperature, top_k)
        print("[bold green]" + generated_text)

        user_input = input("User input (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Append the user input directly after the stop string
        x = torch.cat((x, torch.tensor(encode(user_input), dtype=torch.long, device=device)[None, ...]), dim=1)


def save_args(args, out_dir):
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


def write_eval_summary(
    out_dir: Union[str, os.PathLike[str], None],
    summary: Dict[str, object],
    *,
    extra_dirs: Optional[Sequence[Union[str, os.PathLike[str]]]] = None,
) -> None:
    if not summary:
        return

    def _convert(value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                return float(value.item())
            return value.cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    destinations: List[str] = []
    if out_dir:
        destinations.append(os.fspath(out_dir))
    if extra_dirs:
        for path in extra_dirs:
            if path:
                destinations.append(os.fspath(path))

    if not destinations:
        return

    serializable = {
        key: _convert(value)
        for key, value in summary.items()
        if value is not None
    }

    saved_paths: List[str] = []
    seen_dirs: set[str] = set()
    for directory in destinations:
        normalized = os.path.normpath(directory)
        if normalized in seen_dirs:
            continue
        seen_dirs.add(normalized)
        os.makedirs(normalized, exist_ok=True)
        eval_path = os.path.join(normalized, "eval_loss.txt")
        with open(eval_path, "w", encoding="utf-8") as eval_file:
            json.dump(serializable, eval_file, indent=2, sort_keys=True)
            eval_file.write("\n")
        saved_paths.append(eval_path)

    if saved_paths:
        if len(saved_paths) == 1:
            print(f"Saved evaluation metrics to {saved_paths[0]}")
        else:
            print("Saved evaluation metrics to:")
            for path in saved_paths:
                print(f"  {path}")


#TODO: Rename to reflect general purpose
def save_quantized_data(state_dict, out_file):
    to_save = OrderedDict()
    for k, v in list(state_dict.items()):
        # if "mlp_act" in k or "attn_act" in k or k.endswith("quantized_bias") or k.endswith("bias_norm") or k.endswith("zero_point") or k.endswith("quantized_weight") or k.endswith("weight_norm"):
        to_save[k] = v.cpu().numpy()

    with open(f"{out_file}.pkl", 'wb') as f:
        pickle.dump(to_save, f)

def load_validation_data(block_size, eval_dataset):
    # Load validation data similar to how train data is handled
    val_path = os.path.join('data', eval_dataset, 'val.bin')
    assert os.path.exists(val_path), f"Validation data file {val_path} not found."
    # Assuming validation data is similar in format to train data
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    return val_data

def get_batch(data, block_size, device):
    # Create a random batch from the dataset
    ix = torch.randint(len(data) - block_size, (1,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def calculate_validation_loss(model, val_data, block_size, eval_iters, device, dtype, dataset_idx: Optional[int] = None):
    model.eval()
    losses: List[float] = []
    total_time = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            X, Y = get_batch(val_data,  block_size, device)
            with torch.amp.autocast(device_type=device, dtype=dtype):
                start = time.perf_counter()
                logits, loss = model(X, Y, dataset_idx=dataset_idx)
                end = time.perf_counter()
                total_time += (end - start)
            losses.append(float(loss.item()))

    if losses:
        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses)) if len(losses) > 1 else 0.0
    else:
        mean_loss = float("nan")
        std_loss = float("nan")

    return {
        "val": mean_loss,
        "val_std": std_loss,
        "eval_iters": int(eval_iters),
        "num_batches": len(losses),
        "elapsed_time_s": float(total_time),
    }

def custom_char_with_byte_fallback_encode(text: str, stoi: dict) -> list[int]:
    """Encode ``text`` using a byte-level vocabulary with optional custom tokens.

    This mirrors the logic in ``CustomCharTokenizerWithByteFallback``. For each
    position in the UTF-8 byte stream we try each custom token (in the order
    they were defined) and fall back to emitting the raw byte ID when none
    match.
    """

    custom_token_bytes = [
        (tok, tok.encode("utf-8"))
        for tok in stoi.keys()
        if isinstance(tok, str)
    ]

    data_bytes = text.encode("utf-8")
    i, n = 0, len(data_bytes)
    ids: list[int] = []

    while i < n:
        matched = False
        for token_str, token_bytes in custom_token_bytes:
            l = len(token_bytes)
            if data_bytes[i:i + l] == token_bytes:
                ids.append(stoi[token_str])
                i += l
                matched = True
                break
        if not matched:
            byte_token = data_bytes[i:i+1]
            ids.append(stoi[byte_token])
            i += 1

    return ids


def custom_char_with_byte_fallback_decode(ids: list[int], itos: dict) -> str:
    """Decode a list of token IDs produced by the byte-fallback tokenizer."""
    out_parts: list[str] = []
    byte_buffer: list[bytes] = []

    def flush_bytes() -> None:
        if byte_buffer:
            out_parts.append(b"".join(byte_buffer).decode("utf-8", errors="replace"))
            byte_buffer.clear()

    for tok_id in ids:
        if tok_id < 256:
            byte_buffer.append(itos[tok_id])
        else:
            flush_bytes()
            out_parts.append(itos[tok_id])

    flush_bytes()
    return "".join(out_parts)


def byte_encode(text: str) -> list[int]:
    """Encode text into raw UTF-8 byte values."""
    return list(text.encode("utf-8"))


def byte_decode(ids: list[int]) -> str:
    """Decode a list of raw byte values back into text."""
    return bytes(ids).decode("utf-8", errors="replace")

def get_tokenizer_functions(meta):
    """Get encode/decode functions based on tokenizer metadata"""
    if 'tokenizer' not in meta:
        # Default character-level tokenizer
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        return encode, decode

    if meta['tokenizer'] == "sinewave":
        def encode_fn(s: str):
            s = s.strip()
            if not s:
                return []
            return [int(v) for v in s.split(',')]

        def decode_fn(values):
            return ','.join(str(int(v)) for v in values)

        return encode_fn, decode_fn

    if meta['tokenizer'] == 'tiktoken':
        enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)
        return encode, decode

    if meta['tokenizer'] == 'byte':
        return byte_encode, byte_decode

    if meta['tokenizer'] == 'custom_char_with_byte_fallback':
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: custom_char_with_byte_fallback_encode(s, stoi)
        decode = lambda l: custom_char_with_byte_fallback_decode(l, itos)
        return encode, decode

    if meta['tokenizer'] == 'json_byte_fallback':
        stoi, itos = meta['stoi'], meta['itos']

        # Sort tokens by length in descending order for precedence
        string_token_tuples = [(token, token_id) for token, token_id in stoi.items() if isinstance(token, str)]

        def encode(text):
            ids = []
            current_pos = 0
            text_len = len(text)

            while current_pos < text_len:
                remaining_text = text[current_pos:]
                token_found = False

                # Try string tokens first, from longest to shortest
                for token, token_id in string_token_tuples:
                    if remaining_text.startswith(token):
                        ids.append(token_id)
                        current_pos += len(token)
                        token_found = True
                        break

                if not token_found:
                    # If no token matches, fall back to byte encoding
                    char = text[current_pos]
                    char_bytes = char.encode('utf-8')
                    for byte in char_bytes:
                        byte_token = bytes([byte])
                        if byte_token in stoi:
                            ids.append(stoi[byte_token])
                        else:
                            # Use UNK token if available
                            ids.append(stoi.get('<unk>', 0))
                    current_pos += 1

            return ids

        def decode(token_ids):
            tokens = []
            byte_buffer = []

            for id in token_ids:
                if id not in itos:
                    continue

                token = itos[id]

                # Handle bytes vs string tokens
                if isinstance(token, bytes):
                    byte_buffer.append(token[0])  # Append the actual byte value
                else:
                    # If we have bytes in buffer, try to decode them first
                    if byte_buffer:
                        try:
                            decoded = bytes(byte_buffer).decode('utf-8', errors='replace')
                            tokens.append(decoded)
                        except UnicodeDecodeError:
                            tokens.append('')  # Unicode replacement character
                        byte_buffer = []

                    # Handle the string token
                    token = token.replace('Ġ', ' ')  # Replace Ġ with space
                    tokens.append(token)

            # Handle any remaining bytes in the buffer
            if byte_buffer:
                try:
                    decoded = bytes(byte_buffer).decode('utf-8', errors='replace')
                    tokens.append(decoded)
                except UnicodeDecodeError:
                    tokens.append('')

            return ''.join(tokens)

        return encode, decode

    # Default fallback
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    save_args(args, out_dir)

    checkpoint = None
    checkpoint_config: Dict[str, object] = {}

    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        checkpoint_config = checkpoint.get('config', {})
        checkpoint['model_args']['dropout'] = 0.0
        if args.save_avg_vector:
            print(f"saving {args.save_avg_vector}")
            checkpoint['model_args']['obtain_vector_at_layer_idx'] = args.apply_to_layer_idx
            checkpoint['model_args']['obtain_vector_file'] = args.save_avg_vector
        # If vectors are provided, load and subtract them, then apply to a designated layer during generation
        if args.apply_vector_file1 and args.apply_vector_file2:
            vector1 = np.load(args.apply_vector_file1)
            vector2 = np.load(args.apply_vector_file2)
            diff_vector = vector1 - vector2
            torch.from_numpy(diff_vector).float().to(args.device)
            diff_vector_tensor = torch.from_numpy(diff_vector).float().to(args.device)
            diff_vector_cpu = diff_vector_tensor.cpu().numpy()  # Move the tensor to CPU and convert it to a NumPy array
            np.save("temp.npy", diff_vector_cpu)

            # Convert to tensor and set in the model for application at the designated layer
            checkpoint['model_args']['apply_vector_file']= "temp.npy"
            checkpoint['model_args']['apply_vector_at_layer_idx']= args.apply_to_layer_idx
            checkpoint['model_args']['apply_vector_scaling_factor']= args.steering_vector_scaling_factor
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        if args.quantization_data_file:
            save_quantized_data(state_dict, args.quantization_data_file)

        model.load_state_dict(state_dict, strict=False)

    else:
        # Need to create a completely "default" GPTConfig and overwrite using model_variations
        gptconf = GPTConfig()
        variation_dict = model_variation_dictionary[args.init_from]
        for k, v in variation_dict.items():
            setattr(gptconf, k, v)
        model = GPT.from_pretrained(gptconf, model_type=args.init_from)

    if args.init_from == 'resume' and args.multicontext is None:
        args.multicontext = bool(getattr(model.config, "multicontext", False))

    if (
        args.init_from == 'resume'
        and args.multicontext
        and args.multicontext_datasets is None
    ):
        datasets_from_ckpt = checkpoint_config.get('multicontext_datasets') if checkpoint_config else None
        if datasets_from_ckpt:
            args.multicontext_datasets = list(datasets_from_ckpt)

    # Load meta information if available
    load_meta = False
    meta_path = None
    separator_token = None
    if args.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_paths = [
            os.path.join(args.out_dir, 'meta.pkl'),
            os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        ]

        for meta_path in meta_paths:
            if os.path.exists(meta_path):
                load_meta = True
                break

    # For using gpt2 pretrained models
    if args.init_from.startswith('gpt2'):
        # use tiktoken for gpt2
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)

    meta = None
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            encode, decode = get_tokenizer_functions(meta)

    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r', encoding='utf-8') as f:
            args.start = f.read()

    if args.multicontext and args.multicontext_start is None and args.multicontext_datasets:
        args.multicontext_start = [args.start] * len(args.multicontext_datasets)

    start_ids = encode(args.start)
    if len(start_ids) == 0:
        if meta and meta.get('tokenizer') == 'sinewave':
            print("Start string produced no tokens for sinewave tokenizer; defaulting to '0'.")
            start_ids = [0]
        elif not args.multicontext:
            raise ValueError(
                "The provided --start text resulted in an empty context. "
                "Please supply a non-empty prompt or comma-separated values for numerical tokenizers."
            )
    model.eval()
    model.to(args.device)

    # Print the model summary
    if args.print_model_info:
        print_summary(model)
        print_model_blocks(model)
        print_module_structure(model)


    if args.compile:
        model = torch.compile(model)

    # Inference with different block size (note: for this one cannot use abs pos embeddings)
    if args.block_size:
        model.update_block_size(args.block_size)

    # Inference with different number of angles
    if args.sym_rot_num_angles:
        model.update_num_angles(args.sym_rot_num_angles)

    # Inference with different Rope Length
    if args.rope_length:
        model.update_rope_length(args.rope_length)

    if args.lm_eval_tasks:
        # Prepare wrapped model
        wrapped_model = NanoGPTLM.create_model(model=model, encode_fn=encode, decode_fn=decode, args=args)

        wrapped_model.evaluate_and_save(
            tasks=args.lm_eval_tasks.split(","),
            batch_size=args.batch_size,
            out_dir=out_dir,
            timestamp=timestamp,
            results_output=args.lm_eval_results_output
        )
        return

    if args.eval_only:
        print("Running in eval_only mode...")
        dataset_name = args.eval_dataset
        if dataset_name is None and args.init_from == 'resume':
            dataset_name = (
                checkpoint.get('config', {}).get('dataset')
                if isinstance(checkpoint, dict)
                else None
            )
        if dataset_name is None:
            raise ValueError(
                "--eval_dataset must be provided when running in eval_only mode"
            )

        print(f"Using validation dataset: {dataset_name}")
        print(f"Model block size: {model.config.block_size}")
        val_data = load_validation_data(model.config.block_size, dataset_name)
        metrics = calculate_validation_loss(
            model,
            val_data,
            model.config.block_size,
            args.eval_iters,
            args.device,
            ptdtype,
        )

        val_loss = metrics.get("val", float("nan"))
        print(f"Validation Loss: {val_loss:.4f}")
        if metrics.get("elapsed_time_s") is not None:
            print(f"Elapsed time: {metrics['elapsed_time_s']:.4f} seconds")

        summary: Dict[str, object] = dict(metrics)
        summary.setdefault("eval_dataset", dataset_name)
        summary.setdefault("timestamp", timestamp)
        summary.setdefault("out_dir", args.out_dir)
        summary.setdefault("init_from", args.init_from)

        write_eval_summary(
            args.out_dir,
            summary,
            extra_dirs=[out_dir],
        )
        return

    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
    # Obtain vector from the specified layer and save it to a file if required
    if args.save_avg_vector:
        x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]
        # Run the model to trigger vector extraction
        with torch.no_grad():
            with ctx:
                block_size = args.block_size if args.block_size else model.config.block_size
                idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                logits, _ = model(idx_cond, dataset_idx=dataset_idx)
        print(f"Obtained vector saved to {args.save_avg_vector}")

    if args.interactive:
        interactive_generation(model, start_ids, args.device, args.max_new_tokens, args.temperature, args.top_k, args.stop_strings, decode, encode)
    elif args.multicontext:
        if not args.multicontext_datasets:
            raise ValueError("Must specify --multicontext_datasets when using --multicontext")
        if args.multicontext_start is None:
            raise ValueError("Must specify --multicontext_start when using --multicontext")
        if len(args.multicontext_datasets) != len(args.multicontext_start):
            raise ValueError(
                "Number of --multicontext_datasets must match number of --multicontext_start strings."
            )

        dataset_names = list(args.multicontext_datasets)
        start_strings = list(args.multicontext_start)

        dataset_meta: Dict[str, Dict[str, object]] = {}
        decode_lookup: Dict[str, Callable[[Sequence[int]], str]] = {}
        initial_tokens: Dict[str, torch.Tensor] = {}

        for dataset_name, start_str in zip(dataset_names, start_strings):
            meta_path = os.path.join("data", dataset_name, "meta.pkl")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
            with open(meta_path, "rb") as f:
                dataset_meta[dataset_name] = pickle.load(f)

            encode_i, decode_i = get_tokenizer_functions(dataset_meta[dataset_name])
            token_ids = encode_i(start_str)
            if len(token_ids) == 0:
                if dataset_meta[dataset_name].get('tokenizer') == 'sinewave':
                    print(
                        f"Start string for dataset '{dataset_name}' produced no tokens; defaulting to '0'."
                    )
                    token_ids = [0]
                else:
                    raise ValueError(
                        f"Start string for dataset '{dataset_name}' produced no tokens. "
                        "Provide a valid prompt or comma-separated values for numerical tokenizers."
                    )

            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=args.device)[None, ...]
            initial_tokens[dataset_name] = token_tensor
            decode_lookup[dataset_name] = decode_i

        block_size = args.block_size if args.block_size else model.config.block_size
        with torch.no_grad(), ctx:
            for sample_idx in range(args.num_samples):
                if args.use_lsv and hasattr(args, 'lsv_size'):
                    model.set_lsv_index(sample_idx % args.lsv_size)
                    if args.lsv_scaling_factor is not None:
                        model.set_lsv_scaling_factor(args.lsv_scaling_factor)
                    if args.lsv_mixture is not None:
                        model.set_lsv_mode(2)
                        model.set_lsv_mixture(args.lsv_mixture)
                    else:
                        model.set_lsv_mode(1)

                token_state = {name: tensor.clone() for name, tensor in initial_tokens.items()}

                for _ in range(args.max_new_tokens):
                    idx_cond_dict = {}
                    for name in dataset_names:
                        tokens = token_state[name]
                        idx_cond_dict[name] = tokens if tokens.size(1) <= block_size else tokens[:, -block_size:]

                    logits_list, _ = model(None, token_dict=idx_cond_dict, target_dict=None)

                    for i, name in enumerate(dataset_names):
                        if model.config.numerical_multicontext:
                            preds = logits_list[i][:, -1]
                            preds = preds.squeeze(-1)
                            if preds.ndim == 0:
                                preds = preds.unsqueeze(0)
                            rounded = preds.round()
                            min_val = 0.0
                            max_val = None
                            meta_info = dataset_meta.get(name, {})
                            tokenizer_name = meta_info.get('tokenizer') if isinstance(meta_info, dict) else None
                            if tokenizer_name == 'sinewave':
                                max_val = 255.0
                            elif isinstance(meta_info, dict) and 'vocab_size' in meta_info:
                                max_val = float(meta_info['vocab_size'] - 1)

                            if max_val is not None:
                                rounded = torch.clamp(rounded, min=min_val, max=max_val)
                            else:
                                rounded = torch.clamp(rounded, min=min_val)

                            idx_next = rounded.to(torch.long).unsqueeze(-1)
                        else:
                            cur_logits = logits_list[i][:, -1, :] / args.temperature
                            if args.top_k is not None:
                                top_k_val = (
                                    args.top_k[0]
                                    if isinstance(args.top_k, (list, tuple))
                                    else args.top_k
                                )
                                k = min(top_k_val, cur_logits.size(-1))
                                v, _ = torch.topk(cur_logits, k)
                                cur_logits[cur_logits < v[:, [-1]]] = -float("inf")

                            probs = F.softmax(cur_logits, dim=-1)
                            idx_next = torch.multinomial(probs, num_samples=1)

                        token_state[name] = torch.cat((token_state[name], idx_next), dim=1)

                output_dict: Dict[str, str] = {}
                for name in dataset_names:
                    decode_fn = decode_lookup[name]
                    output_dict[name] = decode_fn(token_state[name][0].tolist())

                for name, text in output_dict.items():
                    key_color = "bold light_slate_blue"
                    text_color = "bold cyan"
                    print(f"\n[{key_color}]{name}:[/{key_color}]\n[{text_color}]{text}[/{text_color}]")
                print("---------------")

                if args.sample_file:
                    with open(args.sample_file, "w") as file:
                        for name, text in output_dict.items():
                            file.write(f"\n{name}: \n{text}\n")
    else:
        sample_with_existing_model(
                model,
                torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...],
                decode,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                num_samples=args.num_samples,
                colorize_output=args.colorize_output,
                colorize_mode=args.colorize_mode,
                token_boundary=args.token_boundary,
                show_heatmaps=args.show_heatmaps,
                chart_type=args.chart_type,
                last_k_tokens=args.last_k_tokens,
                out_dir=out_dir,
                sample_file=args.sample_file,
                args=args,
                dataset_idx=0,
                )

if __name__ == "__main__":
    main()


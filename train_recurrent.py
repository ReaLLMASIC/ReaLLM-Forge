# ======================================================================
# train_recurrent.py  –  latent-chaining fine-tuning (modularized)
# ======================================================================
#  * resumes from an existing checkpoint (no scratch / no GPT-2 import)
#  * feeds the HIDDEN state (after ln_f / scale_down) back as the next
#    “token”, skipping de-embedding, for the first `--latent_steps`
#  * keeps cross-entropy vs. ground-truth, with optional per-position
#    linear weighting and an initial “skip” window
# ----------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import GPT, GPTConfig  # patched model.py
from recurrent_variations.recurrent_block_variants import (
    RECURRENT_BLOCK_VARIANTS,
    RecurrentBlockConfig,
)
from train_args import parse_args as parse_generic_args
from train_variations.optimizer_variants import optimizer_dictionary


# ----------------------------------------------------------------------
# 1) ARGUMENTS
# ----------------------------------------------------------------------

def build_recurrent_parser() -> argparse.ArgumentParser:
    """Parser with options specific to latent-chaining recurrence."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--resume_ckpt",
        required=True,
        help="Path to .pt checkpoint produced by train.py",
    )
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=0,
        help="Chain this many hidden states before teacher-forcing",
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=0,
        help="Mask loss for the first K positions in every block",
    )
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument(
        "--reset_optim",
        action="store_true",
        help="Ignore optimiser state in the checkpoint",
    )
    parser.add_argument(
        "--recurrent_variant",
        default="latent_chaining",
        choices=sorted(RECURRENT_BLOCK_VARIANTS.keys()),
        help="Which recurrent block variant to use.",
    )
    parser.add_argument(
        "--max_tokens_per_epoch",
        type=int,
        default=None,
        help="Limit each epoch to a contiguous slice of tokens.",
    )
    parser.add_argument(
        "--progress_bar",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show a stdout progress bar during epochs.",
    )
    parser.add_argument(
        "--recurrent_output_ckpt",
        type=str,
        default="ckpt_lat.pt",
        help="Checkpoint filename for train_recurrent outputs.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Merge generic train_args with recurrent-specific flags."""
    recur_parser = build_recurrent_parser()
    latent_args, remaining = recur_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    generic_args, *_ = parse_generic_args()

    args = generic_args
    for key, value in vars(latent_args).items():
        setattr(args, key, value)

    return args


# ----------------------------------------------------------------------
# 2) CHECKPOINT + MODEL SETUP
# ----------------------------------------------------------------------

def unwrap_state_dict(wrapped_sd):
    """
    Remove '_orig_mod.' (torch.compile) and 'module.' (DDP) prefixes so the
    keys match a plain, single-GPU GPT instance.
    """
    clean = {}
    for key, value in wrapped_sd.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        clean[key] = value
    return clean


def load_checkpoint(args: argparse.Namespace, device: str):
    ckpt = torch.load(args.resume_ckpt, map_location=device)
    gpt_conf = GPTConfig(**ckpt["model_args"])
    model = GPT(gpt_conf).to(device)

    state_dict = unwrap_state_dict(ckpt["model"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"warning: {len(missing)} missing params (OK if all zero-grad)")
    if unexpected:
        print(f"warning: {len(unexpected)} extra params ignored")

    return ckpt, model, gpt_conf


def build_optimizer(model: GPT, args: argparse.Namespace, ckpt: dict):
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)

    param_groups = [
        {"params": decay, "weight_decay": args.opt_weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = optimizer_dictionary[args.optimizer](param_groups, args)
    if ckpt.get("optimizer") and not getattr(args, "reset_optim", False):
        optimizer.load_state_dict(ckpt["optimizer"])

    return optimizer


# ----------------------------------------------------------------------
# 3) DATA
# ----------------------------------------------------------------------

def load_bin(args: argparse.Namespace, split: str):
    path = os.path.join("data", args.dataset, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


# ----------------------------------------------------------------------
# 4) TRAINING LOOP BUILDING BLOCKS
# ----------------------------------------------------------------------

@dataclass
class TrainingState:
    global_step: int
    best_val_loss: float
    iter_num: int


def build_recurrent_config(args: argparse.Namespace) -> RecurrentBlockConfig:
    return RecurrentBlockConfig(
        latent_steps=args.latent_steps,
        skip_steps=args.skip_steps,
        weight_start=args.weight_start,
        weight_end=args.weight_end,
    )


def select_recurrent_block(args: argparse.Namespace):
    return RECURRENT_BLOCK_VARIANTS[args.recurrent_variant]


def train_block(
    *,
    recurrent_block,
    loss_config: RecurrentBlockConfig,
    embed_tokens,
    forward_embedded,
    x_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
):
    return recurrent_block(
        embed_tokens=embed_tokens,
        forward_embedded=forward_embedded,
        x_tokens=x_tokens,
        y_tokens=y_tokens,
        config=loss_config,
    )


def save_best_checkpoint(
    *,
    model: GPT,
    ckpt_model_args: dict,
    best_ckpt_path: str,
    best_val_loss: float,
    global_step: int,
):
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": ckpt_model_args,
            "iter_num": global_step,
            "best_val_loss": best_val_loss,
        },
        best_ckpt_path,
    )
    print(
        f"  ➜ new best @ step {global_step}; checkpoint saved to {best_ckpt_path}"
    )


def save_checkpoint(
    *,
    model: GPT,
    ckpt_model_args: dict,
    ckpt_path: str,
    best_val_loss: float,
    global_step: int,
    tag: str,
):
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": ckpt_model_args,
            "iter_num": global_step,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )
    print(f"  ➜ {tag} checkpoint saved to {ckpt_path}")


def run_epoch(
    *,
    split: str,
    data: np.memmap,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    state: TrainingState,
    device: str,
    block_size: int,
    loss_config: RecurrentBlockConfig,
    recurrent_block,
    embed_tokens,
    forward_embedded,
    tb: SummaryWriter | None,
    ckpt_model_args: dict,
    best_ckpt_path: str,
    evaluate_fn=None,
) -> float:
    losses = []
    ptr = 0
    data_view = data
    total_tokens = len(data_view) - 1
    save_enabled = args.always_save_checkpoint or not args.never_save_checkpoint

    if args.max_tokens_per_epoch:
        if args.max_tokens_per_epoch < block_size + 1:
            raise ValueError(
                "--max_tokens_per_epoch must be larger than block_size + 1"
            )
        if args.max_tokens_per_epoch < len(data_view):
            max_start = len(data_view) - args.max_tokens_per_epoch - 1
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + args.max_tokens_per_epoch + 1
            data_view = data_view[start:end]
            total_tokens = len(data_view) - 1

    total_blocks = max(1, (total_tokens - 1) // block_size)
    last_print_len = 0

    while ptr + block_size + 1 < len(data_view):
        seq = torch.from_numpy(
            np.array(data_view[ptr : ptr + block_size + 1], dtype=np.int64)
        ).to(device)
        x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)

        if split == "train":
            model.train()
            optimizer.zero_grad()
            loss = train_block(
                recurrent_block=recurrent_block,
                loss_config=loss_config,
                embed_tokens=embed_tokens,
                forward_embedded=forward_embedded,
                x_tokens=x,
                y_tokens=y,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            state.global_step += 1

            if state.global_step % args.log_interval == 0:
                print(f"iter {state.global_step:>7} | loss {loss.item():.4f}")

            if evaluate_fn and state.global_step % args.eval_interval == 0:
                val = evaluate_fn()

                if tb:
                    tb.add_scalar("loss/val", val, state.global_step)

                if val < state.best_val_loss:
                    state.best_val_loss = val
                    if save_enabled:
                        save_best_checkpoint(
                            model=model,
                            ckpt_model_args=ckpt_model_args,
                            best_ckpt_path=best_ckpt_path,
                            best_val_loss=state.best_val_loss,
                            global_step=state.global_step,
                        )
                if args.always_save_checkpoint:
                    save_checkpoint(
                        model=model,
                        ckpt_model_args=ckpt_model_args,
                        ckpt_path=best_ckpt_path,
                        best_val_loss=state.best_val_loss,
                        global_step=state.global_step,
                        tag="latest",
                    )
        else:
            model.eval()
            with torch.no_grad():
                loss = train_block(
                    recurrent_block=recurrent_block,
                    loss_config=loss_config,
                    embed_tokens=embed_tokens,
                    forward_embedded=forward_embedded,
                    x_tokens=x,
                    y_tokens=y,
                )

        losses.append(loss.item())
        ptr += block_size

        if args.progress_bar:
            completed = min(total_blocks, ptr // block_size)
            pct = completed / total_blocks
            bar_width = 24
            filled = int(bar_width * pct)
            bar = "#" * filled + "-" * (bar_width - filled)
            line = f"{split} [{bar}] {completed}/{total_blocks} ({pct:.0%})"
            padding = max(0, last_print_len - len(line))
            print(f"\r{line}{' ' * padding}", end="", flush=True)
            last_print_len = len(line)

    if args.progress_bar:
        print()

    return sum(losses) / len(losses)


# ----------------------------------------------------------------------
# 5) TRAINING DRIVER
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt, model, gpt_conf = load_checkpoint(args, device)
    optimizer = build_optimizer(model, args, ckpt)

    best_val_loss = ckpt["best_val_loss"].item()
    print("best_val_loss", best_val_loss)
    best_val_loss = 5.00  # TODO: allow configurable start threshold

    state = TrainingState(
        global_step=0,
        best_val_loss=best_val_loss,
        iter_num=ckpt["iter_num"],
    )

    train_bin = load_bin(args, "train")
    val_bin = load_bin(args, "val")

    embed_tokens = model.embed_tokens
    forward_embedded = lambda x: model.forward_embedded(x, return_hidden=True)

    block_size = gpt_conf.block_size
    loss_config = build_recurrent_config(args)
    recurrent_block = select_recurrent_block(args)

    tb = SummaryWriter() if getattr(args, "tensorboard_log", False) else None
    best_ckpt_path = os.path.join(
        os.path.dirname(args.resume_ckpt), args.recurrent_output_ckpt
    )
    save_enabled = args.always_save_checkpoint or not args.never_save_checkpoint

    val_loss = 999.9

    def evaluate_validation() -> float:
        val = run_epoch(
            split="val",
            data=val_bin,
            model=model,
            optimizer=None,
            args=args,
            state=state,
            device=device,
            block_size=block_size,
            loss_config=loss_config,
            recurrent_block=recurrent_block,
            embed_tokens=embed_tokens,
            forward_embedded=forward_embedded,
            tb=tb,
            ckpt_model_args=ckpt["model_args"],
            best_ckpt_path=best_ckpt_path,
        )
        print(f"val loss {val:.4f}")
        return val

    while state.global_step < args.max_iters:
        t0 = time.time()
        train_loss = run_epoch(
            split="train",
            data=train_bin,
            model=model,
            optimizer=optimizer,
            args=args,
            state=state,
            device=device,
            block_size=block_size,
            loss_config=loss_config,
            recurrent_block=recurrent_block,
            embed_tokens=embed_tokens,
            forward_embedded=forward_embedded,
            tb=tb,
            ckpt_model_args=ckpt["model_args"],
            best_ckpt_path=best_ckpt_path,
            evaluate_fn=evaluate_validation,
        )

        if state.global_step % args.eval_interval == 0:
            val_loss = evaluate_validation()

            if tb:
                tb.add_scalar("loss/val", val_loss, state.global_step)

            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                if save_enabled:
                    save_best_checkpoint(
                        model=model,
                        ckpt_model_args=ckpt["model_args"],
                        best_ckpt_path=best_ckpt_path,
                        best_val_loss=state.best_val_loss,
                        global_step=state.global_step,
                    )
            if args.always_save_checkpoint:
                save_checkpoint(
                    model=model,
                    ckpt_model_args=ckpt["model_args"],
                    ckpt_path=best_ckpt_path,
                    best_val_loss=state.best_val_loss,
                    global_step=state.global_step,
                    tag="latest",
                )

        if tb:
            tb.add_scalar("loss/train", train_loss, state.global_step)

        print(
            "iter "
            f"{state.global_step:03d} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | {(time.time() - t0):.1f}s"
        )

    if tb:
        tb.flush()
        tb.close()

    print("done.")


if __name__ == "__main__":
    main()
# ======================================================================

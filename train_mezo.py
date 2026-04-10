#!/usr/bin/env python3
"""MeZO training loop (zeroth-order) for nanoGPT models.

This script mirrors the existing train.py CLI arguments where possible, but
uses in-place zeroth-order updates that only require forward passes.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import time
from dataclasses import dataclass, fields

import numpy as np
import torch

from model import GPT, GPTConfig
from sample import get_tokenizer_functions, sample_with_existing_model
from train_args import parse_args
from train_variations.loss_variants import build_loss_function


@dataclass
class CheckpointState:
    iter_num: int
    best_val_loss: float
    best_iter: int


def get_vocab_size_from_meta(dataset: str, out_dir: str) -> int:
    meta = load_meta(dataset, out_dir)
    if "vocab_size" not in meta:
        raise KeyError(f"meta.pkl missing vocab_size at data/{dataset}/meta.pkl")
    return meta["vocab_size"]


def load_meta(dataset: str, out_dir: str) -> dict:
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as handle:
        pickle.dump(meta, handle)
    return meta


def load_dataset(dataset: str, vocab_size: int) -> tuple[np.memmap, np.memmap]:
    dtype = np.uint32 if vocab_size == 100277 else np.uint16
    train_data = np.memmap(os.path.join("data", dataset, "train.bin"), dtype=dtype, mode="r")
    val_data = np.memmap(os.path.join("data", dataset, "val.bin"), dtype=dtype, mode="r")
    return train_data, val_data


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in idx])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def perturb_parameters(model: torch.nn.Module, epsilon: float, generator: torch.Generator) -> None:
    for param in model.parameters():
        if not param.requires_grad:
            continue
        noise = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator)
        param.add_(epsilon * noise)


def mezo_update(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
    epsilon: float,
    lr: float,
    seed: int,
    iter_num: int,
) -> torch.Tensor:
    device = inputs.device
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(seed)
        perturb_parameters(model, epsilon, generator)
        _, loss_pos = model(inputs, targets, iter_num=iter_num, loss_fn=loss_fn)

        generator = torch.Generator(device=device).manual_seed(seed)
        perturb_parameters(model, -2.0 * epsilon, generator)
        _, loss_neg = model(inputs, targets, iter_num=iter_num, loss_fn=loss_fn)

        generator = torch.Generator(device=device).manual_seed(seed)
        perturb_parameters(model, epsilon, generator)

        projected_grad = (loss_pos - loss_neg) / (2.0 * epsilon)

        generator = torch.Generator(device=device).manual_seed(seed)
        for param in model.parameters():
            if not param.requires_grad:
                continue
            noise = torch.randn(param.shape, device=param.device, dtype=param.dtype, generator=generator)
            param.add_(-lr * projected_grad * noise)

        return (loss_pos + loss_neg) * 0.5


def estimate_loss(
    model: torch.nn.Module,
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: torch.device,
    loss_fn,
    eval_iters: int,
    iter_num: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y, iter_num=iter_num, loss_fn=loss_fn)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    out_dir: str,
    model: torch.nn.Module,
    model_args: dict,
    state: CheckpointState,
    filename: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "model_args": model_args,
        "iter_num": state.iter_num,
        "best_val_loss": state.best_val_loss,
        "best_iter": state.best_iter,
    }
    torch.save(ckpt, os.path.join(out_dir, filename))


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def build_model_args(args, vocab_size: int) -> dict:
    model_fields = {field.name for field in fields(GPTConfig)}
    model_args = {name: getattr(args, name) for name in model_fields if hasattr(args, name)}
    model_args["vocab_size"] = vocab_size
    return model_args


def sample_and_print(
    model: torch.nn.Module,
    args,
    device: torch.device,
    encode,
    decode,
    iter_num: int,
    best_val_loss: float,
) -> None:
    model.eval()
    start_ids = torch.tensor(encode(args.sample_start_tokens), dtype=torch.long, device=device)[None, ...]
    sample_with_existing_model(
        model=model,
        start_ids=start_ids,
        start_tokens=args.sample_start_tokens,
        decode=decode,
        device=device,
        out_dir=args.out_dir,
        max_new_tokens=args.max_sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        colorize_output=args.colorize_output,
        colorize_mode=args.colorize_mode,
        token_boundary=(args.token_boundary or None),
        show_heatmaps=args.show_heatmaps,
        chart_type=args.chart_type,
        sample_file=args.sample_file,
        num_samples=args.num_samples,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        run_name=args.tensorboard_run_name,
        args=args,
        writer=None,
    )
    model.train()


def main() -> None:
    args, _model_group, _training_group, _logging_group = parse_args()
    if args.training_mode != "single":
        raise ValueError("train_mezo.py only supports training_mode=single")
    if args.dataset_list is not None or args.multicontext_datasets is not None:
        raise ValueError("train_mezo.py does not support multi-dataset or multicontext training")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.init_from == "scratch":
        vocab_size = getattr(args, "vocab_size", None) or get_vocab_size_from_meta(args.dataset, args.out_dir)
        model_args = build_model_args(args, vocab_size)
        config_json = {**model_args, "max_iters": args.max_iters, "batch_size": args.batch_size}
        with open(os.path.join(args.out_dir, "full_config.json"), "w") as handle:
            json.dump(config_json, handle, indent=4)
        model = GPT(GPTConfig(**model_args)).to(device)
        state = CheckpointState(iter_num=0, best_val_loss=float("inf"), best_iter=0)
    elif args.init_from in {"resume", "prev_run"}:
        ckpt_root = args.out_dir if args.init_from == "resume" else args.prev_run_ckpt
        ckpt_path = os.path.join(ckpt_root, args.init_from_ckpt)
        checkpoint = load_checkpoint(ckpt_path, device)
        model_args = checkpoint["model_args"]
        model = GPT(GPTConfig(**model_args)).to(device)
        model.load_state_dict(checkpoint["model"])
        state = CheckpointState(
            iter_num=checkpoint["iter_num"] if args.init_from == "resume" else 0,
            best_val_loss=checkpoint.get("best_val_loss", float("inf")),
            best_iter=checkpoint.get("best_iter", 0),
        )
    else:
        raise ValueError(f"Unsupported init_from value: {args.init_from}")

    train_data, val_data = load_dataset(args.dataset, model_args["vocab_size"])
    loss_fn = build_loss_function(args)
    encode = decode = None
    if args.max_sample_tokens is not None:
        meta = load_meta(args.dataset, args.out_dir)
        encode, decode = get_tokenizer_functions(meta)

    t_start = time.time()
    while state.iter_num < args.max_iters:
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        if args.mezo_seed is None:
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = args.mezo_seed + state.iter_num
        loss = mezo_update(
            model,
            x,
            y,
            loss_fn,
            epsilon=args.mezo_epsilon,
            lr=args.learning_rate,
            seed=seed,
            iter_num=state.iter_num,
        )

        if state.iter_num % args.log_interval == 0:
            elapsed = time.time() - t_start
            print(
                f"iter {state.iter_num}: loss {loss.item():.4f}, "
                f"lr {args.learning_rate:.2e}, time {elapsed:.2f}s"
            )

        if state.iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                val_data,
                args.block_size,
                args.batch_size,
                device,
                loss_fn,
                args.eval_iters,
                state.iter_num,
            )
            print(f"iter {state.iter_num}: val loss {val_loss:.4f}")
            is_best = val_loss < state.best_val_loss
            if is_best:
                state.best_val_loss = val_loss
                state.best_iter = state.iter_num
                save_checkpoint(args.out_dir, model, model_args, state, args.mezo_output_ckpt)
            if args.max_sample_tokens is not None and (is_best or args.sample_each_eval):
                if encode is None or decode is None:
                    meta = load_meta(args.dataset, args.out_dir)
                    encode, decode = get_tokenizer_functions(meta)
                sample_and_print(model, args, device, encode, decode, state.iter_num, state.best_val_loss)

        state.iter_num += 1

    save_checkpoint(args.out_dir, model, model_args, state, args.mezo_output_ckpt)


if __name__ == "__main__":
    main()

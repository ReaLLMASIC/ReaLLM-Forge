#!/usr/bin/env python3
"""End-to-end modular addition grokking demo.

This script prepares a held-out modular-addition dataset, optionally trains a
small nanoGPT model, and evaluates a saved checkpoint by querying every
``a+b=`` prompt and checking whether the generated answer equals ``(a+b) % p``.

The defaults follow common grokking replications: prime modulus 113, a limited
30% training split, no dropout, AdamW with substantial weight decay, and a small
1-layer transformer trained for many iterations.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate modular-addition grokking.")
    parser.add_argument("--modulo", "--modulus", dest="modulus", type=int, default=113, help="Modulo p for a+b mod p.")
    parser.add_argument("--train-fraction", type=float, default=0.3, help="Fraction of ordered pairs used for training.")
    parser.add_argument("--train-repeats", type=int, default=200, help="How often to repeat training equations in train.bin.")
    parser.add_argument("--val-repeats", type=int, default=20, help="How often to repeat held-out equations in val.bin.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed and evaluation sample seed.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "modular_arithmetic", help="Dataset output directory.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Training/checkpoint output directory.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint to evaluate. Defaults to OUT_DIR/ckpt.pt.")
    parser.add_argument("--skip-train", action="store_true", help="Only prepare data and evaluate an existing checkpoint.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing data files instead of regenerating them.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"), help="Torch device for training/eval.")
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "float16"), choices=["float32", "float16", "bfloat16"], help="Training/eval dtype.")
    parser.add_argument("--max-iters", type=int, default=20000, help="Training iterations.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Training eval interval.")
    parser.add_argument("--eval-iters", type=int, default=100, help="Training eval iterations.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0, help="AdamW weight decay; high values encourage grokking.")
    parser.add_argument("--n-layer", type=int, default=1, help="Transformer layers.")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads.")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding width.")
    parser.add_argument("--block-size", type=int, default=32, help="Context window.")
    parser.add_argument("--eval-split", choices=["all", "train", "val"], default="all", help="Which equation split to evaluate.")
    parser.add_argument("--max-eval-examples", type=int, default=None, help="Optional cap on evaluated equations.")
    parser.add_argument("--show-examples", type=int, default=12, help="Number of predictions to print.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_data(args: argparse.Namespace) -> None:
    run([
        sys.executable, str(ROOT / "data" / "modular_arithmetic" / "prepare.py"),
        "--out-dir", str(args.data_dir),
        "--modulus", str(args.modulus),
        "--train-fraction", str(args.train_fraction),
        "--train-repeats", str(args.train_repeats),
        "--val-repeats", str(args.val_repeats),
        "--seed", str(args.seed),
    ])


def train(args: argparse.Namespace, out_dir: Path) -> None:
    run([
        sys.executable, "train.py",
        "--dataset", "modular_arithmetic",
        "--out_dir", str(out_dir),
        "--device", args.device,
        "--dtype", args.dtype,
        "--block_size", str(args.block_size),
        "--batch_size", str(args.batch_size),
        "--n_layer", str(args.n_layer),
        "--n_head", str(args.n_head),
        "--n_embd", str(args.n_embd),
        "--dropout", "0.0",
        "--bias",
        "--max_iters", str(args.max_iters),
        "--eval_interval", str(args.eval_interval),
        "--eval_iters", str(args.eval_iters),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_iters", "100",
        "--decay_lr",
        "--min_lr", "1e-5",
        "--always_save_checkpoint",
        "--only_save_checkpoint_at_end",
        "--no-compile",
    ])


def load_model(ckpt_path: Path, device: str):
    import torch
    from model import GPT
    from gpt_conf import GPTConfig

    from inspect import signature

    load_kwargs = {"map_location": device}
    if "weights_only" in signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(ckpt_path, **load_kwargs)
    checkpoint["model_args"]["dropout"] = 0.0
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, checkpoint


def split_pairs(args: argparse.Namespace):
    rng = random.Random(args.seed)
    pairs = [(a, b) for a in range(args.modulus) for b in range(args.modulus)]
    rng.shuffle(pairs)
    split = int(len(pairs) * args.train_fraction)
    return pairs[:split], pairs[split:]


def encode(text: str, stoi: dict[str, int]):
    import torch

    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long).unsqueeze(0)


def predict_answer(model, prompt: str, stoi: dict[str, int], itos: dict[int, str], device: str, max_digits: int) -> str:
    import torch
    from torch.nn import functional as F

    idx = encode(prompt, stoi).to(device)
    out_chars: list[str] = []
    for _ in range(max_digits + 1):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = int(torch.argmax(probs, dim=-1).item())
        ch = itos[next_id]
        if ch == "\n":
            break
        out_chars.append(ch)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return "".join(out_chars)


def evaluate(args: argparse.Namespace, ckpt_path: Path) -> dict[str, float]:
    import torch

    meta_path = args.data_dir / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    model, _ = load_model(ckpt_path, args.device)
    train_pairs, val_pairs = split_pairs(args)
    pairs = {"train": train_pairs, "val": val_pairs, "all": train_pairs + val_pairs}[args.eval_split]
    if args.max_eval_examples is not None:
        pairs = pairs[: args.max_eval_examples]
    max_digits = len(str(args.modulus - 1))
    correct = 0
    examples = []
    with torch.no_grad():
        for a, b in pairs:
            expected = str((a + b) % args.modulus)
            pred = predict_answer(model, f"{a}+{b}=", stoi, itos, args.device, max_digits)
            ok = pred == expected
            correct += int(ok)
            if len(examples) < args.show_examples:
                examples.append({"prompt": f"{a}+{b}=", "prediction": pred, "expected": expected, "correct": ok})
    accuracy = correct / max(1, len(pairs))
    result = {"split": args.eval_split, "accuracy": accuracy, "correct": correct, "total": len(pairs), "examples": examples}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or ROOT / "out" / f"modular_arithmetic_grokking_p{args.modulus}"
    ckpt_path = args.ckpt or out_dir / "ckpt.pt"
    args.data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        prepare_data(args)
    if not args.skip_train:
        train(args, out_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    evaluate(args, ckpt_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare modular arithmetic data for grokking experiments.

This creates the standard nanoGPT train.bin/val.bin/meta.pkl artifacts from all
ordered pairs (a, b) for modular addition: "a+b=c\n" where c=(a+b) mod p.
The default train split is a random subset of equations, while validation holds
out the remaining equations, matching the train/held-out split used in modular
addition grokking demos.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

from array import array


def parse_args():
    parser = argparse.ArgumentParser(description="Create a modular addition grokking dataset.")
    parser.add_argument("--modulus", type=int, default=113, help="Prime modulus p for a+b mod p.")
    parser.add_argument("--train-fraction", type=float, default=0.3, help="Fraction of equations used for training.")
    parser.add_argument("--train-repeats", type=int, default=200, help="How often to repeat the train equation set.")
    parser.add_argument("--val-repeats", type=int, default=20, help="How often to repeat the held-out equation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split and train order.")
    parser.add_argument("--operator", default="+", choices=["+"], help="Operation to render; currently modular addition.")
    parser.add_argument("--out-dir", default=".", help="Directory for train.bin, val.bin, and meta.pkl.")
    parser.add_argument("--write-text", action=argparse.BooleanOptionalAction, default=True, help="Also write train.txt and val.txt.")
    return parser.parse_args()


def encode_examples(examples, stoi):
    text = "".join(examples)
    ids = array("H", (stoi[ch] for ch in text))
    return ids, text


def main():
    args = parse_args()
    if args.modulus <= 1:
        raise ValueError("--modulus must be greater than 1")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.train_repeats < 1 or args.val_repeats < 1:
        raise ValueError("repeat counts must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    equations = [
        f"{a}{args.operator}{b}={(a + b) % args.modulus}\n"
        for a in range(args.modulus)
        for b in range(args.modulus)
    ]
    rng.shuffle(equations)
    split = int(len(equations) * args.train_fraction)
    train_equations = equations[:split]
    val_equations = equations[split:]

    train_examples = train_equations * args.train_repeats
    val_examples = val_equations * args.val_repeats
    rng.shuffle(train_examples)

    chars = sorted(set("".join(equations)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    train_ids, train_text = encode_examples(train_examples, stoi)
    val_ids, val_text = encode_examples(val_examples, stoi)
    with open(out_dir / "train.bin", "wb") as f:
        train_ids.tofile(f)
    with open(out_dir / "val.bin", "wb") as f:
        val_ids.tofile(f)

    if args.write_text:
        (out_dir / "train.txt").write_text(train_text, encoding="utf-8")
        (out_dir / "val.txt").write_text(val_text, encoding="utf-8")

    meta = {
        "vocab_size": len(chars),
        "itos": itos,
        "stoi": stoi,
        "tokenizer": "char",
        "dataset": "modular_arithmetic",
        "operation": "addition_mod_p",
        "modulus": args.modulus,
        "train_fraction": args.train_fraction,
        "train_equations": len(train_equations),
        "val_equations": len(val_equations),
        "train_repeats": args.train_repeats,
        "val_repeats": args.val_repeats,
        "seed": args.seed,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir / 'train.bin'} ({len(train_ids):,} tokens)")
    print(f"Wrote {out_dir / 'val.bin'} ({len(val_ids):,} tokens)")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Train equations: {len(train_equations):,}; validation equations: {len(val_equations):,}")


if __name__ == "__main__":
    main()

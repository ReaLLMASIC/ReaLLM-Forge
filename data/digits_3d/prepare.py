#!/usr/bin/env python3
"""Build a configurable character vocabulary for the 3D token demo."""

import argparse
import pickle
from pathlib import Path

import numpy as np


TRAINED_SYMBOLS = "0123456789!@#$%^&*()[]{}<>?/|+-=_~:;,."
HELD_OUT_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_dataset(
    out_dir: Path,
    train_repeats: int,
    val_repeats: int,
    num_digits: int = 10,
    num_letters: int = 10,
    dropout_count: int = 0,
) -> None:
    if train_repeats < 2 or val_repeats < 2:
        raise ValueError("repeat counts must be at least 2")

    if not 1 <= num_digits <= len(TRAINED_SYMBOLS):
        raise ValueError(f"num_digits must be between 1 and {len(TRAINED_SYMBOLS)}")
    if not 0 <= num_letters <= len(HELD_OUT_SYMBOLS):
        raise ValueError(f"num_letters must be between 0 and {len(HELD_OUT_SYMBOLS)}")
    if not 0 <= dropout_count < num_digits:
        raise ValueError("dropout_count must be between 0 and num_digits - 1")
    all_trained = TRAINED_SYMBOLS[:num_digits]
    dropped_tokens = all_trained[-dropout_count:] if dropout_count else ""
    sequence = all_trained[:-dropout_count] if dropout_count else all_trained
    held_out = HELD_OUT_SYMBOLS[:num_letters]
    vocab = all_trained + held_out

    out_dir.mkdir(parents=True, exist_ok=True)
    stoi = {char: index for index, char in enumerate(vocab)}
    itos = {index: char for char, index in stoi.items()}

    def write_split(name: str, repeats: int) -> None:
        # Deliberately encode only digits. The letters remain valid vocabulary
        # entries so their untrained vectors can be compared with trained ones.
        ids = np.asarray([stoi[c] for c in sequence * repeats], dtype=np.uint16)
        ids.tofile(out_dir / f"{name}.bin")

    write_split("train", train_repeats)
    write_split("val", val_repeats)
    metadata = {
        "vocab_size": len(vocab),
        "stoi": stoi,
        "itos": itos,
        "train_tokens": len(sequence) * train_repeats,
        "val_tokens": len(sequence) * val_repeats,
        "trained_tokens": list(sequence),
        "initial_trained_tokens": list(all_trained),
        "dropped_tokens": list(dropped_tokens),
        "unseen_tokens": list(held_out),
        "description": f"Repeated {sequence}; {held_out or 'no symbols'} are vocabulary-only controls.",
    }
    with (out_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--train-repeats", type=int, default=2000)
    parser.add_argument("--val-repeats", type=int, default=200)
    parser.add_argument("--num-digits", type=int, default=10, help="Number of trained symbols (0-9, then punctuation).")
    parser.add_argument("--num-letters", type=int, default=10, help="Number of held-out alphabetic vocabulary symbols.")
    parser.add_argument("--dropout-count", type=int, default=0, help="Remove this many trailing trained symbols from both splits while retaining them in the vocabulary.")
    args = parser.parse_args()
    build_dataset(args.out_dir, args.train_repeats, args.val_repeats, args.num_digits, args.num_letters, args.dropout_count)
    print(f"Wrote {args.num_digits - args.dropout_count} active, {args.dropout_count} dropped, and {args.num_letters} held-out symbols to {args.out_dir}")


if __name__ == "__main__":
    main()

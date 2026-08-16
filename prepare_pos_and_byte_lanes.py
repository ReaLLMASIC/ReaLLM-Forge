#!/usr/bin/env python3
"""Prepare Full POS tag lane and 256-Byte Fallback companion stream from 59M token corpus."""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from hangul_factorizer import (
    HangulFullPosFactorizedTokenizer,
    HangulCoarsePosFactorizedTokenizer,
    make_byte_fallback_meta,
    POS_TAGS_FULL,
    POS_TAGS_COARSE,
)
from kiwipiepy import Kiwi


def prepare_byte_companion_lane(
    input_file: str,
    output_dir: str,
    percentage_train: float = 0.9,
):
    """Encode companion character stream using 256-byte fallback into train.bin/val.bin."""
    print(f"\n[1/2] Preparing 256-Byte Fallback Companion Lane in {output_dir}...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta = make_byte_fallback_meta()
    with open(out_path / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved byte fallback meta.pkl with vocab_size={meta['vocab_size']}")

    # Read input text
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    total_chars = len(text)
    print(f"Total characters to encode: {total_chars:,}")

    # Encode characters to byte values 0-255
    # ASCII < 256 mapped to ord(c), non-ASCII mapped to UTF-8 lead byte / ord(c)%256
    encoded = np.empty(total_chars, dtype=np.uint16)
    for i, c in enumerate(tqdm(text, desc="Encoding byte fallback stream")):
        code = ord(c)
        encoded[i] = code if code < 256 else c.encode("utf-8")[0]

    # Split train / val
    n_train = int(total_chars * percentage_train)
    train_data = encoded[:n_train]
    val_data = encoded[n_train:]

    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    train_data.tofile(out_path / "train.bin")
    val_data.tofile(out_path / "val.bin")
    print(f"Saved {out_path / 'train.bin'} and {out_path / 'val.bin'} successfully.")


def prepare_full_pos_lane(
    input_file: str,
    output_dir: str,
    percentage_train: float = 0.9,
    batch_size: int = 10000,
    num_workers: int = 8
):
    """Tag 59M tokens with Full (46 Sejong tags) POS tokenizer and save binary lane."""
    print(f"\n[2/2] Preparing Full POS Tag Lane (46 tags) in {output_dir}...")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tok = HangulFullPosFactorizedTokenizer(use_pos=True)
    pos_values = tok.id_to_value[23]
    value_to_id = tok.value_to_id[23]

    stoi = {tok.token_for(23, i): i for i in range(len(pos_values))}
    itos = {i: tok.token_for(23, i) for i in range(len(pos_values))}
    meta = {
        "vocab_size": len(pos_values),
        "tokenizer": "hangul_full_pos",
        "pos_tags": pos_values,
        "stoi": stoi,
        "itos": itos,
    }
    with open(out_path / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved full POS meta.pkl with vocab_size={len(pos_values)}")

    # Initialize Kiwi tagger
    kiwi = Kiwi(num_workers=num_workers)
    print(f"Initialized Kiwi POS tagger with {num_workers} workers.")

    # Read and batch process lines
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total_lines = len(lines)
    total_chars = sum(len(l) for l in lines)
    print(f"Loaded {total_lines:,} lines ({total_chars:,} characters). Tagging in batches...")

    all_ids = []
    t0 = time.time()

    for b_start in range(0, total_lines, batch_size):
        b_lines = lines[b_start : b_start + batch_size]
        # Tokenize batch of lines
        token_results = kiwi.tokenize(b_lines)
        for line, tokens in zip(b_lines, token_results):
            char_pos = {}
            for t in tokens:
                for idx in range(t.start, t.start + t.len):
                    char_pos[idx] = t.tag
            for c_idx in range(len(line)):
                tag = char_pos.get(c_idx, "UNK")
                pos_id = value_to_id.get(tag, 1)  # 1 is UNK
                all_ids.append(pos_id)

        done_lines = min(b_start + batch_size, total_lines)
        elapsed = time.time() - t0
        rate = len(all_ids) / elapsed if elapsed > 0 else 0
        print(f"Processed {done_lines:,}/{total_lines:,} lines ({len(all_ids):,} tokens, {rate:.0f} tok/s)...")

    encoded = np.array(all_ids, dtype=np.uint16)
    n_train = int(len(encoded) * percentage_train)
    train_data = encoded[:n_train]
    val_data = encoded[n_train:]

    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    train_data.tofile(out_path / "train.bin")
    val_data.tofile(out_path / "val.bin")
    print(f"Saved {out_path / 'train.bin'} and {out_path / 'val.bin'} successfully.")


def main():
    input_file = "data/korean_pos_mc/input.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input corpus not found at {input_file}")

    # 1. Byte fallback companion lane
    prepare_byte_companion_lane(
        input_file=input_file,
        output_dir="data/korean_pos_mc/char_byte"
    )

    # 2. Full POS tag lane
    prepare_full_pos_lane(
        input_file=input_file,
        output_dir="data/korean_pos_mc/pos_full"
    )

    print("\n==========================================================================")
    print(" Lane Preparation Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()

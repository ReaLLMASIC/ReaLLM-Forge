#!/usr/bin/env python3
"""Prepare the 24 Dataset Lanes for Hangul Hybrid Factorizer without Char Stream Overhead.

Produces:
- SentencePiece model for non-Korean BPE tokens in data/korean_pos_mc_nochar/spm_non_korean.model
- 24 aligned dataset lanes under data/korean_pos_mc_nochar/
  * Lane 0: script_bpe (BPE subwords & byte fallback + <hangul>)
  * Lanes 1..22: 22 Hangul phonetic & structural factor lanes
  * Lane 23 (Coarse): pos (17 coarse tags)
  * Lane 23 (Full): pos_full (46 Sejong tags)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from hangul_pos_hybrid_tokenizer import (
    HangulHybridPosTokenizer,
    train_sentencepiece_non_korean,
)
from kiwipiepy import Kiwi


def main():
    parser = argparse.ArgumentParser(description="Prepare 24-lane hybrid Hangul factorizer dataset.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/korean_pos_mc/input.txt",
        help="Input text corpus path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/korean_pos_mc_nochar",
        help="Output root directory for lanes",
    )
    parser.add_argument(
        "--bpe_vocab_size",
        type=int,
        default=4096,
        help="SentencePiece BPE vocabulary size for non-Korean text",
    )
    parser.add_argument(
        "--percentage_train",
        type=float,
        default=0.9,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Batch size (lines) for tagging and encoding",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Kiwi worker processes",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    spm_prefix = str(out_root / "spm_non_korean")
    spm_model_path = f"{spm_prefix}.model"

    # Step 1: Train SentencePiece model if not already trained
    if not os.path.exists(spm_model_path):
        print("\n==========================================================================")
        print(f" STEP 1: Training SentencePiece Model (vocab={args.bpe_vocab_size})")
        print("==========================================================================")
        spm_model_path = train_sentencepiece_non_korean(
            input_file=args.input_file,
            output_prefix=spm_prefix,
            vocab_size=args.bpe_vocab_size,
        )
    else:
        print(f"\nFound existing SentencePiece model at {spm_model_path}")

    # Step 2: Initialize tokenizers for coarse and full POS
    tok_coarse = HangulHybridPosTokenizer(spm_model_path, use_pos=True, pos_mode="coarse")
    tok_full = HangulHybridPosTokenizer(spm_model_path, use_pos=True, pos_mode="full")

    # Step 3: Save meta.pkl for all 23 common lanes + 2 POS variants
    print("\n==========================================================================")
    print(" STEP 2: Writing Lane Metadata & meta.pkl Files")
    print("==========================================================================")
    tok_coarse.save_meta_pkls(out_root)

    # Save pos_full meta.pkl
    pos_full_dir = out_root / "pos_full"
    pos_full_dir.mkdir(parents=True, exist_ok=True)
    stoi_full = {v: idx for idx, v in enumerate(tok_full.lanes[23].values)}
    itos_full = {idx: v for idx, v in enumerate(tok_full.lanes[23].values)}
    meta_full = {
        "vocab_size": len(tok_full.lanes[23].values),
        "tokenizer": "hangul_hybrid_lane_23_pos_full",
        "lane_index": 23,
        "lane_name": "pos_full",
        "pos_tags": list(tok_full.lanes[23].values),
        "stoi": stoi_full,
        "itos": itos_full,
    }
    with open(pos_full_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta_full, f)

    print(f"Saved meta.pkl for all 24 lanes (including pos and pos_full) in {out_root}")

    # Step 4: Batch process corpus
    print("\n==========================================================================")
    print(" STEP 3: Tagging and Encoding Corpus across 24 Lanes")
    print("==========================================================================")
    kiwi = Kiwi(num_workers=args.num_workers)
    print(f"Initialized Kiwi with {args.num_workers} workers.")

    with open(args.input_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Loaded {total_lines:,} lines from {args.input_file}.")

    # Collect 25 token arrays: 23 common lanes + pos (coarse) + pos_full
    # Lane 0: script_bpe
    # Lanes 1..22: factor lanes
    # Lane 23: pos (coarse)
    # Lane 24: pos_full
    all_lane_tokens: List[List[int]] = [[] for _ in range(25)]

    t0 = time.time()
    for b_start in range(0, total_lines, args.batch_size):
        b_lines = lines[b_start : b_start + args.batch_size]
        token_results = kiwi.tokenize(b_lines)

        for line, tokens in zip(b_lines, token_results):
            if not line:
                continue

            # Map character index to coarse and full POS tags
            char_pos_coarse: Dict[int, str] = {}
            char_pos_full: Dict[int, str] = {}
            for t in tokens:
                raw_tag = t.tag
                coarse_tag = tok_coarse.pos_tags if raw_tag in tok_coarse.pos_tags else "UNK"
                from hangul_factorizer import POS_TAG_MAP
                mapped_coarse = POS_TAG_MAP.get(raw_tag, "UNK")
                for idx in range(t.start, t.start + t.len):
                    char_pos_full[idx] = raw_tag
                    char_pos_coarse[idx] = mapped_coarse

            # Encode with coarse tokenizer
            steps_coarse = tok_coarse.encode_text(line, char_pos_tags=char_pos_coarse)

            # Also compute pos_full for each step
            # For Korean syllables, find the full tag
            # For non-Korean BPE tokens, pos_full is PAD (0)
            pattern = tok_coarse.sp.piece_to_id("<hangul>")
            cur_line_pos = 0

            # Encode line segments to align full POS tags
            import re
            segments = [m.group(0) for m in re.finditer(r"([가-힣]+|[^가-힣]+)", line)]
            pos_full_ids = []
            cur_p = 0
            for seg in segments:
                if re.match(r"^[가-힣]+$", seg):
                    for ch in seg:
                        raw_tag = char_pos_full.get(cur_p, "UNK")
                        f_id = tok_full.value_to_id[23].get(raw_tag, 1)
                        pos_full_ids.append(f_id)
                        cur_p += 1
                else:
                    b_ids = tok_coarse.sp.encode_as_ids(seg)
                    for _ in b_ids:
                        pos_full_ids.append(0)  # PAD
                    cur_p += len(seg)

            assert len(steps_coarse) == len(pos_full_ids), (
                f"Mismatch: {len(steps_coarse)} coarse steps vs {len(pos_full_ids)} full steps"
            )

            # Append to lane buffers
            for step_idx, step in enumerate(steps_coarse):
                for lane_i in range(24):
                    all_lane_tokens[lane_i].append(step[lane_i])
                all_lane_tokens[24].append(pos_full_ids[step_idx])

        done = min(b_start + args.batch_size, total_lines)
        elapsed = time.time() - t0
        rate = len(all_lane_tokens[0]) / elapsed if elapsed > 0 else 0
        print(f"Processed {done:,}/{total_lines:,} lines ({len(all_lane_tokens[0]):,} hybrid steps, {rate:.0f} steps/s)...")

    total_steps = len(all_lane_tokens[0])
    n_train = int(total_steps * args.percentage_train)
    n_val = total_steps - n_train
    print(f"\nEncoding complete! Total steps: {total_steps:,} (Train: {n_train:,}, Val: {n_val:,})")

    # Step 5: Save binary files for each lane
    print("\n==========================================================================")
    print(" STEP 4: Writing train.bin and val.bin for All Lanes")
    print("==========================================================================")
    # Map lane names
    lane_dest_dirs = [out_root / name for name in tok_coarse.lane_names]  # 24 lanes
    lane_dest_dirs.append(out_root / "pos_full")  # 25th is pos_full

    for i, dest_dir in enumerate(lane_dest_dirs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        lane_arr = np.array(all_lane_tokens[i], dtype=np.uint16)
        train_data = lane_arr[:n_train]
        val_data = lane_arr[n_train:]

        train_data.tofile(dest_dir / "train.bin")
        val_data.tofile(dest_dir / "val.bin")
        print(f"Wrote {dest_dir.name}: train={len(train_data):,} tokens, val={len(val_data):,} tokens")

    print("\n==========================================================================")
    print(" All 24 Dataset Lanes Successfully Prepared!")
    print(f" Output directory: {out_root}")
    print("==========================================================================")


if __name__ == "__main__":
    main()

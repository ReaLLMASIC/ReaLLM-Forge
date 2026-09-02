#!/usr/bin/env python3
"""Data preparation script for English-to-Korean translation benchmark.

Downloads Moo/korean-parallel-corpora, applies filtering (<= 100 syllables on Korean),
creates train (rest), val (5k), test (5k) splits, and trains English SentencePiece BPE.
"""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.request
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

DEFAULT_URL = "https://huggingface.co/datasets/Moo/korean-parallel-corpora/resolve/main/train.csv?download=true"


def download_if_needed(url: str, dest_path: Path) -> Path:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"File already exists at {dest_path}")
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading parallel corpus from {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print("Download complete.")
    return dest_path


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Normalize whitespaces
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(
    csv_path: Path,
    output_dir: Path,
    max_ko_len: int = 100,
    max_en_len: int = 500,
    val_size: int = 5000,
    test_size: int = 5000,
    en_vocab_size: int = 30000,
    random_seed: int = 42,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, keep_default_na=False)

    print(f"Initial raw pairs: {len(df)}")
    df["en"] = df["en"].apply(clean_text)
    df["ko"] = df["ko"].apply(clean_text)

    # Filter according to EACL 2023 criteria
    valid_mask = (
        (df["en"].str.len() > 0)
        & (df["ko"].str.len() > 0)
        & (df["ko"].str.len() <= max_ko_len)
        & (df["en"].str.len() <= max_en_len)
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"Filtered pairs (ko <= {max_ko_len} chars): {len(df)}")

    total_holdout = val_size + test_size
    if len(df) <= total_holdout:
        raise ValueError(f"Dataset too small ({len(df)}) for {total_holdout} holdout.")

    # Shuffle deterministically
    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    test_df = shuffled.iloc[val_size : val_size + test_size].reset_index(drop=True)
    train_df = shuffled.iloc[val_size + test_size :].reset_index(drop=True)

    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save splits as JSONL
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_file = output_dir / f"{name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                f.write(json.dumps({"en": row["en"], "ko": row["ko"]}, ensure_ascii=False) + "\n")
        print(f"Saved {name} split ({len(split_df)} pairs) to {out_file}")

    # Train English SentencePiece model on train split
    en_corpus_path = output_dir / "en_train_raw.txt"
    with open(en_corpus_path, "w", encoding="utf-8") as f:
        for en in train_df["en"]:
            f.write(en + "\n")

    spm_prefix = output_dir / "spm_en_30k"
    print(f"Training English SentencePiece BPE (vocab_size={en_vocab_size})...")
    # Determine safe vocab size
    with open(en_corpus_path, "r", encoding="utf-8") as f:
        unique_tokens = len(set(f.read().split()))
    actual_vocab_size = min(en_vocab_size, max(4000, unique_tokens))
    print(f"Target vocab size: {actual_vocab_size}")

    spm.SentencePieceTrainer.train(
        input=str(en_corpus_path),
        model_prefix=str(spm_prefix),
        vocab_size=actual_vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )
    print(f"Trained SentencePiece model saved to {spm_prefix}.model")


def main():
    parser = argparse.ArgumentParser(description="Prepare parallel corpus and English BPE.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--data_dir", type=str, default="data/korean_seq2seq_bench")
    parser.add_argument("--max_ko_len", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--en_vocab_size", type=int, default=30000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "train.csv"
    download_if_needed(args.url, csv_path)
    prepare_dataset(
        csv_path=csv_path,
        output_dir=data_dir,
        max_ko_len=args.max_ko_len,
        val_size=args.val_size,
        test_size=args.test_size,
        en_vocab_size=args.en_vocab_size,
    )


if __name__ == "__main__":
    main()

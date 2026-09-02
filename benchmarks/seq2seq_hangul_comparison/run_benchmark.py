#!/usr/bin/env python3
"""End-to-end benchmark runner comparing Three-Hot Tokenizer vs Hangul Factorizer.

Orchestrates:
1. Data preparation: Downloads Moo/korean-parallel-corpora and trains English BPE.
2. Training:
   - Model 1: Seq2SeqThreeHotConditional (EACL 2023 3-step unrolled RNN decoder)
   - Model 2: Seq2SeqThreeHotIndependent (Song et al. 3 independent heads)
   - Model 3: Seq2SeqHangulFactorizer (23-lane articulatory factorizer)
3. Evaluation: Computes BPJ, BLEU, chrF on the test set.
4. Summary Report: Prints formatted comparison table and saves to results.json / benchmark_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from benchmarks.seq2seq_hangul_comparison.prepare_data import (
    download_if_needed,
    prepare_dataset,
    DEFAULT_URL,
)
from benchmarks.seq2seq_hangul_comparison.tokenizers import (
    EnglishBPETokenizer,
    ThreeHotSeq2SeqTokenizer,
    HangulFactorizerSeq2SeqTokenizer,
    build_non_korean_vocab,
)
from benchmarks.seq2seq_hangul_comparison.models import (
    Seq2SeqThreeHotConditional,
    Seq2SeqThreeHotIndependent,
    Seq2SeqHangulFactorizer,
)
from benchmarks.seq2seq_hangul_comparison.train import (
    train_seq2seq,
    ParallelTranslationDataset,
    make_collate_fn,
)
from benchmarks.seq2seq_hangul_comparison.evaluate import evaluate_model


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_train_samples: int | None = None,
    max_val_samples: int = 500,
    max_test_samples: int = 1000,
    skip_prep: bool = False,
):
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data
    if not skip_prep and not (data_dir / "spm_en_30k.model").exists():
        print("=== Step 1: Preparing Dataset Splits and English BPE ===")
        csv_path = data_dir / "train.csv"
        download_if_needed(DEFAULT_URL, csv_path)
        prepare_dataset(
            csv_path=csv_path,
            output_dir=data_dir,
            max_ko_len=100,
            val_size=5000,
            test_size=5000,
            en_vocab_size=30000,
        )
    else:
        print("=== Step 1: Data preparation already done. Skipping. ===")

    architectures = [
        ("three_hot_conditional", "Three-Hot (Conditional RNN, EACL 2023)"),
        ("three_hot_independent", "Three-Hot (Independent Heads, Song et al.)"),
        ("hangul_factorizer", "Hangul Factorizer (23-Lane Multi-Head)"),
    ]

    checkpoints: Dict[str, Path] = {}

    # 2. Train Models
    print("\n=== Step 2: Training Architectures ===")
    for arch_key, arch_name in architectures:
        print(f"\n>>> Training {arch_name} ...")
        ckpt_path = output_dir / f"{arch_key}_best.pt"
        if ckpt_path.exists():
            print(f"Found existing checkpoint at {ckpt_path}. Skipping training.")
        else:
            ckpt_path = train_seq2seq(
                arch=arch_key,
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
            )
        checkpoints[arch_key] = ckpt_path

    # 3. Final Evaluation on Held-out Test Set
    print("\n=== Step 3: Evaluating Models on Test Set ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spm_path = data_dir / "spm_en_30k.model"
    src_tok = EnglishBPETokenizer(spm_path)
    test_jsonl = data_dir / "test.jsonl"

    results: List[Dict[str, Any]] = []

    for arch_key, arch_name in architectures:
        ckpt_path = checkpoints[arch_key]
        print(f"\nEvaluating {arch_name} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        non_ko_vocab = ckpt["non_ko_vocab"]
        tgt_vocab_sizes = ckpt["tgt_vocab_sizes"]

        is_factorizer = (arch_key == "hangul_factorizer")
        is_conditional = (arch_key == "three_hot_conditional")

        if is_factorizer:
            tgt_tok = HangulFactorizerSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqHangulFactorizer(
                src_vocab_size=src_tok.vocab_size,
                tgt_vocab_sizes=tgt_vocab_sizes,
            ).to(device)
            num_lanes = len(tgt_vocab_sizes)
        elif arch_key == "three_hot_conditional":
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotConditional(
                src_vocab_size=src_tok.vocab_size,
                tgt_vocab_sizes=tgt_vocab_sizes,
            ).to(device)
            num_lanes = 3
        else:
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotIndependent(
                src_vocab_size=src_tok.vocab_size,
                tgt_vocab_sizes=tgt_vocab_sizes,
            ).to(device)
            num_lanes = 3

        model.load_state_dict(ckpt["model_state_dict"])
        total_params = sum(p.numel() for p in model.parameters())

        test_dataset = ParallelTranslationDataset(
            test_jsonl, src_tok, tgt_tok, max_samples=max_test_samples
        )
        collate = make_collate_fn(num_lanes)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2
        )

        t0 = time.time()
        test_metrics = evaluate_model(
            model=model,
            val_loader=test_loader,
            src_tokenizer=src_tok,
            tgt_tokenizer=tgt_tok,
            device=device,
            is_factorizer=is_factorizer,
            is_conditional=is_conditional,
            max_eval_samples=max_test_samples,
        )
        elapsed = time.time() - t0

        result_row = {
            "Architecture": arch_name,
            "Key": arch_key,
            "Parameters": total_params,
            "BPJ": round(test_metrics["BPJ"], 4),
            "BLEU": round(test_metrics["BLEU"], 2),
            "chrF": round(test_metrics["chrF"], 2),
            "Eval_Time_Sec": round(elapsed, 1),
            "Test_Samples": test_metrics["samples"],
        }
        results.append(result_row)
        print(f"Result -> BPJ: {result_row['BPJ']}, BLEU: {result_row['BLEU']}, chrF: {result_row['chrF']}")

    # 4. Generate Markdown Report
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    header = f"| {'Model Architecture':<45} | {'Params':<10} | {'BPJ (lower)':<12} | {'BLEU (higher)':<14} | {'chrF (higher)':<14} |"
    sep = f"|{'-'*47}|{'-'*12}|{'-'*14}|{'-'*16}|{'-'*16}|"
    print(header)
    print(sep)
    for r in results:
        row_str = f"| {r['Architecture']:<45} | {r['Parameters']:<10,} | {r['BPJ']:<12.4f} | {r['BLEU']:<14.2f} | {r['chrF']:<14.2f} |"
        print(row_str)
    print("=" * 80)

    # Save JSON & Markdown
    res_path = output_dir / "benchmark_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_report_path = output_dir / "benchmark_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Seq2Seq Benchmark: Hangul Factorizer vs Three-Hot Tokenizer\n\n")
        f.write(f"Dataset: `Moo/korean-parallel-corpora` (English-to-Korean translation)\n\n")
        f.write(f"{header}\n{sep}\n")
        for r in results:
            f.write(f"| {r['Architecture']} | {r['Parameters']:,} | {r['BPJ']:.4f} | {r['BLEU']:.2f} | {r['chrF']:.2f} |\n")
        f.write("\n\n## Metrics Explanation:\n")
        f.write("- **BPJ (Bits-Per-Jamo)**: Negative log-likelihood divided by (ln 2 * 3 jamos). Lower is better.\n")
        f.write("- **BLEU**: Word 4-gram BLEU computed on Compatibility Jamo canonicalized text. Higher is better.\n")
        f.write("- **chrF**: Character 18-gram F-score (6 syllables) on Compatibility Jamo text. Higher is better.\n")

    print(f"Reports saved to {res_path} and {md_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Seq2Seq Hangul benchmark.")
    parser.add_argument("--data_dir", type=str, default="data/korean_seq2seq_bench")
    parser.add_argument("--output_dir", type=str, default="out_seq2seq_bench")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--skip_prep", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        skip_prep=args.skip_prep,
    )


if __name__ == "__main__":
    main()

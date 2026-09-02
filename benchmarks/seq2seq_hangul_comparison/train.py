#!/usr/bin/env python3
"""Training script for English-to-Korean Seq2Seq models.

Trains Seq2SeqThreeHotConditional, Seq2SeqThreeHotIndependent, or Seq2SeqHangulFactorizer.
Supports saving checkpoints based on lowest validation BPJ.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
from benchmarks.seq2seq_hangul_comparison.evaluate import evaluate_model


class ParallelTranslationDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        src_tokenizer: EnglishBPETokenizer,
        tgt_tokenizer,
        max_samples: Optional[int] = None,
    ):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.samples.append((data["en"], data["ko"]))
                if max_samples and len(self.samples) >= max_samples:
                    break

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        en_text, ko_text = self.samples[idx]
        src_ids = self.src_tokenizer.encode(en_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(ko_text, add_special_tokens=True)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "en_text": en_text,
            "ref_text": ko_text,
        }


def make_collate_fn(num_tgt_lanes: int):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        src_list = [item["src"] for item in batch]
        tgt_list = [item["tgt"] for item in batch]
        ref_texts = [item["ref_text"] for item in batch]
        en_texts = [item["en_text"] for item in batch]

        # Pad src
        src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)

        # Pad tgt: tgt items have shape (T, num_lanes)
        max_tgt_len = max(t.size(0) for t in tgt_list)
        b_size = len(tgt_list)
        tgt_padded = torch.zeros((b_size, max_tgt_len, num_tgt_lanes), dtype=torch.long)
        for i, t in enumerate(tgt_list):
            tgt_padded[i, : t.size(0), :] = t

        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "ref_text": ref_texts,
            "en_text": en_texts,
        }

    return collate_fn


def safe_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0
) -> torch.Tensor:
    """Computes cross entropy without returning NaN when all targets are ignore_index."""
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)
    return nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    arch: str,
    aux_loss_weight: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        # Shift target for autoregressive prediction
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        optimizer.zero_grad()

        if arch == "hangul_factorizer":
            logits_list = model(src, tgt_input)
            loss = torch.zeros((), device=device, dtype=logits_list[0].dtype)
            
            # Lane 0: script / non-ko
            loss = loss + safe_cross_entropy(
                logits_list[0].transpose(1, 2), tgt_target[:, :, 0], ignore_index=0
            )
            # Lanes 1, 2, 3: primary choseong, jungseong, jongseong
            for k in (1, 2, 3):
                loss = loss + safe_cross_entropy(
                    logits_list[k].transpose(1, 2), tgt_target[:, :, k], ignore_index=0
                )
            # Lanes 4..22: auxiliary phonetic / articulatory lanes
            for k in range(4, len(logits_list)):
                aux_l = safe_cross_entropy(
                    logits_list[k].transpose(1, 2), tgt_target[:, :, k], ignore_index=0
                )
                loss = loss + aux_loss_weight * aux_l
        else:
            logits_i, logits_v, logits_f = model(src, tgt_input)
            loss_i = safe_cross_entropy(
                logits_i.transpose(1, 2), tgt_target[:, :, 0], ignore_index=0
            )
            loss_v = safe_cross_entropy(
                logits_v.transpose(1, 2), tgt_target[:, :, 1], ignore_index=0
            )
            loss_f = safe_cross_entropy(
                logits_f.transpose(1, 2), tgt_target[:, :, 2], ignore_index=0
            )
            loss = loss_i + loss_v + loss_f

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return total_loss / max(1, num_batches)


def train_seq2seq(
    arch: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 512,
    d_ff: int = 512,
    nhead: int = 8,
    num_layers: int = 6,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 500,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load English Tokenizer
    spm_path = data_dir / "spm_en_30k.model"
    if not spm_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {spm_path}. Run prepare_data.py first.")
    src_tok = EnglishBPETokenizer(spm_path)

    # 2. Build non-Korean vocab from training corpus
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"

    ko_corpus = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_train_samples and i >= max_train_samples:
                break
            ko_corpus.append(json.loads(line)["ko"])
    non_ko_vocab = build_non_korean_vocab(ko_corpus)

    # 3. Build Korean Tokenizer
    if arch == "hangul_factorizer":
        tgt_tok = HangulFactorizerSeq2SeqTokenizer(non_ko_vocab)
        num_lanes = len(tgt_tok.vocab_sizes)
        model = Seq2SeqHangulFactorizer(
            src_vocab_size=src_tok.vocab_size,
            tgt_vocab_sizes=tgt_tok.vocab_sizes,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        ).to(device)
    elif arch == "three_hot_conditional":
        tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
        num_lanes = 3
        model = Seq2SeqThreeHotConditional(
            src_vocab_size=src_tok.vocab_size,
            tgt_vocab_sizes=tgt_tok.vocab_sizes,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        ).to(device)
    elif arch == "three_hot_independent":
        tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
        num_lanes = 3
        model = Seq2SeqThreeHotIndependent(
            src_vocab_size=src_tok.vocab_size,
            tgt_vocab_sizes=tgt_tok.vocab_sizes,
            d_model=d_model,
            d_ff=d_ff,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {arch} with {param_count:,} trainable parameters.")

    # 4. Datasets and Loaders
    train_dataset = ParallelTranslationDataset(
        train_jsonl, src_tok, tgt_tok, max_samples=max_train_samples
    )
    val_dataset = ParallelTranslationDataset(
        val_jsonl, src_tok, tgt_tok, max_samples=max_val_samples
    )

    collate = make_collate_fn(num_lanes)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    best_bpj = float("inf")
    best_checkpoint_path = output_dir / f"{arch}_best.pt"
    training_log = []

    is_factorizer = (arch == "hangul_factorizer")
    is_conditional = (arch == "three_hot_conditional")

    print(f"\n--- Starting Training for {arch} ({epochs} Epochs) ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, arch)

        # Validate BPJ and generation metrics
        val_metrics = evaluate_model(
            model,
            val_loader,
            src_tok,
            tgt_tok,
            device=device,
            is_factorizer=is_factorizer,
            is_conditional=is_conditional,
            max_eval_samples=max_val_samples,
        )

        ep_time = time.time() - ep_start
        val_bpj = val_metrics["BPJ"]
        val_bleu = val_metrics["BLEU"]
        val_chrf = val_metrics["chrF"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val BPJ: {val_bpj:.4f} | "
            f"Val BLEU: {val_bleu:.2f} | "
            f"Val chrF: {val_chrf:.2f}"
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_bpj": val_bpj,
            "val_bleu": val_bleu,
            "val_chrf": val_chrf,
            "epoch_seconds": ep_time,
        }
        training_log.append(log_entry)

        if val_bpj < best_bpj:
            best_bpj = val_bpj
            torch.save(
                {
                    "epoch": epoch,
                    "arch": arch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "tgt_vocab_sizes": tgt_tok.vocab_sizes,
                    "non_ko_vocab": non_ko_vocab,
                },
                best_checkpoint_path,
            )
            print(f"  -> Lowest Val BPJ achieved ({best_bpj:.4f}). Saved checkpoint to {best_checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\nFinished training {arch} in {total_time/60:.2f} minutes. Best Val BPJ: {best_bpj:.4f}")

    # Save training log
    log_path = output_dir / f"{arch}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq model on parallel corpus.")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["three_hot_conditional", "three_hot_independent", "hangul_factorizer"],
    )
    parser.add_argument("--data_dir", type=str, default="data/korean_seq2seq_bench")
    parser.add_argument("--output_dir", type=str, default="out_seq2seq_bench")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    args = parser.parse_args()

    train_seq2seq(
        arch=args.arch,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


if __name__ == "__main__":
    main()

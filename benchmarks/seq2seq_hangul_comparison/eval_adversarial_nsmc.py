#!/usr/bin/env python3
"""Adversarial Noise & Slang Benchmark on NSMC.

Evaluates Three-Hot Tokenizer vs. Hangul Factorizer on:
1. Uncorrupted NSMC dataset (real colloquial movie reviews with natural slang).
2. 80%-Corrupted NSMC dataset (severe adversarial typos, jamo swaps, and deletions).

Metrics:
- Bits-Per-Jamo (BPJ) on Uncorrupted vs. 80%-Corrupted.
- BPJ Sensitivity (Delta BPJ / degradation multiplier).
- Tokenization UNK Rate and Reversibility.
- Continuation chrF (prompting with first 50% of the sentence).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benchmarks.seq2seq_hangul_comparison.tokenizers import (
    EnglishBPETokenizer,
    ThreeHotSeq2SeqTokenizer,
    HangulFactorizerSeq2SeqTokenizer,
    S_BASE,
    S_COUNT,
    N_COUNT,
    T_COUNT,
    V_COUNT,
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
)
from benchmarks.seq2seq_hangul_comparison.models import (
    Seq2SeqThreeHotConditional,
    Seq2SeqThreeHotIndependent,
    Seq2SeqHangulFactorizer,
)
from benchmarks.seq2seq_hangul_comparison.evaluate import (
    canonicalize,
    calculate_bpj,
    evaluate_corpus_metrics,
)

NSMC_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

# Korean Dubeolsik keyboard adjacent key map for realistic typos
KEYBOARD_NEIGHBORS = {
    # Choseong / Consonants
    "ㄱ": ["ㅅ", "ㅈ", "ㅋ", "ㄲ"],
    "ㄴ": ["ㅇ", "ㄹ", "ㅁ", "ㄷ"],
    "ㄷ": ["ㄱ", "ㅈ", "ㅌ", "ㄸ"],
    "ㄹ": ["ㄴ", "ㅇ", "ㅎ"],
    "ㅁ": ["ㄴ", "ㅇ", "ㅋ"],
    "ㅂ": ["ㅈ", "ㅅ", "ㅃ", "ㅍ"],
    "ㅅ": ["ㄱ", "ㅂ", "ㅈ", "ㅆ"],
    "ㅇ": ["ㄴ", "ㄹ", "ㅎ"],
    "ㅈ": ["ㄷ", "ㄱ", "ㅂ", "ㅉ", "ㅊ"],
    "ㅊ": ["ㅈ", "ㅌ", "ㅋ"],
    "ㅋ": ["ㄱ", "ㅊ", "ㅌ"],
    "ㅌ": ["ㄷ", "ㅊ", "ㅋ"],
    "ㅍ": ["ㅂ", "ㅁ"],
    "ㅎ": ["ㅇ", "ㄹ", "ㅗ"],
    # Jungseong / Vowels
    "ㅏ": ["ㅑ", "ㅓ", "ㅗ", "ㅣ"],
    "ㅓ": ["ㅕ", "ㅏ", "ㅜ", "ㅡ"],
    "ㅗ": ["ㅛ", "ㅏ", "ㅜ"],
    "ㅜ": ["ㅠ", "ㅓ", "ㅡ"],
    "ㅡ": ["ㅜ", "ㅓ", "ㅣ"],
    "ㅣ": ["ㅏ", "ㅓ", "ㅡ"],
    "ㅐ": ["ㅔ", "ㅒ", "ㅑ"],
    "ㅔ": ["ㅐ", "ㅖ", "ㅕ"],
}


def download_nsmc(dest_path: Path) -> Path:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NSMC from {NSMC_URL}...")
    urllib.request.urlretrieve(NSMC_URL, dest_path)
    return dest_path


def load_nsmc_reviews(file_path: Path, max_samples: int = 1000) -> List[str]:
    reviews: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()  # id, document, label
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                doc = parts[1].strip()
                if 10 <= len(doc) <= 80:  # reasonable review length
                    reviews.append(doc)
            if len(reviews) >= max_samples * 2:
                break
    random.seed(42)
    random.shuffle(reviews)
    return reviews[:max_samples]


def corrupt_korean_text(text: str, corruption_rate: float = 0.8, seed: int = 42) -> str:
    """Applies an 80% corruption rate on Korean syllables."""
    rng = random.Random(seed)
    corrupted_chars = []

    for char in text:
        if rng.random() >= corruption_rate:
            corrupted_chars.append(char)
            continue

        # Check if modern Hangul syllable
        if len(char) == 1 and S_BASE <= ord(char) < S_BASE + S_COUNT:
            s = ord(char) - S_BASE
            l = s // N_COUNT
            v = (s % N_COUNT) // T_COUNT
            t = s % T_COUNT

            cho = CHOSEONG[l]
            jung = JUNGSEONG[v]
            jong = JONGSEONG[t] if t > 0 else ""

            # Adversarial mutation options
            mutation_type = rng.choice(["keyboard_typo", "batchim_drop_swap", "vowel_shift", "jamo_decompose"])

            if mutation_type == "keyboard_typo" and cho in KEYBOARD_NEIGHBORS:
                new_cho = rng.choice(KEYBOARD_NEIGHBORS[cho])
                if new_cho in CHOSEONG:
                    l = CHOSEONG.index(new_cho)
                new_char = chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)
                corrupted_chars.append(new_char)

            elif mutation_type == "vowel_shift" and jung in KEYBOARD_NEIGHBORS:
                new_jung = rng.choice(KEYBOARD_NEIGHBORS[jung])
                if new_jung in JUNGSEONG:
                    v = JUNGSEONG.index(new_jung)
                new_char = chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)
                corrupted_chars.append(new_char)

            elif mutation_type == "batchim_drop_swap":
                if t > 0 and rng.random() < 0.5:
                    t = 0  # drop batchim (common slang typo e.g. 있 -> 이)
                else:
                    t = rng.randint(0, len(JONGSEONG) - 1)
                new_char = chr(S_BASE + (l * V_COUNT + v) * T_COUNT + t)
                corrupted_chars.append(new_char)

            elif mutation_type == "jamo_decompose":
                # Slang decomposition e.g. 밥 -> ㅂㅏㅂ or ㅂㅏ
                corrupted_chars.extend([cho, jung] + ([jong] if jong else []))
            else:
                corrupted_chars.append(char)
        else:
            # Non-Korean character or space
            if char == " " and rng.random() < 0.5:
                continue  # drop space (common in internet reviews)
            elif rng.random() < 0.3:
                corrupted_chars.append(rng.choice(["ㅋ", "ㅎ", "ㅠ", "!", "?"]))
            else:
                corrupted_chars.append(char)

    return "".join(corrupted_chars)


class NSMCEvalDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        src_tokenizer: EnglishBPETokenizer,
        tgt_tokenizer: Any,
        dummy_en_prompt: str = "Review:",
    ):
        self.texts = texts
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.src_ids = src_tokenizer.encode(dummy_en_prompt, add_special_tokens=True)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        tgt_ids = self.tgt_tok.encode(text, add_special_tokens=True)
        return {
            "src_ids": torch.tensor(self.src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "text": text,
        }


def make_nsmc_collate(num_lanes: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        src_seqs = [b["src_ids"] for b in batch]
        tgt_seqs = [b["tgt_ids"] for b in batch]
        texts = [b["text"] for b in batch]

        src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)

        # Pad targets
        max_len = max(t.size(0) for t in tgt_seqs)
        b_size = len(tgt_seqs)

        if num_lanes == 3:
            tgt_padded = torch.zeros((b_size, max_len, 3), dtype=torch.long)
            for i, t in enumerate(tgt_seqs):
                tgt_padded[i, : t.size(0)] = t
        else:
            tgt_padded = torch.zeros((b_size, max_len, num_lanes), dtype=torch.long)
            for i, t in enumerate(tgt_seqs):
                tgt_padded[i, : t.size(0)] = t

        return {"src": src_padded, "tgt": tgt_padded, "text": texts}

    return collate


@torch.no_grad()
def evaluate_nsmc_bpj_and_completion(
    model: nn.Module,
    loader: DataLoader,
    src_tok: EnglishBPETokenizer,
    tgt_tok: Any,
    device: torch.device,
    is_factorizer: bool,
    is_conditional: bool,
    max_eval_samples: int = 500,
) -> Dict[str, float]:
    model.eval()
    total_jamo_nll = 0.0
    total_jamo_count = 0
    total_tokens = 0
    total_unks = 0

    hypotheses = []
    references = []

    samples = 0

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        texts = batch["text"]

        b_size = src.size(0)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        # Count UNKs in targets
        if is_factorizer:
            unk_mask = tgt_target[:, :, 0] == tgt_tok.unk_id
            total_unks += unk_mask.sum().item()
            total_tokens += (tgt_target[:, :, 0] != 0).sum().item()
        else:
            unk_mask = tgt_target[:, :, 0] == tgt_tok.unk_id
            total_unks += unk_mask.sum().item()
            total_tokens += (tgt_target[:, :, 0] != 0).sum().item()

        # Teacher-forced NLL & BPJ
        if is_factorizer:
            logits_list = model(src, tgt_input)
            ko_mask = tgt_target[:, :, 0] == 4  # Korean syllable flag
            if ko_mask.any():
                loss_i = nn.functional.cross_entropy(
                    logits_list[1].transpose(1, 2), tgt_target[:, :, 1], reduction="none"
                )[ko_mask].sum().item()
                loss_v = nn.functional.cross_entropy(
                    logits_list[2].transpose(1, 2), tgt_target[:, :, 2], reduction="none"
                )[ko_mask].sum().item()
                loss_f = nn.functional.cross_entropy(
                    logits_list[3].transpose(1, 2), tgt_target[:, :, 3], reduction="none"
                )[ko_mask].sum().item()
                num_syls = ko_mask.sum().item()
                total_jamo_nll += loss_i + loss_v + loss_f
                total_jamo_count += num_syls * 3
        else:
            logits_i, logits_v, logits_f = model(src, tgt_input)
            ko_mask = tgt_target[:, :, 1] != 0  # Jungseong is non-zero
            if ko_mask.any():
                loss_i = nn.functional.cross_entropy(
                    logits_i.transpose(1, 2), tgt_target[:, :, 0], reduction="none"
                )[ko_mask].sum().item()
                loss_v = nn.functional.cross_entropy(
                    logits_v.transpose(1, 2), tgt_target[:, :, 1], reduction="none"
                )[ko_mask].sum().item()
                loss_f = nn.functional.cross_entropy(
                    logits_f.transpose(1, 2), tgt_target[:, :, 2], reduction="none"
                )[ko_mask].sum().item()
                num_syls = ko_mask.sum().item()
                total_jamo_nll += loss_i + loss_v + loss_f
                total_jamo_count += num_syls * 3

        # Completion test on subset
        if samples < max_eval_samples:
            for text in texts:
                if samples >= max_eval_samples:
                    break
                # Prompt with first 50% of the sentence
                half_len = max(1, len(text) // 2)
                prompt_prefix = text[:half_len]
                ref_continuation = text[half_len:]
                if not ref_continuation.strip():
                    continue

                # Run completion
                if is_factorizer:
                    prompt_encoded = [[tgt_tok.sos_id] + [0] * 22] + tgt_tok.encode(
                        prompt_prefix, add_special_tokens=False
                    )
                    gen = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
                    memory = model.encode(src[:1])
                    for _ in range(len(ref_continuation) + 10):
                        emb = model.embed_target(gen)
                        mask = nn.Transformer.generate_square_subsequent_mask(
                            gen.size(1), device=device
                        )
                        h_t = model.decode_step(emb, memory, tgt_mask=mask)
                        last_h = h_t[:, -1, :]
                        preds = [head(last_h).argmax(dim=-1) for head in model.heads]
                        nxt = torch.stack(preds, dim=-1).unsqueeze(1)
                        gen = torch.cat([gen, nxt], dim=1)
                        if preds[0].item() == tgt_tok.eos_id:
                            break
                    completed_full = tgt_tok.decode(gen[0].tolist())
                    hyp_continuation = completed_full[len(prompt_prefix) :]
                elif is_conditional:
                    prompt_encoded = [(tgt_tok.sos_id, 0, 0)] + tgt_tok.encode(
                        prompt_prefix, add_special_tokens=False
                    )
                    gen = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
                    memory = model.encode(src[:1])
                    for _ in range(len(ref_continuation) + 10):
                        emb = model.embed_target(gen)
                        mask = nn.Transformer.generate_square_subsequent_mask(
                            gen.size(1), device=device
                        )
                        h_t = model.decode_step(emb, memory, tgt_mask=mask)
                        last_h = h_t[:, -1:, :]
                        h0 = model.h0.expand(1, 1, model.d_model)
                        h_i = torch.tanh(model.W_e(last_h) + model.W_h(h0))
                        pi_i = model.head_i(h_i).argmax(dim=-1)
                        embi = model.emb_i(pi_i)
                        h_v = torch.tanh(model.W_e(embi) + model.W_h(h_i))
                        pi_v = model.head_v(h_v).argmax(dim=-1)
                        embv = model.emb_v(pi_v)
                        h_f = torch.tanh(model.W_e(embv) + model.W_h(h_v))
                        pi_f = model.head_f(h_f).argmax(dim=-1)
                        nxt = torch.stack([pi_i, pi_v, pi_f], dim=-1)
                        gen = torch.cat([gen, nxt], dim=1)
                        if pi_i.item() == tgt_tok.eos_id:
                            break
                    completed_full = tgt_tok.decode(gen[0].tolist())
                    hyp_continuation = completed_full[len(prompt_prefix) :]
                else:
                    prompt_encoded = [(tgt_tok.sos_id, 0, 0)] + tgt_tok.encode(
                        prompt_prefix, add_special_tokens=False
                    )
                    gen = torch.tensor([prompt_encoded], dtype=torch.long, device=device)
                    memory = model.encode(src[:1])
                    for _ in range(len(ref_continuation) + 10):
                        emb = model.embed_target(gen)
                        mask = nn.Transformer.generate_square_subsequent_mask(
                            gen.size(1), device=device
                        )
                        h_t = model.decode_step(emb, memory, tgt_mask=mask)
                        last_h = h_t[:, -1, :]
                        pi_i = model.head_i(last_h).argmax(dim=-1)
                        pi_v = model.head_v(last_h).argmax(dim=-1)
                        pi_f = model.head_f(last_h).argmax(dim=-1)
                        nxt = torch.stack([pi_i, pi_v, pi_f], dim=-1).unsqueeze(1)
                        gen = torch.cat([gen, nxt], dim=1)
                        if pi_i.item() == tgt_tok.eos_id:
                            break
                    completed_full = tgt_tok.decode(gen[0].tolist())
                    hyp_continuation = completed_full[len(prompt_prefix) :]

                hypotheses.append(canonicalize(hyp_continuation))
                references.append(canonicalize(ref_continuation))
                samples += 1

    bpj = calculate_bpj(total_jamo_nll, total_jamo_count)
    unk_rate = (total_unks / max(1, total_tokens)) * 100.0
    metrics = evaluate_corpus_metrics(hypotheses, references)
    bleu = metrics.get("BLEU", 0.0)
    chrf = metrics.get("chrF", 0.0)

    return {
        "BPJ": bpj,
        "UNK_Rate": unk_rate,
        "BLEU": bleu,
        "chrF": chrf,
        "Samples": samples,
    }


def run_nsmc_benchmark(
    nsmc_path: Path,
    ckpt_dir: Path,
    num_samples: int = 500,
    batch_size: int = 32,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NSMC from {nsmc_path}...")
    raw_reviews = load_nsmc_reviews(nsmc_path, max_samples=num_samples)
    print(f"Sampled {len(raw_reviews)} reviews.")

    corrupted_reviews = [corrupt_korean_text(r, corruption_rate=0.8, seed=i) for i, r in enumerate(raw_reviews)]

    print("\n--- Example Uncorrupted vs. 80%-Corrupted NSMC Pairs ---")
    for i in range(3):
        print(f"[{i+1}] Uncorrupted: {raw_reviews[i]}")
        print(f"    Corrupted:   {corrupted_reviews[i]}\n")

    # Load models
    src_spm = REPO_ROOT / "data/korean_seq2seq_bench/spm_en_30k.model"
    src_tok = EnglishBPETokenizer(src_spm)

    architectures = [
        ("three_hot_conditional", "Three-Hot (Conditional RNN, EACL 2023)", False, True),
        ("three_hot_independent", "Three-Hot (Independent Heads, Song et al.)", False, False),
        ("hangul_factorizer", "Hangul Factorizer (23-Lane Multi-Head)", True, False),
    ]

    benchmark_results: Dict[str, Any] = {}

    for arch_key, arch_name, is_factorizer, is_conditional in architectures:
        ckpt_path = ckpt_dir / f"{arch_key}_best.pt"
        print(f"\n=======================================================")
        print(f"Evaluating {arch_name} on NSMC...")
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        non_ko_vocab = ckpt["non_ko_vocab"]
        tgt_vocab_sizes = ckpt["tgt_vocab_sizes"]

        if is_factorizer:
            tgt_tok = HangulFactorizerSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqHangulFactorizer(src_tok.vocab_size, tgt_vocab_sizes).to(device)
            num_lanes = len(tgt_vocab_sizes)
        elif is_conditional:
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotConditional(src_tok.vocab_size, tgt_vocab_sizes).to(device)
            num_lanes = 3
        else:
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotIndependent(src_tok.vocab_size, tgt_vocab_sizes).to(device)
            num_lanes = 3

        model.load_state_dict(ckpt["model_state_dict"])
        collate_fn = make_nsmc_collate(num_lanes)

        # 1. Evaluate Uncorrupted
        ds_clean = NSMCEvalDataset(raw_reviews, src_tok, tgt_tok)
        loader_clean = DataLoader(ds_clean, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print("  Evaluating on Uncorrupted NSMC...")
        clean_res = evaluate_nsmc_bpj_and_completion(
            model, loader_clean, src_tok, tgt_tok, device, is_factorizer, is_conditional, max_eval_samples=50
        )

        # 2. Evaluate 80%-Corrupted
        ds_corrupt = NSMCEvalDataset(corrupted_reviews, src_tok, tgt_tok)
        loader_corrupt = DataLoader(ds_corrupt, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print("  Evaluating on 80%-Corrupted NSMC...")
        corrupt_res = evaluate_nsmc_bpj_and_completion(
            model, loader_corrupt, src_tok, tgt_tok, device, is_factorizer, is_conditional, max_eval_samples=50
        )

        delta_bpj = corrupt_res["BPJ"] - clean_res["BPJ"]
        bpj_ratio = corrupt_res["BPJ"] / max(0.001, clean_res["BPJ"])

        benchmark_results[arch_key] = {
            "name": arch_name,
            "clean_bpj": round(clean_res["BPJ"], 4),
            "corrupt_bpj": round(corrupt_res["BPJ"], 4),
            "delta_bpj": round(delta_bpj, 4),
            "bpj_degradation_ratio": round(bpj_ratio, 2),
            "clean_unk_rate": round(clean_res["UNK_Rate"], 2),
            "corrupt_unk_rate": round(corrupt_res["UNK_Rate"], 2),
            "clean_chrf": round(clean_res["chrF"], 2),
            "corrupt_chrf": round(corrupt_res["chrF"], 2),
        }

        print(f"  -> Clean BPJ: {clean_res['BPJ']:.4f} | Corrupt BPJ: {corrupt_res['BPJ']:.4f} (Δ={delta_bpj:+.2f}, x{bpj_ratio:.2f})")
        print(f"  -> UNK Rate: {clean_res['UNK_Rate']:.1f}% -> {corrupt_res['UNK_Rate']:.1f}%")
        print(f"  -> Completion chrF: {clean_res['chrF']:.2f} -> {corrupt_res['chrF']:.2f}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Uncorrupted vs 80%-Corrupted NSMC.")
    parser.add_argument("--nsmc_path", type=str, default="data/nsmc_test.txt")
    parser.add_argument("--ckpt_dir", type=str, default="out_seq2seq_bench")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_json", type=str, default="out_seq2seq_bench/adversarial_nsmc_results.json")
    args = parser.parse_args()

    download_nsmc(Path(args.nsmc_path))
    results = run_nsmc_benchmark(
        Path(args.nsmc_path),
        Path(args.ckpt_dir),
        num_samples=args.num_samples,
    )

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_p}")


if __name__ == "__main__":
    main()

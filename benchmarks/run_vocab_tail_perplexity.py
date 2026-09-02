#!/usr/bin/env python3
"""Experiment: The 'Vocabulary Tail' Perplexity Test (Zero-Shot).

Buckets all 11,172 modern Hangul syllables into 10 deciles based on training corpus frequency:
- Decile 1: Top 10% most frequent
- Decile 10: Bottom 10% (rarely/never seen in training)

Evaluates Cross-Entropy Loss and Perplexity specifically for generating target syllables in each decile.
Compares Multicontext (HangulPosFactorizedTokenizer) against Baseline on 3k trained checkpoints.
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from dataclasses import fields
from inspect import signature
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from variations.model_variations import model_variation_dictionary

# Import HangulPosFactorizedTokenizer
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))
try:
    from hangul_factorizer import HangulPosFactorizedTokenizer
except ImportError:
    HangulPosFactorizedTokenizer = None

DEFAULT_MULTICONTEXT_LANES = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness",
    "korean_pos_mc/jung_round", "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim",
    "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod", "korean_pos_mc/pos", "korean_pos_mc/char"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vocabulary Tail Perplexity Test")
    parser.add_argument("--mc_dir", type=str, default="out_mc_korean_pos", help="Multicontext checkpoint dir")
    parser.add_argument("--base_dir", type=str, default="out_baseline_korean_pos", help="Baseline checkpoint dir")
    parser.add_argument("--mc_ckpt", type=str, default=None, help="Explicit path to Multicontext ckpt.pt")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Explicit path to Baseline ckpt.pt")
    parser.add_argument("--train_corpus", type=str, default="data/korean_pos_mc/input.txt", help="Training text file for decile bucketing")
    parser.add_argument("--dataset_name", type=str, default="KETI-AIR/kor_hellaswag", help="Evaluation dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--max_examples", type=int, default=500, help="Max evaluation examples (0 for full)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--sample_syllables_per_decile", type=int, default=50, help="Number of representative syllables to evaluate per decile (0 for all)")
    parser.add_argument("--output_json", type=str, default="vocab_tail_perplexity_results.json", help="Output JSON")
    return parser.parse_args()


def build_syllable_deciles(corpus_path: str) -> Tuple[Dict[str, int], Dict[int, Dict[str, Any]]]:
    """Bucket all 11,172 modern Hangul syllables into 10 deciles by training corpus frequency."""
    counts = Counter()
    if os.path.exists(corpus_path):
        print(f"Counting Hangul syllable frequencies in corpus: {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                for c in line:
                    if 0xAC00 <= ord(c) <= 0xD7A3:
                        counts[c] += 1
    else:
        print(f"Warning: Corpus file {corpus_path} not found. Deciles will be unweighted.")

    all_hangul = [chr(code) for code in range(0xAC00, 0xD7A3 + 1)]
    all_hangul.sort(key=lambda c: counts[c], reverse=True)

    total_syllables = len(all_hangul)
    decile_size = total_syllables // 10

    char_to_decile = {}
    decile_meta = {}

    for d in range(10):
        decile_num = d + 1
        start = d * decile_size
        end = (d + 1) * decile_size if d < 9 else total_syllables
        syllables_in_decile = all_hangul[start:end]

        for c in syllables_in_decile:
            char_to_decile[c] = decile_num

        freqs = [counts[c] for c in syllables_in_decile]
        decile_meta[decile_num] = {
            "num_syllables": len(syllables_in_decile),
            "max_freq": max(freqs) if freqs else 0,
            "min_freq": min(freqs) if freqs else 0,
            "zero_freq_count": freqs.count(0),
            "sample_syllables": syllables_in_decile[:5],
        }

    return char_to_decile, decile_meta


def _load_checkpoint(out_dir: str, device: str, ckpt_path_override: str = None):
    ckpt_path = ckpt_path_override or os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_kwargs = {"map_location": device}
    if "weights_only" in signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(ckpt_path, **load_kwargs)
    checkpoint_config = checkpoint.get("config", {})

    model_args = checkpoint.get("model_args", {})
    allowed_keys = {field.name for field in fields(GPTConfig)}
    model_args = {k: v for k, v in model_args.items() if k in allowed_keys}
    model_args["dropout"] = 0.0

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return model, checkpoint_config


class MulticontextEncoder:
    def __init__(self, lane_paths: List[str]):
        self.is_hybrid = any("korean_pos_mc_nochar" in lp for lp in lane_paths)
        pos_mode = "full" if any("pos_full" in lp for lp in lane_paths) else "coarse"
        if self.is_hybrid:
            from hangul_pos_hybrid_tokenizer import HangulHybridPosTokenizer
            spm_path = os.path.join(REPO_ROOT, "data", "korean_pos_mc_nochar", "spm_non_korean.model")
            self.tok = HangulHybridPosTokenizer(spm_path, use_pos=True, pos_mode=pos_mode)
        else:
            if HangulPosFactorizedTokenizer is None:
                raise ImportError("HangulPosFactorizedTokenizer not available.")
            self.tok = HangulPosFactorizedTokenizer(use_pos=True, pos_mode=pos_mode)

        self.lane_stois = []
        for lane_path in lane_paths:
            meta_path = os.path.join(REPO_ROOT, "data", lane_path, "meta.pkl")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(lane_path, "meta.pkl")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"meta.pkl not found: {lane_path}")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.lane_stois.append(meta["stoi"])

    def encode(self, text: str) -> List[List[int]]:
        if self.is_hybrid:
            steps = self.tok.encode_text(text)
            n_lanes = len(self.lane_stois)
            token_lists = [[] for _ in range(n_lanes)]
            for step in steps:
                for i in range(min(n_lanes, len(step))):
                    token_lists[i].append(step[i])
            return token_lists
        else:
            encoded_seq = self.tok.encode_text(text)
            token_lists = [[] for _ in range(25)]
            for item in encoded_seq:
                ch = item["char"]
                indices = item["indices"]
                for i in range(24):
                    token_char = self.tok.token_for(i, indices[i])
                    token_lists[i].append(self.lane_stois[i].get(token_char, 0))
                byte_id = self.lane_stois[24].get(ch, ch.encode("utf-8")[0] if len(ch) > 0 and ord(ch) > 255 else 0)
                token_lists[24].append(byte_id)
            return token_lists


def _load_encoder(out_dir: str, config: dict, model: GPT):
    is_mc = getattr(model.config, "multicontext", False)
    if is_mc:
        lanes = config.get("multicontext_datasets") or DEFAULT_MULTICONTEXT_LANES
        return True, MulticontextEncoder(lanes)

    meta_path = os.path.join(out_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        meta_path = os.path.join("data", "korean_pos_mc", "char", "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    encode_fn, _ = get_tokenizer_functions(meta)
    return False, encode_fn


def extract_texts_from_example(example: dict) -> List[str]:
    texts = []
    if "document" in example and str(example["document"]).strip():
        texts.append(str(example["document"]).strip())
    if "ctx" in example and example["ctx"]:
        texts.append(example["ctx"].strip())
    if "endings" in example:
        for ending in example["endings"]:
            if ending:
                texts.append(ending.strip())
    if "text" in example and example["text"]:
        texts.append(example["text"].strip())
    return texts


def _safe_encode(encode_fn, text):
    try:
        return encode_fn(text)
    except KeyError:
        tokens = []
        for c in text:
            try:
                tokens.extend(encode_fn(c))
            except KeyError:
                tokens.append(0)
        return tokens


def evaluate_decile_perplexity_baseline(
    model: GPT,
    encode_fn,
    texts: List[str],
    char_to_decile: Dict[str, int],
    device: str,
) -> Dict[int, List[float]]:
    block_size = model.config.block_size
    decile_losses: Dict[int, List[float]] = {d: [] for d in range(1, 11)}

    for text in texts:
        tokens = _safe_encode(encode_fn, text)
        if len(tokens) < 2:
            continue
        if len(tokens) > block_size:
            tokens = tokens[:block_size]

        input_ids = torch.tensor(tokens[:-1], device=device).unsqueeze(0)
        target_ids = torch.tensor(tokens[1:], device=device).unsqueeze(0)

        logits, _ = model(input_ids, target_ids)
        logprobs = torch.log_softmax(logits, dim=-1)

        target_slice = target_ids.squeeze(0)  # [L]
        target_logprobs = logprobs.squeeze(0).gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)  # [L]

        for idx in range(len(target_slice)):
            char_idx = idx + 1
            if char_idx < len(text):
                ch = text[char_idx]
                if ch in char_to_decile:
                    d = char_to_decile[ch]
                    lp = target_logprobs[idx].item()
                    decile_losses[d].append(-lp)

    return decile_losses


def evaluate_decile_perplexity_multicontext(
    model: GPT,
    mc_encoder: MulticontextEncoder,
    texts: List[str],
    char_to_decile: Dict[str, int],
    device: str,
) -> Dict[int, List[float]]:
    block_size = model.config.block_size
    decile_losses: Dict[int, List[float]] = {d: [] for d in range(1, 11)}

    for text in texts:
        full_lanes = mc_encoder.encode(text)
        seq_len = len(full_lanes[0])
        if seq_len < 2:
            continue
        if seq_len > block_size:
            full_lanes = [lane[:block_size] for lane in full_lanes]
            seq_len = len(full_lanes[0])

        input_lanes = [torch.tensor(lane[:-1], device=device).unsqueeze(0) for lane in full_lanes]
        target_lanes = [torch.tensor(lane[1:], device=device).unsqueeze(0) for lane in full_lanes]
        token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
        target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

        with torch.no_grad():
            logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)

        if getattr(mc_encoder, "is_hybrid", False):
            # Joint logprob across script, cho, jung, jong heads for Hangul
            l0_probs = torch.log_softmax(logits_list[0], dim=-1).squeeze(0).gather(-1, target_lanes[0].squeeze(0).unsqueeze(-1)).squeeze(-1)
            l1_probs = torch.log_softmax(logits_list[1], dim=-1).squeeze(0).gather(-1, target_lanes[1].squeeze(0).unsqueeze(-1)).squeeze(-1)
            l2_probs = torch.log_softmax(logits_list[2], dim=-1).squeeze(0).gather(-1, target_lanes[2].squeeze(0).unsqueeze(-1)).squeeze(-1)
            l3_probs = torch.log_softmax(logits_list[3], dim=-1).squeeze(0).gather(-1, target_lanes[3].squeeze(0).unsqueeze(-1)).squeeze(-1)

            # Sum of logprobs for the syllable
            target_logprobs = l0_probs + l1_probs + l2_probs + l3_probs
        else:
            char_logits = logits_list[-1]
            logprobs = torch.log_softmax(char_logits, dim=-1)
            target_char_ids = target_lanes[-1].squeeze(0)
            target_logprobs = logprobs.squeeze(0).gather(-1, target_char_ids.unsqueeze(-1)).squeeze(-1)

        # Map to deciles
        for idx in range(len(target_logprobs)):
            char_idx = idx + 1
            if char_idx < len(text):
                ch = text[char_idx]
                if ch in char_to_decile:
                    d = char_to_decile[ch]
                    lp = target_logprobs[idx].item()
                    decile_losses[d].append(-lp)

    return decile_losses


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("==========================================================================")
    print(" EXPERIMENT: 'VOCABULARY TAIL' PERPLEXITY TEST (ZERO-SHOT) ")
    print("==========================================================================")
    print(f"Device: {args.device}")
    print(f"Evaluation Dataset: {args.dataset_name} (split={args.split}, max_examples={args.max_examples})")

    # 1. Build Syllable Deciles based on training corpus
    char_to_decile, decile_meta = build_syllable_deciles(args.train_corpus)

    print("\n--- Training Frequency Decile Distribution ---")
    for d in range(1, 11):
        m = decile_meta[d]
        sample_str = ", ".join(m["sample_syllables"])
        print(f"Decile {d:2d}: max_freq={m['max_freq']:6d}, min_freq={m['min_freq']:5d}, zero_count={m['zero_freq_count']:4d} | Samples: {sample_str}")

    # 2. Load Models
    print("\n[1/3] Loading Multicontext (HangulPosFactorizedTokenizer) model...")
    mc_model, mc_config = _load_checkpoint(args.mc_dir, args.device, ckpt_path_override=args.mc_ckpt)
    mc_model.eval().to(args.device)
    _, mc_encoder = _load_encoder(args.mc_dir, mc_config, mc_model)

    print("[2/3] Loading Baseline (Single-Context) model...")
    base_model, base_config = _load_checkpoint(args.base_dir, args.device, ckpt_path_override=args.base_ckpt)
    base_model.eval().to(args.device)
    _, base_tokenizer = _load_encoder(args.base_dir, base_config, base_model)

    # 3. Load Evaluation Texts
    print(f"\n[3/3] Loading dataset split '{args.split}'...")
    raw_ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_examples > 0:
        ds = raw_ds.shuffle(seed=args.seed).select(range(min(args.max_examples, len(raw_ds))))
    else:
        ds = raw_ds

    all_texts = []
    for example in ds:
        all_texts.extend(extract_texts_from_example(example))

    # Construct Zero-Shot Syllable Prompt Texts
    sample_per_decile = getattr(args, "sample_syllables_per_decile", 50)
    if sample_per_decile > 0:
        decile_to_chars = {}
        for c, d in char_to_decile.items():
            decile_to_chars.setdefault(d, []).append(c)
        rng = random.Random(args.seed)
        sampled_chars = []
        for d in range(1, 11):
            chars_d = decile_to_chars.get(d, [])
            sampled_chars.extend(rng.sample(chars_d, min(sample_per_decile, len(chars_d))))
        print(f"Constructing Zero-Shot evaluation prompts for {len(sampled_chars)} representative Hangul syllables ({sample_per_decile}/decile)...")
        syllable_prompts = [f"글자: {c}" for c in sampled_chars]
    else:
        print("Constructing Zero-Shot evaluation prompts for all 11,172 modern Hangul syllables...")
        syllable_prompts = [f"글자: {c}" for c in char_to_decile.keys()]
    all_texts.extend(syllable_prompts)

    print(f"Extracted {len(all_texts)} total evaluation text blocks.")

    # 4. Evaluate Decile Losses
    print("\nEvaluating Baseline model across vocabulary deciles...")
    with torch.inference_mode():
        base_decile_losses = evaluate_decile_perplexity_baseline(
            base_model, base_tokenizer, all_texts, char_to_decile, args.device
        )

    print("Evaluating Multicontext (HangulPos) model across vocabulary deciles...")
    with torch.inference_mode():
        mc_decile_losses = evaluate_decile_perplexity_multicontext(
            mc_model, mc_encoder, all_texts, char_to_decile, args.device
        )


    # 5. Summarize Metrics
    summary_data = []

    print("\n=========================================================================================")
    print("                     VOCABULARY TAIL PERPLEXITY & LOSS COMPARISON                        ")
    print("=========================================================================================")
    print(f"{'Decile Bucket':<12} | {'Syllable Count':<14} | {'Baseline Loss':<14} | {'MC (HangulPos) Loss':<20} | {'MC Perplexity Gain':<18}")
    print("-" * 90)

    results_json = {
        "dataset": args.dataset_name,
        "split": args.split,
        "max_examples": args.max_examples,
        "deciles": {},
    }

    for d in range(1, 11):
        b_losses = base_decile_losses[d]
        m_losses = mc_decile_losses[d]

        b_loss = np.mean(b_losses) if len(b_losses) > 0 else float("nan")
        b_ppl = math.exp(b_loss) if not math.isnan(b_loss) else float("nan")

        m_loss = np.mean(m_losses) if len(m_losses) > 0 else float("nan")
        m_ppl = math.exp(m_loss) if not math.isnan(m_loss) else float("nan")

        loss_diff = m_loss - b_loss  # negative means MC has lower loss
        ppl_ratio = b_ppl / m_ppl if (not math.isnan(b_ppl) and not math.isnan(m_ppl) and m_ppl > 0) else float("nan")

        print(
            f"Decile {d:<5d} | {len(b_losses):<14d} | {b_loss:.4f} (PPL:{b_ppl:.1f}) | {m_loss:.4f} (PPL:{m_ppl:.1f})      | {ppl_ratio:.2f}x better"
        )

        results_json["deciles"][f"Decile_{d}"] = {
            "num_targets_evaluated": len(b_losses),
            "baseline_loss": b_loss,
            "baseline_perplexity": b_ppl,
            "multicontext_loss": m_loss,
            "multicontext_perplexity": m_ppl,
            "perplexity_ratio_gain": ppl_ratio,
        }

    print("=========================================================================================")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved decile perplexity results to {args.output_json}")


if __name__ == "__main__":
    main()

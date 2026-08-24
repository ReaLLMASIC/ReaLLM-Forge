#!/usr/bin/env python3
"""Experiment 1: Phonetic Slang & Typo Resilience (Robustness) Evaluation.

Evaluates HangulPosFactorizedTokenizer (Multicontext) against Baseline on 3k trained checkpoints
under Clean vs. Phonetically Corrupted Korean text.
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
from collections import Counter
from contextlib import nullcontext
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

# ==============================================================================
# Algorithmic Korean Phonetic Corruption Engine (Liaison, Tensification, Slang)
# ==============================================================================
S_BASE = 0xAC00
CHOSEONG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNGSEONG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

LIAISON_MAP = {
    1: (0, 0),    # ㄱ -> ㄱ
    2: (0, 1),    # ㄲ -> ㄲ
    3: (1, 9),    # ㄳ -> ㄱ + ㅅ
    4: (0, 2),    # ㄴ -> ㄴ
    5: (4, 12),   # ㄵ -> ㄴ + ㅈ
    6: (4, 11),   # ㄶ -> ㄴ + ㅎ(silent)
    7: (0, 3),    # ㄷ -> ㄷ
    8: (0, 5),    # ㄹ -> ㄹ
    9: (8, 0),    # ㄺ -> ㄹ + ㄱ
    10: (8, 6),   # ㄻ -> ㄹ + ㅁ
    11: (8, 7),   # ㄼ -> ㄹ + ㅂ
    12: (8, 9),   # ㄽ -> ㄹ + ㅅ
    13: (8, 16),  # ㄾ -> ㄹ + ㅌ
    14: (8, 17),  # ㄿ -> ㄹ + ㅍ
    15: (8, 11),  # ㅀ -> ㄹ + ㅎ(silent)
    16: (0, 6),   # ㅁ -> ㅁ
    17: (0, 7),   # ㅂ -> ㅂ
    18: (17, 9),  # ㅄ -> ㅂ + ㅅ
    19: (0, 9),   # ㅅ -> ㅅ
    20: (0, 10),  # ㅆ -> ㅆ
    22: (0, 12),  # ㅈ -> ㅈ
    23: (0, 14),  # ㅊ -> ㅊ
    24: (0, 15),  # ㅋ -> ㅋ
    25: (0, 16),  # ㅌ -> ㅌ
    26: (0, 17),  # ㅍ -> ㅍ
    27: (0, 11),  # ㅎ -> silent
}

TENSIFICATION_MAP = {
    0: 1,   # ㄱ -> ㄲ
    3: 4,   # ㄷ -> ㄸ
    7: 8,   # ㅂ -> ㅃ
    9: 10,  # ㅅ -> ㅆ
    12: 13, # ㅈ -> ㅉ
}

VOWEL_SHIFT_MAP = {
    4: 6,   # ㅓ -> ㅕ ("그래요" -> "그래여")
    0: 2,   # ㅏ -> ㅑ
    1: 3,   # ㅐ -> ㅒ
    5: 7,   # ㅔ -> ㅖ
}


def decompose_hangul(c: str) -> Tuple[int, int, int] | None:
    if len(c) == 1 and 0xAC00 <= ord(c) <= 0xD7A3:
        s = ord(c) - 0xAC00
        return s // (21 * 28), (s % (21 * 28)) // 28, s % 28
    return None


def recompose_hangul(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def corrupt_korean_text(text: str, corruption_rate: float = 0.8, seed: int = 42) -> str:
    rng = random.Random(seed)
    chars = list(text)
    n = len(chars)
    i = 0
    res = []

    while i < n:
        c1 = chars[i]
        d1 = decompose_hangul(c1)

        # 1. Phonetic Liaison (연음화): "먹었어요" -> "머거써요", "앉아" -> "안자"
        if d1 is not None and i + 1 < n and d1[2] != 0:
            c2 = chars[i + 1]
            d2 = decompose_hangul(c2)
            if d2 is not None and d2[0] == 11 and d1[2] in LIAISON_MAP:  # next choseong is 'ㅇ'
                if rng.random() < corruption_rate:
                    rem_jong, trans_cho = LIAISON_MAP[d1[2]]
                    new_c1 = recompose_hangul(d1[0], d1[1], rem_jong)
                    new_c2 = recompose_hangul(trans_cho, d2[1], d2[2])
                    res.append(new_c1)
                    res.append(new_c2)
                    i += 2
                    continue

        # 2. Tensification Slang / Vowel Shift: "좋아" -> "쬽아", "그래요" -> "그래여"
        if d1 is not None and rng.random() < (corruption_rate * 0.4):
            cho, jung, jong = d1
            if cho in TENSIFICATION_MAP and rng.random() < 0.6:
                cho = TENSIFICATION_MAP[cho]
            elif jung in VOWEL_SHIFT_MAP and rng.random() < 0.6:
                jung = VOWEL_SHIFT_MAP[jung]
            res.append(recompose_hangul(cho, jung, jong))
            i += 1
            continue

        res.append(c1)
        i += 1

    return "".join(res)


# ==============================================================================
# Model Loading & Evaluation Helpers
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phonetic Slang & Typo Resilience Evaluation")
    parser.add_argument("--mc_dir", type=str, default="out_mc_korean_pos", help="Multicontext checkpoint dir")
    parser.add_argument("--base_dir", type=str, default="out_baseline_korean_pos", help="Baseline checkpoint dir")
    parser.add_argument("--mc_ckpt", type=str, default=None, help="Explicit path to Multicontext ckpt.pt")
    parser.add_argument("--base_ckpt", type=str, default=None, help="Explicit path to Baseline ckpt.pt")
    parser.add_argument("--dataset_name", type=str, default="KETI-AIR/kor_hellaswag", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--max_examples", type=int, default=100, help="Max evaluation examples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--corruption_rate", type=float, default=0.8, help="Phonetic corruption probability")
    parser.add_argument("--output_json", type=str, default="phonetic_slang_resilience_results.json", help="Output JSON")
    return parser.parse_args()


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
        if HangulPosFactorizedTokenizer is None:
            raise ImportError("HangulPosFactorizedTokenizer not available.")
        pos_mode = "full" if any("pos_full" in lp for lp in lane_paths) else "coarse"
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


def _build_context(example: dict) -> str:
    ctx = example.get("ctx")
    if ctx:
        return ctx.strip()
    return (example.get("ctx_a", "") + " " + example.get("ctx_b", "")).strip()


def score_ending_baseline(model: GPT, encode, ctx_text: str, ending: str, block_size: int, device: str) -> float:
    ctx_tokens = encode(ctx_text)
    end_tokens = encode(ending)
    if len(end_tokens) == 0:
        return -math.inf
    if len(end_tokens) > block_size:
        end_tokens = end_tokens[-block_size:]
    max_ctx_len = max(0, block_size - len(end_tokens))
    ctx_trim = ctx_tokens[-max_ctx_len:] if max_ctx_len > 0 else []
    full = ctx_trim + end_tokens
    if len(full) < 2:
        return -math.inf

    input_ids = torch.tensor(full[:-1], device=device).unsqueeze(0)
    target_ids = torch.tensor(full[1:], device=device).unsqueeze(0)
    ending_start = max(len(ctx_trim) - 1, 0)

    logits, _ = model(input_ids, target_ids)
    logprobs = torch.log_softmax(logits, dim=-1)
    target_slice = target_ids[:, ending_start:]
    lp_cond = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
    return lp_cond.mean().item()


def score_ending_multicontext(model: GPT, encoder: MulticontextEncoder, ctx_text: str, ending: str, block_size: int, device: str) -> float:
    num_end_tokens = len(ending)
    if num_end_tokens == 0:
        return -math.inf
    full_text = ctx_text + ending
    full_lanes = encoder.encode(full_text)
    seq_len = len(full_lanes[0])
    if seq_len > block_size:
        full_lanes = [lane[-block_size:] for lane in full_lanes]
        seq_len = len(full_lanes[0])
    ctx_trim_len = max(0, seq_len - num_end_tokens)
    if seq_len < 2:
        return -math.inf

    input_lanes = [torch.tensor(lane[:-1], device=device).unsqueeze(0) for lane in full_lanes]
    target_lanes = [torch.tensor(lane[1:], device=device).unsqueeze(0) for lane in full_lanes]
    token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
    target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}
    ending_start = max(ctx_trim_len - 1, 0)

    logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)
    char_logits = logits_list[-1]
    logprobs = torch.log_softmax(char_logits, dim=-1)
    target_char_ids = target_lanes[-1][:, ending_start:]
    lp_cond = logprobs[:, ending_start:, :].gather(-1, target_char_ids.unsqueeze(-1)).squeeze(-1)
    return lp_cond.mean().item()


def _build_example_context_and_endings(example: dict) -> Tuple[str, List[str], int] | None:
    raw_label = example.get("label")
    if raw_label is None or raw_label == "":
        return None
    try:
        label = int(raw_label)
    except (ValueError, TypeError):
        return None

    if "document" in example:
        doc = str(example["document"]).strip()
        if not doc:
            return None
        ctx = f"리뷰: {doc} 평가:"
        endings = [" 부정", " 긍정"]
        return ctx, endings, label

    ctx = _build_context(example)
    endings = example.get("endings", [])
    if not endings:
        return None
    return ctx, endings, label



def evaluate_model_on_dataset(model: GPT, is_mc: bool, encoder_or_tok, dataset, is_corrupted: bool, corruption_rate: float, seed: int, device: str) -> Tuple[float, float]:
    block_size = model.config.block_size
    correct = 0
    total = 0
    total_logprob = 0.0

    for idx, example in enumerate(dataset):
        parsed = _build_example_context_and_endings(example)
        if parsed is None:
            continue
        clean_ctx, clean_endings, label = parsed

        if is_corrupted:
            ctx_text = corrupt_korean_text(clean_ctx, corruption_rate=corruption_rate, seed=seed + idx)
            endings = [corrupt_korean_text(e, corruption_rate=corruption_rate, seed=seed + idx + i * 1000) for i, e in enumerate(clean_endings)]
        else:
            ctx_text = clean_ctx
            endings = clean_endings

        scores = []
        for ending in endings:
            if is_mc:
                s = score_ending_multicontext(model, encoder_or_tok, ctx_text, ending, block_size, device)
            else:
                s = score_ending_baseline(model, encoder_or_tok, ctx_text, ending, block_size, device)
            scores.append(s)

        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        total += 1
        total_logprob += scores[label]

        if total % 100 == 0:
            print(f"  ...processed {total}/{len(dataset)} examples (current acc: {correct/total:.4f}, avg logprob: {total_logprob/total:.4f})")

    acc = correct / total if total > 0 else 0.0
    avg_logprob = total_logprob / total if total > 0 else 0.0
    return acc, avg_logprob



def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("==========================================================================")
    print(" EXPERIMENT 1: PHONETIC SLANG & TYPO RESILIENCE (ROBUSTNESS) EVALUATION ")
    print("==========================================================================")
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset_name} (split={args.split}, max_examples={args.max_examples})")
    print(f"Phonetic Corruption Rate: {args.corruption_rate}")

    # Load Multicontext Model (HangulPosFactorizedTokenizer)
    print("\n[1/4] Loading Multicontext (HangulPosFactorizedTokenizer) model checkpoint...")
    mc_model, mc_config = _load_checkpoint(args.mc_dir, args.device, ckpt_path_override=args.mc_ckpt)
    mc_model.eval().to(args.device)
    _, mc_encoder = _load_encoder(args.mc_dir, mc_config, mc_model)

    # Load Baseline Model
    print("[2/4] Loading Baseline (Single-Context) model checkpoint...")
    base_model, base_config = _load_checkpoint(args.base_dir, args.device, ckpt_path_override=args.base_ckpt)
    base_model.eval().to(args.device)
    _, base_tokenizer = _load_encoder(args.base_dir, base_config, base_model)

    # Load Evaluation Dataset
    print(f"\n[3/4] Loading and preparing dataset split '{args.split}'...")
    raw_ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_examples is not None and args.max_examples > 0:
        dataset = raw_ds.shuffle(seed=args.seed).select(range(min(args.max_examples, len(raw_ds))))
    else:
        dataset = raw_ds


    # Sample Demonstration of Corrupted Sentences
    print("\n--- Example Phonetic Slang & Typo Transformations ---")
    printed_count = 0
    for i in range(len(dataset)):
        parsed = _build_example_context_and_endings(dataset[i])
        if parsed is None:
            continue
        sample_ctx, _, _ = parsed
        corrupted_ctx = corrupt_korean_text(sample_ctx, corruption_rate=args.corruption_rate, seed=args.seed + i)
        printed_count += 1
        print(f"Original [{printed_count}]:  {sample_ctx[:80]}")
        print(f"Corrupted [{printed_count}]: {corrupted_ctx[:80]}\n")
        if printed_count >= 3:
            break


    # Evaluate Models on Clean vs Corrupted Test Sets
    print("[4/4] Evaluating models on Clean vs. Phonetically Corrupted Test Sets...")
    with torch.inference_mode():
        # Baseline Evaluation
        base_clean_acc, base_clean_lp = evaluate_model_on_dataset(
            base_model, False, base_tokenizer, dataset, is_corrupted=False, corruption_rate=args.corruption_rate, seed=args.seed, device=args.device
        )
        base_corr_acc, base_corr_lp = evaluate_model_on_dataset(
            base_model, False, base_tokenizer, dataset, is_corrupted=True, corruption_rate=args.corruption_rate, seed=args.seed, device=args.device
        )

        # Multicontext Evaluation
        mc_clean_acc, mc_clean_lp = evaluate_model_on_dataset(
            mc_model, True, mc_encoder, dataset, is_corrupted=False, corruption_rate=args.corruption_rate, seed=args.seed, device=args.device
        )
        mc_corr_acc, mc_corr_lp = evaluate_model_on_dataset(
            mc_model, True, mc_encoder, dataset, is_corrupted=True, corruption_rate=args.corruption_rate, seed=args.seed, device=args.device
        )

    # Compute Deltas (Performance Drop)
    base_acc_drop = base_clean_acc - base_corr_acc
    mc_acc_drop = mc_clean_acc - mc_corr_acc

    base_lp_drop = base_clean_lp - base_corr_lp
    mc_lp_drop = mc_clean_lp - mc_corr_lp

    print("=========================================================================================")
    print("                           EVALUATION SUMMARY & COMPARISON                               ")
    print("=========================================================================================")
    print(f"{'Model Architecture':<30} | {'Clean Acc':<10} | {'Corrupted Acc':<14} | {'Δ Acc Drop (lower=better)':<25}")
    print("-" * 88)
    print(f"{'Baseline (Single Context)':<30} | {base_clean_acc:.4f}     | {base_corr_acc:.4f}         | {base_acc_drop:+.4f}")
    print(f"{'Multicontext (HangulPos)':<30} | {mc_clean_acc:.4f}     | {mc_corr_acc:.4f}         | {mc_acc_drop:+.4f}")
    print("=" * 88)
    print(f"{'Model Architecture':<30} | {'Clean LogProb':<12} | {'Corrupted LogProb':<16} | {'Δ LogProb Degradation':<25}")
    print("-" * 88)
    print(f"{'Baseline (Single Context)':<30} | {base_clean_lp:.4f}       | {base_corr_lp:.4f}           | {base_lp_drop:+.4f}")
    print(f"{'Multicontext (HangulPos)':<30} | {mc_clean_lp:.4f}       | {mc_corr_lp:.4f}           | {mc_lp_drop:+.4f}")
    print("=" * 88)

    results = {
        "dataset": args.dataset_name,
        "split": args.split,
        "max_examples": args.max_examples,
        "corruption_rate": args.corruption_rate,
        "baseline": {
            "clean_accuracy": base_clean_acc,
            "corrupted_accuracy": base_corr_acc,
            "accuracy_drop": base_acc_drop,
            "clean_logprob": base_clean_lp,
            "corrupted_logprob": base_corr_lp,
            "logprob_degradation": base_lp_drop,
        },
        "multicontext": {
            "clean_accuracy": mc_clean_acc,
            "corrupted_accuracy": mc_corr_acc,
            "accuracy_drop": mc_acc_drop,
            "clean_logprob": mc_clean_lp,
            "corrupted_logprob": mc_corr_lp,
            "logprob_degradation": mc_lp_drop,
        },
        "robustness_gain_acc_delta": base_acc_drop - mc_acc_drop,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()

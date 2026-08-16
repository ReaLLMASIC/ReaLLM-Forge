#!/usr/bin/env python3
"""Run 4 Benchmark Capability Evaluations across Checkpoints:
1. Token-Level Extraction (KLUE-NER)
2. Syntactic Parsing (KLUE-DP)
3. Informal & Noisy Text (NSMC / UnSmile with 80% noise)
4. Rare Vocabulary / OOV (KorMedMCQA)
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
import pickle
import random
import sys
from inspect import signature
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from model import GPT, GPTConfig

try:
    from hangul_factorizer import HangulPosFactorizedTokenizer
except ImportError:
    HangulPosFactorizedTokenizer = None

DEFAULT_LANES = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
    "korean_pos_mc/pos", "korean_pos_mc/char"
]


def load_ckpt(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    kwargs = {"map_location": device}
    if "weights_only" in signature(torch.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch.load(ckpt_path, **kwargs)
    config = checkpoint.get("config", {})
    model_args = checkpoint.get("model_args", {})
    allowed = {field.name for field in fields(GPTConfig)}
    model_args = {k: v for k, v in model_args.items() if k in allowed}
    model_args["dropout"] = 0.0
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, config


class Encoder:
    def __init__(self, is_mc: bool, lane_paths: List[str] = None):
        self.is_mc = is_mc
        if is_mc:
            if HangulPosFactorizedTokenizer is None:
                raise ImportError("HangulPosFactorizedTokenizer not available.")
            lane_paths = lane_paths or DEFAULT_LANES
            pos_mode = "full" if any("pos_full" in lp for lp in lane_paths) else "coarse"
            self.tok = HangulPosFactorizedTokenizer(use_pos=True, pos_mode=pos_mode)
            self.stois = []
            for lp in lane_paths:
                mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
                if not os.path.exists(mp):
                    mp = os.path.join(lp, "meta.pkl")
                if not os.path.exists(mp):
                    mp = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "char", "meta.pkl")
                with open(mp, "rb") as f:
                    m = pickle.load(f)
                self.stois.append(m["stoi"])
        else:
            mp = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "char", "meta.pkl")
            if not os.path.exists(mp):
                mp = os.path.join(REPO_ROOT, "out_baseline_korean_pos", "meta.pkl")
            with open(mp, "rb") as f:
                m = pickle.load(f)
            self.stoi = m["stoi"]

    def encode(self, text: str):
        if self.is_mc:
            seq = self.tok.encode_text(text)
            n_lanes = len(self.stois)
            token_lists = [[] for _ in range(n_lanes)]
            for item in seq:
                ch = item["char"]
                indices = item["indices"]
                for i in range(min(24, n_lanes - 1)):
                    t_char = self.tok.token_for(i, indices[i])
                    token_lists[i].append(self.stois[i].get(t_char, 0))
                if n_lanes > 24:
                    byte_id = self.stois[24].get(ch, ch.encode("utf-8")[0] if len(ch) > 0 and ord(ch) > 255 else 0)
                    token_lists[24].append(byte_id)
            return token_lists
        else:
            return [self.stoi.get(c, 0) for c in text]


def eval_sequence_logprob(model: GPT, encoder: Encoder, text: str, device: str, block_size: int = 256) -> float:
    if not text:
        return 0.0
    if encoder.is_mc:
        full_lanes = encoder.encode(text)
        n_model_lanes = len(model.config.multicontext_datasets) if hasattr(model.config, "multicontext_datasets") and model.config.multicontext_datasets else len(full_lanes)
        if len(full_lanes) > n_model_lanes:
            full_lanes = full_lanes[:n_model_lanes]
        elif len(full_lanes) < n_model_lanes:
            full_lanes = full_lanes + [full_lanes[-1]] * (n_model_lanes - len(full_lanes))

        seq_len = len(full_lanes[0])
        if seq_len < 2:
            return 0.0
        if seq_len > block_size:
            full_lanes = [lane[:block_size] for lane in full_lanes]

        input_lanes = [torch.tensor(lane[:-1], device=device).unsqueeze(0) for lane in full_lanes]
        target_lanes = [torch.tensor(lane[1:], device=device).unsqueeze(0) for lane in full_lanes]

        vocab_sizes = getattr(model.config, "vocab_sizes", None)
        if vocab_sizes is not None and len(vocab_sizes) == len(input_lanes):
            for i in range(len(input_lanes)):
                input_lanes[i] = torch.clamp(input_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)
                target_lanes[i] = torch.clamp(target_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)

        token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
        target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}

        with torch.no_grad():
            logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)
            char_logits = logits_list[-1]
            log_probs = F.log_softmax(char_logits, dim=-1)
            target_ids = target_lanes[-1].squeeze(0)
            target_log_probs = log_probs.squeeze(0).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            return target_log_probs.mean().item()
    else:
        tokens = encoder.encode(text)
        seq_len = len(tokens)
        if seq_len < 2:
            return 0.0
        if seq_len > block_size:
            tokens = tokens[:block_size]

        vocab_size = model.config.vocab_size
        input_ids = torch.tensor(tokens[:-1], device=device, dtype=torch.long).unsqueeze(0)
        target_ids = torch.tensor(tokens[1:], device=device, dtype=torch.long).unsqueeze(0)
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1).to(torch.long)
        target_ids = torch.clamp(target_ids, 0, vocab_size - 1).to(torch.long)

        with torch.no_grad():
            logits, _ = model(input_ids, target_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs.squeeze(0).gather(-1, target_ids.squeeze(0).unsqueeze(-1)).squeeze(-1)
            return target_log_probs.mean().item()


def corrupt_text(text: str, rate: float = 0.8) -> str:
    res = []
    for c in text:
        if random.random() < rate:
            res.append(chr(ord(c) ^ 1))
        else:
            res.append(c)
    return "".join(res)


def eval_mc_qa(model: GPT, encoder: Encoder, question: str, options: List[str], correct_idx: int, device: str, block_size: int = 256) -> bool:
    scores = []
    for opt in options:
        prompt = f"질문: {question} 정답: {opt}"
        lp = eval_sequence_logprob(model, encoder, prompt, device, block_size)
        scores.append(lp)
    pred_idx = int(torch.argmax(torch.tensor(scores)).item())
    return pred_idx == correct_idx


def run_evaluations_for_ckpt(ckpt_name: str, ckpt_path: str, device: str, max_examples: int = 200) -> dict:
    print(f"\n==========================================================================")
    print(f" EVALUATING CHECKPOINT: {ckpt_name}")
    print(f" Path: {ckpt_path}")
    print(f"==========================================================================")

    model, config = load_ckpt(ckpt_path, device)
    is_mc = getattr(model.config, "multicontext", False)
    lanes = config.get("multicontext_datasets") or DEFAULT_LANES
    encoder = Encoder(is_mc=is_mc, lane_paths=lanes)

    results = {"checkpoint": ckpt_name, "is_multicontext": is_mc}

    # 1. Token-Level Extraction (KLUE-NER)
    print("[1/4] Category 1: Token-Level Extraction (KLUE-NER)...")
    try:
        ds_ner = load_dataset("klue", "ner", split="validation")
        lps = []
        for i, row in enumerate(ds_ner):
            if i >= max_examples:
                break
            sent = row.get("sentence", "")
            if sent:
                lp = eval_sequence_logprob(model, encoder, sent, device)
                lps.append(lp)
        avg_lp_ner = sum(lps) / max(len(lps), 1)
        ppl_ner = math.exp(-avg_lp_ner) if avg_lp_ner < 0 else float("nan")
        results["klue_ner"] = {"samples": len(lps), "avg_logprob": avg_lp_ner, "perplexity": ppl_ner}
        print(f"   KLUE-NER -> Avg LogProb: {avg_lp_ner:.4f}, Perplexity: {ppl_ner:.2f}")
    except Exception as e:
        print("   KLUE-NER evaluation failed:", e)

    # 2. Syntactic Parsing (KLUE-DP)
    print("[2/4] Category 2: Syntactic Parsing (KLUE-DP)...")
    try:
        ds_dp = load_dataset("klue", "dp", split="validation")
        lps = []
        for i, row in enumerate(ds_dp):
            if i >= max_examples:
                break
            sent = row.get("sentence", "")
            if sent:
                lp = eval_sequence_logprob(model, encoder, sent, device)
                lps.append(lp)
        avg_lp_dp = sum(lps) / max(len(lps), 1)
        ppl_dp = math.exp(-avg_lp_dp) if avg_lp_dp < 0 else float("nan")
        results["klue_dp"] = {"samples": len(lps), "avg_logprob": avg_lp_dp, "perplexity": ppl_dp}
        print(f"   KLUE-DP  -> Avg LogProb: {avg_lp_dp:.4f}, Perplexity: {ppl_dp:.2f}")
    except Exception as e:
        print("   KLUE-DP evaluation failed:", e)

    # 3. Informal & Noisy Text (NSMC & UnSmile with 80% noise)
    print("[3/4] Category 3: Informal & Noisy Text (NSMC & UnSmile @ 80% Noise)...")
    try:
        ds_nsmc = load_dataset("Blpeng/nsmc", split="test")
        ds_unsmile = load_dataset("smilegate-ai/kor_unsmile", split="valid")
        
        # NSMC Noise test
        nsmc_clean_lps, nsmc_corr_lps = [], []
        for i, row in enumerate(ds_nsmc):
            if i >= max_examples:
                break
            doc = row.get("document", "")
            if doc:
                clean_lp = eval_sequence_logprob(model, encoder, doc, device)
                corr_doc = corrupt_text(doc, rate=0.8)
                corr_lp = eval_sequence_logprob(model, encoder, corr_doc, device)
                nsmc_clean_lps.append(clean_lp)
                nsmc_corr_lps.append(corr_lp)
        avg_nsmc_clean = sum(nsmc_clean_lps) / max(len(nsmc_clean_lps), 1)
        avg_nsmc_corr = sum(nsmc_corr_lps) / max(len(nsmc_corr_lps), 1)
        nsmc_degr = avg_nsmc_clean - avg_nsmc_corr

        # UnSmile Noise test
        un_clean_lps, un_corr_lps = [], []
        for i, row in enumerate(ds_unsmile):
            if i >= max_examples:
                break
            text = row.get("문장", "")
            if text:
                clean_lp = eval_sequence_logprob(model, encoder, text, device)
                corr_text = corrupt_text(text, rate=0.8)
                corr_lp = eval_sequence_logprob(model, encoder, corr_text, device)
                un_clean_lps.append(clean_lp)
                un_corr_lps.append(corr_lp)
        avg_un_clean = sum(un_clean_lps) / max(len(un_clean_lps), 1)
        avg_un_corr = sum(un_corr_lps) / max(len(un_corr_lps), 1)
        un_degr = avg_un_clean - avg_un_corr

        results["noisy_text"] = {
            "nsmc": {"clean_logprob": avg_nsmc_clean, "corrupted_logprob": avg_nsmc_corr, "degradation": nsmc_degr},
            "unsmile": {"clean_logprob": avg_un_clean, "corrupted_logprob": avg_un_corr, "degradation": un_degr}
        }
        print(f"   NSMC @ 80% Noise -> Clean LogProb: {avg_nsmc_clean:.4f}, Corrupted: {avg_nsmc_corr:.4f}, Degradation: {nsmc_degr:.4f}")
        print(f"   UnSmile @ 80% Noise -> Clean LogProb: {avg_un_clean:.4f}, Corrupted: {avg_un_corr:.4f}, Degradation: {un_degr:.4f}")
    except Exception as e:
        print("   Informal & Noisy Text evaluation failed:", e)

    # 4. Rare Vocabulary / OOV (KorMedMCQA Medical QA)
    print("[4/4] Category 4: Rare Vocabulary / OOV (KorMedMCQA Medical QA)...")
    try:
        correct_cnt, total_cnt = 0, 0
        for subj in ["doctor", "dentist", "nurse", "pharm"]:
            ds_med = load_dataset("sean0042/KorMedMCQA", subj, split="test")
            for i, row in enumerate(ds_med):
                if total_cnt >= max_examples:
                    break
                q = row.get("question", "")
                opts = [row.get("A", ""), row.get("B", ""), row.get("C", ""), row.get("D", ""), row.get("E", "")]
                opts = [o for o in opts if o]
                raw_ans = row.get("answer")
                if q and opts and raw_ans is not None:
                    ans_idx = int(raw_ans) - 1
                    if 0 <= ans_idx < len(opts):
                        if eval_mc_qa(model, encoder, q, opts, ans_idx, device):
                            correct_cnt += 1
                        total_cnt += 1
        acc = (correct_cnt / total_cnt) if total_cnt > 0 else 0.0
        results["kormedqa"] = {"total_questions": total_cnt, "correct": correct_cnt, "accuracy": acc}
        print(f"   KorMedMCQA Zero-Shot Accuracy: {acc*100:.2f}% ({correct_cnt}/{total_cnt})")
    except Exception as e:
        print("   KorMedMCQA evaluation failed:", e)

    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate 4 Capability Categories on Checkpoints")
    p.add_argument("--ckpts", nargs="+", required=True, help="List of checkpoint names and paths in format name:path")
    p.add_argument("--device", default="cuda:0", help="Torch device")
    p.add_argument("--max_examples", type=int, default=200, help="Max examples per dataset")
    p.add_argument("--output_json", default="four_capability_eval_results.json", help="Output JSON path")
    args = p.parse_args()

    all_results = {}
    for item in args.ckpts:
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            name, path = os.path.basename(item), item
        res = run_evaluations_for_ckpt(name, path, args.device, max_examples=args.max_examples)
        all_results[name] = res

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll 4 capability evaluations completed and saved to {args.output_json}")


if __name__ == "__main__":
    main()

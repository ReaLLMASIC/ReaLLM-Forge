#!/usr/bin/env python3
"""Evaluate a trained checkpoint (Multicontext HangulPosFactorizedTokenizer or Baseline) on Korean HellaSwag.

Supports both:
1. Multicontext models trained with HangulPosFactorizedTokenizer (25 parallel lanes: 24 factor lanes + char lane).
2. Baseline single-context models trained on character tokens (korean_pos_mc/char).
"""
import argparse
import json
import math
import os
import pickle
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ko-HellaSwag evaluation on HangulPosFactorizedTokenizer or Baseline model")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory containing ckpt.pt and meta.pkl")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional explicit path to ckpt.pt")
    parser.add_argument("--config_path", type=str, default=None, help="Optional JSON config to supplement model_args")
    parser.add_argument("--init_from", type=str, default="resume", help="'resume' or a GPT-2 variant")
    parser.add_argument("--dataset_name", type=str, default="KETI-AIR/kor_hellaswag", help="HuggingFace dataset name for Ko-HellaSwag")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Autocast dtype")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap on number of examples")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
    parser.add_argument("--block_size", type=int, default=None, help="Override model block size")
    parser.add_argument("--length_norm", action=argparse.BooleanOptionalAction, default=True, help="Normalize by ending length")
    parser.add_argument("--prior_norm", action=argparse.BooleanOptionalAction, default=False, help="Enable prior (unconditional context) probability normalization")
    parser.add_argument("--unigram_norm", action=argparse.BooleanOptionalAction, default=False, help="Enable unigram character frequency probability normalization")
    parser.add_argument(
        "--norm_type",
        type=str,
        default=None,
        choices=["none", "length", "prior", "prior_length", "unigram", "unigram_length", "all"],
        help="Explicit normalization mode override (none, length, prior, prior_length, unigram, unigram_length, all)",
    )
    parser.add_argument("--eval_all_norms", default=False, action=argparse.BooleanOptionalAction, help="Evaluate all normalization modes in a single dataset pass")
    parser.add_argument("--weights_only", default=False, action=argparse.BooleanOptionalAction, help="Disable to allow full pickle loading for legacy checkpoints")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write metrics JSON")
    parser.add_argument(
        "--print_examples",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print correct and incorrect example predictions",
    )
    return parser.parse_args()


def _load_json_config(config_path: str) -> Tuple[dict, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object")

    if "model_args" in raw and isinstance(raw["model_args"], dict):
        return raw["model_args"], raw.get("config", raw)

    return raw, raw


def _load_checkpoint(args: argparse.Namespace):
    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    load_kwargs = {"map_location": args.device}
    if "weights_only" in signature(torch.load).parameters:
        load_kwargs["weights_only"] = args.weights_only

    checkpoint = torch.load(ckpt_path, **load_kwargs)
    checkpoint_config = checkpoint.get("config", {})

    config_override = None
    config_meta_hint = None
    if args.config_path:
        config_override, config_meta_hint = _load_json_config(args.config_path)
        if not checkpoint_config:
            checkpoint_config = config_meta_hint

    if args.init_from == "resume":
        model_args = {}
        if isinstance(checkpoint.get("model_args"), dict):
            model_args.update(checkpoint["model_args"])
        if config_override:
            model_args.update(config_override)
        if not model_args:
            raise ValueError("No model_args found in checkpoint; provide --config_path.")

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
    else:
        gptconf = GPTConfig()
        variation_dict = model_variation_dictionary[args.init_from]
        for k, v in variation_dict.items():
            setattr(gptconf, k, v)
        model = GPT.from_pretrained(gptconf, model_type=args.init_from)

    return model, checkpoint_config


class MulticontextHangulPosEncoder:
    """Encoder for multicontext models with HangulPosFactorizedTokenizer (25 lanes)."""

    def __init__(self, lane_paths: List[str]):
        if HangulPosFactorizedTokenizer is None:
            raise ImportError("HangulPosFactorizedTokenizer could not be imported.")
        pos_mode = "full" if any("pos_full" in lp for lp in lane_paths) else "coarse"
        self.tok = HangulPosFactorizedTokenizer(use_pos=True, pos_mode=pos_mode)
        self.lane_stois = []
        for lane_path in lane_paths:
            meta_path = os.path.join(REPO_ROOT, "data", lane_path, "meta.pkl")
            if not os.path.exists(meta_path):
                meta_path = os.path.join(lane_path, "meta.pkl")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"meta.pkl not found for lane: {lane_path}")
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
            # 25th lane is char lane
            byte_id = self.lane_stois[24].get(ch, ch.encode("utf-8")[0] if len(ch) > 0 and ord(ch) > 255 else 0)
            token_lists[24].append(byte_id)
        return token_lists


def _load_tokenizer_or_encoder(args: argparse.Namespace, checkpoint_config: dict, model: GPT):
    is_multicontext = getattr(model.config, "multicontext", False)

    if is_multicontext:
        multicontext_datasets = checkpoint_config.get("multicontext_datasets") if isinstance(checkpoint_config, dict) else None
        if not multicontext_datasets:
            multicontext_datasets = DEFAULT_MULTICONTEXT_LANES
        encoder = MulticontextHangulPosEncoder(multicontext_datasets)
        return True, encoder

    # Baseline single-context model tokenizer
    meta_paths: List[str] = [
        os.path.join(args.out_dir, "meta.pkl"),
        os.path.join("data", "korean_pos_mc", "char", "meta.pkl"),
    ]
    dataset_name = checkpoint_config.get("dataset") if isinstance(checkpoint_config, dict) else None
    if dataset_name:
        meta_paths.append(os.path.join("data", dataset_name, "meta.pkl"))

    for meta_path in meta_paths:
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            encode, decode = get_tokenizer_functions(meta)
            stoi = meta.get("stoi", {})
            def safe_encode(s: str) -> List[int]:
                return [stoi.get(c, 0) for c in s]
            return False, safe_encode

    raise FileNotFoundError("No meta.pkl found for baseline model tokenizer.")


def _build_context(example: dict) -> str:
    ctx = example.get("ctx")
    if ctx:
        return ctx.strip()
    ctx_a = example.get("ctx_a", "").strip()
    ctx_b = example.get("ctx_b", "").strip()
    return (ctx_a + " " + ctx_b).strip()


def compute_unigram_logprobs(dataset) -> Tuple[Dict[str, float], float]:
    """Compute empirical character unigram log probabilities over the dataset."""
    counts = Counter()
    for example in dataset:
        ctx_text = _build_context(example)
        for c in ctx_text:
            counts[c] += 1
        for ending in example.get("endings", []):
            for c in ending:
                counts[c] += 1
    total_count = sum(counts.values())
    vocab_size = max(len(counts), 1)
    eps = 1.0
    total_smoothed = total_count + eps * vocab_size
    unigram_dict = {c: math.log((cnt + eps) / total_smoothed) for c, cnt in counts.items()}
    default_logprob = math.log(eps / total_smoothed)
    return unigram_dict, default_logprob


ALL_NORM_MODES = ["length", "prior_length", "unigram_length", "none", "prior", "unigram"]


def _score_example_singlecontext(
    model: GPT,
    encode,
    ctx_text: str,
    endings: List[str],
    block_size: int,
    length_norm: bool,
    prior_norm: bool,
    unigram_norm: bool,
    unigram_dict: Dict[str, float],
    default_unigram_logprob: float,
    device: str,
    ctx_autocast,
    eval_all_norms: bool = False,
):
    ctx_tokens = encode(ctx_text)

    if eval_all_norms:
        scores_by_mode: Dict[str, List[float]] = {m: [] for m in ALL_NORM_MODES}
    else:
        scores: List[float] = []

    for ending in endings:
        end_tokens = encode(ending)
        if len(end_tokens) == 0:
            if eval_all_norms:
                for m in ALL_NORM_MODES:
                    scores_by_mode[m].append(-math.inf)
            else:
                scores.append(-math.inf)
            continue

        if len(end_tokens) > block_size:
            end_tokens = end_tokens[-block_size:]

        max_ctx_len = max(0, block_size - len(end_tokens))
        ctx_trim = ctx_tokens[-max_ctx_len:] if max_ctx_len > 0 else []
        full = ctx_trim + end_tokens
        if len(full) < 2:
            if eval_all_norms:
                for m in ALL_NORM_MODES:
                    scores_by_mode[m].append(-math.inf)
            else:
                scores.append(-math.inf)
            continue

        input_ids = torch.tensor(full[:-1], device=device, dtype=torch.long).unsqueeze(0)
        target_ids = torch.tensor(full[1:], device=device, dtype=torch.long).unsqueeze(0)
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is not None:
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            target_ids = torch.clamp(target_ids, 0, vocab_size - 1)
        ending_start = max(len(ctx_trim) - 1, 0)

        with ctx_autocast:
            logits, _ = model(input_ids, target_ids)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_slice = target_ids[:, ending_start:]
        lp_cond = logprobs[:, ending_start:, :].gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

        need_prior = prior_norm or eval_all_norms
        need_unigram = unigram_norm or eval_all_norms

        lp_uncond = None
        if need_prior:
            uncond_ctx_tokens = encode(" ")
            uncond_max_ctx_len = max(0, block_size - len(end_tokens))
            uncond_ctx_trim = uncond_ctx_tokens[-uncond_max_ctx_len:] if uncond_max_ctx_len > 0 else []
            uncond_full = uncond_ctx_trim + end_tokens
            if len(uncond_full) >= 2:
                uncond_input_ids = torch.tensor(uncond_full[:-1], device=device, dtype=torch.long).unsqueeze(0)
                uncond_target_ids = torch.tensor(uncond_full[1:], device=device, dtype=torch.long).unsqueeze(0)
                if vocab_size is not None:
                    uncond_input_ids = torch.clamp(uncond_input_ids, 0, vocab_size - 1)
                    uncond_target_ids = torch.clamp(uncond_target_ids, 0, vocab_size - 1)
                uncond_ending_start = max(len(uncond_ctx_trim) - 1, 0)

                with ctx_autocast:
                    uncond_logits, _ = model(uncond_input_ids, uncond_target_ids)
                uncond_logprobs = torch.log_softmax(uncond_logits, dim=-1)
                uncond_target_slice = uncond_target_ids[:, uncond_ending_start:]
                lp_uncond = uncond_logprobs[:, uncond_ending_start:, :].gather(-1, uncond_target_slice.unsqueeze(-1)).squeeze(-1)

        lp_unigram = None
        if need_unigram:
            lp_unigram = torch.tensor(
                [unigram_dict.get(c, default_unigram_logprob) for c in ending[-lp_cond.size(-1):]],
                device=device,
                dtype=lp_cond.dtype,
            ).unsqueeze(0)

        if eval_all_norms:
            scores_by_mode["length"].append(lp_cond.mean().item())
            scores_by_mode["none"].append(lp_cond.sum().item())

            if lp_uncond is not None:
                min_len = min(lp_cond.size(-1), lp_uncond.size(-1))
                diff_prior = lp_cond[:, -min_len:] - lp_uncond[:, -min_len:]
                scores_by_mode["prior_length"].append(diff_prior.mean().item())
                scores_by_mode["prior"].append(diff_prior.sum().item())
            else:
                scores_by_mode["prior_length"].append(lp_cond.mean().item())
                scores_by_mode["prior"].append(lp_cond.sum().item())

            if lp_unigram is not None:
                min_len = min(lp_cond.size(-1), lp_unigram.size(-1))
                diff_unigram = lp_cond[:, -min_len:] - lp_unigram[:, -min_len:]
                scores_by_mode["unigram_length"].append(diff_unigram.mean().item())
                scores_by_mode["unigram"].append(diff_unigram.sum().item())
            else:
                scores_by_mode["unigram_length"].append(lp_cond.mean().item())
                scores_by_mode["unigram"].append(lp_cond.sum().item())
        else:
            if prior_norm and lp_uncond is not None:
                min_len = min(lp_cond.size(-1), lp_uncond.size(-1))
                lp = lp_cond[:, -min_len:] - lp_uncond[:, -min_len:]
            elif unigram_norm and lp_unigram is not None:
                min_len = min(lp_cond.size(-1), lp_unigram.size(-1))
                lp = lp_cond[:, -min_len:] - lp_unigram[:, -min_len:]
            else:
                lp = lp_cond

            score = lp.mean().item() if length_norm else lp.sum().item()
            scores.append(score)

    return scores_by_mode if eval_all_norms else scores


def _score_example_multicontext(
    model: GPT,
    mc_encoder: MulticontextHangulPosEncoder,
    ctx_text: str,
    endings: List[str],
    block_size: int,
    length_norm: bool,
    prior_norm: bool,
    unigram_norm: bool,
    unigram_dict: Dict[str, float],
    default_unigram_logprob: float,
    device: str,
    ctx_autocast,
    eval_all_norms: bool = False,
):
    if eval_all_norms:
        scores_by_mode: Dict[str, List[float]] = {m: [] for m in ALL_NORM_MODES}
    else:
        scores: List[float] = []

    for ending in endings:
        num_end_tokens = len(ending)
        if num_end_tokens == 0:
            if eval_all_norms:
                for m in ALL_NORM_MODES:
                    scores_by_mode[m].append(-math.inf)
            else:
                scores.append(-math.inf)
            continue

        full_text = ctx_text + ending
        full_lanes = mc_encoder.encode(full_text)  # List of 25 lists of token IDs
        seq_len = len(full_lanes[0])

        if seq_len > block_size:
            full_lanes = [lane[-block_size:] for lane in full_lanes]
            seq_len = len(full_lanes[0])

        ctx_trim_len = max(0, seq_len - num_end_tokens)

        if seq_len < 2:
            if eval_all_norms:
                for m in ALL_NORM_MODES:
                    scores_by_mode[m].append(-math.inf)
            else:
                scores.append(-math.inf)
            continue

        input_lanes = [torch.tensor(lane[:-1], device=device).unsqueeze(0) for lane in full_lanes]
        target_lanes = [torch.tensor(lane[1:], device=device).unsqueeze(0) for lane in full_lanes]

        vocab_sizes = getattr(model.config, "vocab_sizes", None)
        if vocab_sizes is not None and len(vocab_sizes) == len(input_lanes):
            for i in range(len(input_lanes)):
                input_lanes[i] = torch.clamp(input_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)
                target_lanes[i] = torch.clamp(target_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)

        token_dict = {f"lane_{i}": lane for i, lane in enumerate(input_lanes)}
        target_dict = {f"lane_{i}": lane for i, lane in enumerate(target_lanes)}
        ending_start = max(ctx_trim_len - 1, 0)

        with ctx_autocast:
            logits_list, _ = model(token_dict=token_dict, target_dict=target_dict)

        # Logits for target character head (lane 24, last lane)
        char_logits = logits_list[-1]
        logprobs = torch.log_softmax(char_logits, dim=-1)
        target_char_ids = target_lanes[-1][:, ending_start:]
        lp_cond = logprobs[:, ending_start:, :].gather(-1, target_char_ids.unsqueeze(-1)).squeeze(-1)

        need_prior = prior_norm or eval_all_norms
        need_unigram = unigram_norm or eval_all_norms

        lp_uncond = None
        if need_prior:
            uncond_text = " " + ending
            uncond_lanes = mc_encoder.encode(uncond_text)
            uncond_seq_len = len(uncond_lanes[0])
            if uncond_seq_len > block_size:
                uncond_lanes = [lane[-block_size:] for lane in uncond_lanes]
                uncond_seq_len = len(uncond_lanes[0])

            uncond_ctx_trim_len = max(0, uncond_seq_len - num_end_tokens)

            if uncond_seq_len >= 2:
                uncond_input_lanes = [torch.tensor(lane[:-1], device=device).unsqueeze(0) for lane in uncond_lanes]
                uncond_target_lanes = [torch.tensor(lane[1:], device=device).unsqueeze(0) for lane in uncond_lanes]
                if vocab_sizes is not None and len(vocab_sizes) == len(uncond_input_lanes):
                    for i in range(len(uncond_input_lanes)):
                        uncond_input_lanes[i] = torch.clamp(uncond_input_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)
                        uncond_target_lanes[i] = torch.clamp(uncond_target_lanes[i], 0, vocab_sizes[i] - 1).to(torch.long)
                uncond_token_dict = {f"lane_{i}": lane for i, lane in enumerate(uncond_input_lanes)}
                uncond_target_dict = {f"lane_{i}": lane for i, lane in enumerate(uncond_target_lanes)}
                uncond_ending_start = max(uncond_ctx_trim_len - 1, 0)

                with ctx_autocast:
                    uncond_logits_list, _ = model(token_dict=uncond_token_dict, target_dict=uncond_target_dict)

                uncond_char_logits = uncond_logits_list[-1]
                uncond_logprobs = torch.log_softmax(uncond_char_logits, dim=-1)
                uncond_target_char_ids = uncond_target_lanes[-1][:, uncond_ending_start:]
                lp_uncond = uncond_logprobs[:, uncond_ending_start:, :].gather(-1, uncond_target_char_ids.unsqueeze(-1)).squeeze(-1)

        lp_unigram = None
        if need_unigram:
            lp_unigram = torch.tensor(
                [unigram_dict.get(c, default_unigram_logprob) for c in ending[-lp_cond.size(-1):]],
                device=device,
                dtype=lp_cond.dtype,
            ).unsqueeze(0)

        if eval_all_norms:
            scores_by_mode["length"].append(lp_cond.mean().item())
            scores_by_mode["none"].append(lp_cond.sum().item())

            if lp_uncond is not None:
                min_len = min(lp_cond.size(-1), lp_uncond.size(-1))
                diff_prior = lp_cond[:, -min_len:] - lp_uncond[:, -min_len:]
                scores_by_mode["prior_length"].append(diff_prior.mean().item())
                scores_by_mode["prior"].append(diff_prior.sum().item())
            else:
                scores_by_mode["prior_length"].append(lp_cond.mean().item())
                scores_by_mode["prior"].append(lp_cond.sum().item())

            if lp_unigram is not None:
                min_len = min(lp_cond.size(-1), lp_unigram.size(-1))
                diff_unigram = lp_cond[:, -min_len:] - lp_unigram[:, -min_len:]
                scores_by_mode["unigram_length"].append(diff_unigram.mean().item())
                scores_by_mode["unigram"].append(diff_unigram.sum().item())
            else:
                scores_by_mode["unigram_length"].append(lp_cond.mean().item())
                scores_by_mode["unigram"].append(lp_cond.sum().item())
        else:
            if prior_norm and lp_uncond is not None:
                min_len = min(lp_cond.size(-1), lp_uncond.size(-1))
                lp = lp_cond[:, -min_len:] - lp_uncond[:, -min_len:]
            elif unigram_norm and lp_unigram is not None:
                min_len = min(lp_cond.size(-1), lp_unigram.size(-1))
                lp = lp_cond[:, -min_len:] - lp_unigram[:, -min_len:]
            else:
                lp = lp_cond

            score = lp.mean().item() if length_norm else lp.sum().item()
            scores.append(score)

    return scores_by_mode if eval_all_norms else scores


def _print_example(
    tag: str,
    ctx_text: str,
    endings: List[str],
    probs: List[float],
    label: int,
    pred: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"{tag}: predicted={pred} label={label}")
    print("Context:")
    print(ctx_text)
    print("Options:")
    for i, (ending, prob) in enumerate(zip(endings, probs)):
        print(f"[{i}] p={prob:.4f} {ending}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, checkpoint_config = _load_checkpoint(args)
    is_multicontext, tokenizer_or_encoder = _load_tokenizer_or_encoder(args, checkpoint_config, model)

    model.eval()
    model.to(args.device)

    block_size = args.block_size or model.config.block_size
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    ctx_autocast = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    eval_all_norms = bool(args.eval_all_norms) or (args.norm_type == "all")

    if not eval_all_norms:
        if args.norm_type is not None:
            norm_type = args.norm_type
            if norm_type == "none":
                length_norm, prior_norm, unigram_norm = False, False, False
            elif norm_type == "length":
                length_norm, prior_norm, unigram_norm = True, False, False
            elif norm_type == "prior":
                length_norm, prior_norm, unigram_norm = False, True, False
            elif norm_type == "prior_length":
                length_norm, prior_norm, unigram_norm = True, True, False
            elif norm_type == "unigram":
                length_norm, prior_norm, unigram_norm = False, False, True
            elif norm_type == "unigram_length":
                length_norm, prior_norm, unigram_norm = True, False, True
        else:
            length_norm = bool(args.length_norm)
            prior_norm = bool(args.prior_norm)
            unigram_norm = bool(args.unigram_norm)
            if prior_norm:
                norm_type = "prior_length" if length_norm else "prior"
            elif unigram_norm:
                norm_type = "unigram_length" if length_norm else "unigram"
            elif length_norm:
                norm_type = "length"
            else:
                norm_type = "none"
    else:
        norm_type = "all"
        length_norm, prior_norm, unigram_norm = True, True, True

    print(f"Loading benchmark dataset '{args.dataset_name}' (split={args.split})...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_examples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.max_examples, len(dataset))))

    unigram_dict, default_unigram_logprob = {}, 0.0
    if unigram_norm or eval_all_norms:
        print("Computing dataset unigram log probabilities...")
        unigram_dict, default_unigram_logprob = compute_unigram_logprobs(dataset)

    model_type_str = "HangulPosFactorizedTokenizer (Multicontext)" if is_multicontext else "Baseline (Single Context)"
    print(f"Evaluating {model_type_str} on {len(dataset)} examples (norm_type={norm_type})...")

    skipped = 0
    total = 0

    if eval_all_norms:
        correct_counts = {m: 0 for m in ALL_NORM_MODES}
    else:
        correct = 0
        printed_correct = False
        printed_incorrect = False

    with torch.inference_mode():
        for example in dataset:
            ctx_text = _build_context(example)
            endings = example["endings"]
            raw_label = example.get("label")
            if raw_label is None:
                skipped += 1
                continue
            label = int(raw_label)

            if is_multicontext:
                scores_out = _score_example_multicontext(
                    model=model,
                    mc_encoder=tokenizer_or_encoder,
                    ctx_text=ctx_text,
                    endings=endings,
                    block_size=block_size,
                    length_norm=length_norm,
                    prior_norm=prior_norm,
                    unigram_norm=unigram_norm,
                    unigram_dict=unigram_dict,
                    default_unigram_logprob=default_unigram_logprob,
                    device=args.device,
                    ctx_autocast=ctx_autocast,
                    eval_all_norms=eval_all_norms,
                )
            else:
                scores_out = _score_example_singlecontext(
                    model=model,
                    encode=tokenizer_or_encoder,
                    ctx_text=ctx_text,
                    endings=endings,
                    block_size=block_size,
                    length_norm=length_norm,
                    prior_norm=prior_norm,
                    unigram_norm=unigram_norm,
                    unigram_dict=unigram_dict,
                    default_unigram_logprob=default_unigram_logprob,
                    device=args.device,
                    ctx_autocast=ctx_autocast,
                    eval_all_norms=eval_all_norms,
                )

            total += 1

            if eval_all_norms:
                for m in ALL_NORM_MODES:
                    pred_m = int(np.argmax(scores_out[m]))
                    if pred_m == label:
                        correct_counts[m] += 1
            else:
                scores = scores_out
                pred = int(np.argmax(scores))
                if pred == label:
                    correct += 1
                    if args.print_examples and not printed_correct:
                        probs = torch.softmax(torch.tensor(scores), dim=-1).tolist()
                        _print_example("CORRECT", ctx_text, endings, probs, label, pred)
                        printed_correct = True
                if pred != label and args.print_examples and not printed_incorrect:
                    probs = torch.softmax(torch.tensor(scores), dim=-1).tolist()
                    _print_example("INCORRECT", ctx_text, endings, probs, label, pred)
                    printed_incorrect = True

    if eval_all_norms:
        accuracies = {m: (correct_counts[m] / total) if total else float("nan") for m in ALL_NORM_MODES}
        metrics = {
            "model_type": model_type_str,
            "is_multicontext": is_multicontext,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "total": total,
            "skipped": skipped,
            "block_size": block_size,
            "norm_type": "all",
            "accuracies": accuracies,
            "correct_counts": correct_counts,
        }
    else:
        acc = (correct / total) if total else float("nan")
        metrics = {
            "model_type": model_type_str,
            "is_multicontext": is_multicontext,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "skipped": skipped,
            "block_size": block_size,
            "norm_type": norm_type,
            "length_norm": length_norm,
            "prior_norm": prior_norm,
            "unigram_norm": unigram_norm,
        }

    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()

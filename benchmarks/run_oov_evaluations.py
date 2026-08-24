#!/usr/bin/env python3
"""Run OOV (Out-of-Vocabulary) and Unicode Robustness Evaluations across Models.

Evaluates character, BPE byte-fallback, pure byte, and multicontext POS factorized models
on curated stress-test prompts containing archaic Hangul, rare Hanja, ancient scripts,
complex emojis (ZWJ sequences & flags), mathematical symbols, and internet slang.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "data", "template", "utils", "korean"))

from model import GPT, GPTConfig
from sample import get_tokenizer_functions
from hangul_factorizer import HangulPosFactorizedTokenizer

MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "baseline_character": {
        "display_name": "Original Character Baseline (V=4,555)",
        "ckpt_path": "out_baseline_korean_pos_full1m/ckpt.pt",
        "type": "single_char",
        "meta_path": "out_baseline_korean_pos_full1m/meta.pkl",
    },
    "byte_fallback_bpe": {
        "display_name": "Byte-Fallback BPE Baseline (V=4,924)",
        "ckpt_path": "out_baseline_char_bpe/ckpt.pt",
        "type": "single_bpe",
        "meta_path": "data/korean_pos_mc/char_bpe/meta.pkl",
    },
    "pure_byte": {
        "display_name": "Pure Byte Tokenizer Baseline (V=256)",
        "ckpt_path": "out_baseline_pure_byte/ckpt.pt",
        "type": "single_byte",
        "meta_path": "data/korean_pos_mc/byte/meta.pkl",
    },
    "std_full_pos_weighted": {
        "display_name": "Standard Full POS (Weighted, ws=0.05)",
        "ckpt_path": "out_mc_korean_pos_weighted/ckpt.pt",
        "type": "mc_char_full",
        "pos_mode": "full",
    },
    "std_full_pos_unweighted": {
        "display_name": "Standard Full POS (Unweighted, w=1.0)",
        "ckpt_path": "out_mc_korean_pos_unweighted/ckpt.pt",
        "type": "mc_char_full",
        "pos_mode": "full",
    },
    "std_coarse_pos_weighted": {
        "display_name": "Standard Coarse POS (Weighted, ws=0.05)",
        "ckpt_path": "out_opt1_sw005_10ep/ckpt.pt",
        "type": "mc_char_coarse",
        "pos_mode": "coarse",
    },
    "byte_fallback_full_pos_weighted": {
        "display_name": "Byte Fallback Full POS (Weighted, ws=0.05)",
        "ckpt_path": "out_mc_full_pos_weighted_byte/ckpt.pt",
        "type": "mc_byte_full",
        "pos_mode": "full",
    },
    "byte_fallback_full_pos_unweighted": {
        "display_name": "Byte Fallback Full POS (Unweighted, w=1.0)",
        "ckpt_path": "out_mc_full_pos_unweighted_byte/ckpt.pt",
        "type": "mc_byte_full",
        "pos_mode": "full",
    },
    "byte_fallback_coarse_pos_weighted": {
        "display_name": "Byte Fallback Coarse POS (Weighted, ws=0.05)",
        "ckpt_path": "out_mc_coarse_pos_weighted_byte/ckpt.pt",
        "type": "mc_byte_coarse",
        "pos_mode": "coarse",
    },
    "byte_fallback_coarse_pos_unweighted": {
        "display_name": "Byte Fallback Coarse POS (Unweighted, w=1.0)",
        "ckpt_path": "out_mc_coarse_pos_unweighted_byte/ckpt.pt",
        "type": "mc_byte_coarse",
        "pos_mode": "coarse",
    },
}

DEFAULT_LANES_COARSE = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
    "korean_pos_mc/pos", "korean_pos_mc/char"
]

DEFAULT_LANES_FULL = [
    "korean_pos_mc/script", "korean_pos_mc/choseong", "korean_pos_mc/jungseong", "korean_pos_mc/jongseong",
    "korean_pos_mc/jung_base1", "korean_pos_mc/jung_base2", "korean_pos_mc/jung_has_w", "korean_pos_mc/jung_has_y",
    "korean_pos_mc/jung_has_i", "korean_pos_mc/jong_base1", "korean_pos_mc/jong_base2", "korean_pos_mc/jong_base3",
    "korean_pos_mc/choseong_tense", "korean_pos_mc/choseong_aspirated", "korean_pos_mc/choseong_nasal_liquid",
    "korean_pos_mc/choseong_place", "korean_pos_mc/jung_height", "korean_pos_mc/jung_backness", "korean_pos_mc/jung_round",
    "korean_pos_mc/jong_complex", "korean_pos_mc/has_batchim", "korean_pos_mc/syllable_index_mod", "korean_pos_mc/codepoint_mod",
    "korean_pos_mc/pos_full", "korean_pos_mc/char_byte"
]


def load_model(ckpt_path: str, device: str) -> tuple[GPT, dict]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", {})
    allowed = {f.name for f in fields(GPTConfig)}
    filtered_args = {k: v for k, v in model_args.items() if k in allowed}
    filtered_args["dropout"] = 0.0
    conf = GPTConfig(**filtered_args)
    model = GPT(conf)
    state_dict = checkpoint["model"]
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, checkpoint.get("config", {})


def get_prompts(prompts_file: Optional[str] = None) -> List[str]:
    prompts_dir = os.path.join(REPO_ROOT, "benchmarks", "prompts")
    if prompts_file:
        candidate_paths = [prompts_file, os.path.join(prompts_dir, prompts_file)]
        for cp in candidate_paths:
            if os.path.exists(cp):
                with open(cp, "r", encoding="utf-8") as fp:
                    return [line.strip() for line in fp if line.strip()]
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    all_prompts_file = os.path.join(prompts_dir, "all_oov_prompts.txt")
    if os.path.exists(all_prompts_file):
        with open(all_prompts_file, "r", encoding="utf-8") as fp:
            return [line.strip() for line in fp if line.strip()]

    file_candidates = [
        "oov_prompts.txt",
        "bpe_oov_prompts.txt",
        "all_baseline_oov_prompts.txt",
        "pure_byte_prompts.txt"
    ]
    all_lines: List[str] = []
    for fname in file_candidates:
        fpath = os.path.join(prompts_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        all_lines.append(line)
    return list(dict.fromkeys(all_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on OOV & Unicode stress-test prompts")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to custom prompts file (or filename within benchmarks/prompts/)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    parser.add_argument("--max_new_tokens", type=int, default=35, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling threshold")
    parser.add_argument("--output_json", type=str, default="benchmarks/prompts/oov_prompts_model_responses.json", help="Path to save output results JSON")
    parser.add_argument("--models", nargs="+", default=list(MODELS_CONFIG.keys()), help="Model keys to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    prompts = get_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} unique evaluation prompts.")
    
    # Load lane stois for multicontext models
    mc_stois_coarse: List[Dict[str, int]] = []
    for lp in DEFAULT_LANES_COARSE:
        mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
        with open(mp, "rb") as f:
            mc_stois_coarse.append(pickle.load(f)["stoi"])
            
    mc_stois_full: List[Dict[str, int]] = []
    for lp in DEFAULT_LANES_FULL:
        mp = os.path.join(REPO_ROOT, "data", lp, "meta.pkl")
        if not os.path.exists(mp):
            mp = os.path.join(REPO_ROOT, "data", "korean_pos_mc", "char", "meta.pkl")
        with open(mp, "rb") as f:
            mc_stois_full.append(pickle.load(f)["stoi"])

    tok_full = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="full")
    tok_coarse = HangulPosFactorizedTokenizer(use_pos=True, pos_mode="coarse")

    results_data: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_prompts": len(prompts),
            "device": device,
            "models_evaluated": args.models,
            "generation_parameters": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k
            }
        },
        "prompts": [
            {"id": i, "text": p} for i, p in enumerate(prompts, 1)
        ],
        "model_evaluations": {}
    }

    for model_key in args.models:
        if model_key not in MODELS_CONFIG:
            print(f"Skipping unknown model key: {model_key}")
            continue
        mcfg = MODELS_CONFIG[model_key]
        print(f"\n==================================================")
        print(f" EVALUATING: {mcfg['display_name']}")
        print(f" Checkpoint: {mcfg['ckpt_path']}")
        print(f"==================================================")

        try:
            model, _ = load_model(mcfg["ckpt_path"], device)
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")
            continue

        model_eval_records = []
        m_type = mcfg["type"]
        is_mc = m_type.startswith("mc_")

        # Prepare single tokenizer if applicable
        if not is_mc:
            meta_path = mcfg.get("meta_path")
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            encode_fn, decode_fn = get_tokenizer_functions(meta)
            stoi = meta.get("stoi", {})

        for p_idx, prompt_text in enumerate(prompts, 1):
            t0 = time.time()
            record: Dict[str, Any] = {
                "prompt_id": p_idx,
                "prompt": prompt_text,
            }

            if not is_mc:
                # Single-context encoding
                if m_type == "single_char":
                    oov_chars = [c for c in prompt_text if c not in stoi]
                    tokens = [stoi.get(c, 0) for c in prompt_text]
                    record["oov_characters"] = oov_chars
                    record["num_oov_characters"] = len(oov_chars)
                    record["encoding_status"] = f"{len(oov_chars)} OOV characters replaced with token 0" if oov_chars else "Clean"
                elif m_type == "single_bpe":
                    fallback_bytes = [c for c in prompt_text if c not in stoi]
                    tokens = encode_fn(prompt_text)
                    record["fallback_characters"] = fallback_bytes
                    record["num_fallback_characters"] = len(fallback_bytes)
                    record["encoding_status"] = f"{len(fallback_bytes)} characters triggered byte-fallback" if fallback_bytes else "Clean"
                elif m_type == "single_byte":
                    tokens = encode_fn(prompt_text)
                    record["utf8_byte_length"] = len(tokens)
                    record["encoding_status"] = "Native byte stream (100% covered)"

                seq_len = len(tokens)
                record["prompt_token_count"] = seq_len

                # Calculate prompt loss & perplexity
                x_input = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    if seq_len > 1:
                        inp = x_input[:, :-1]
                        tgt = x_input[:, 1:]
                        logits, loss_tensor = model(inp, targets=tgt)
                        loss = loss_tensor.item()
                        ppl = math.exp(min(loss, 50))
                    else:
                        loss = 0.0
                        ppl = 1.0

                    # Autoregressive generation
                    curr_tokens = x_input.clone()
                    block_size = model.config.block_size
                    for _ in range(args.max_new_tokens):
                        cond = curr_tokens if curr_tokens.size(1) <= block_size else curr_tokens[:, -block_size:]
                        logits, _ = model(cond)
                        next_token_logits = logits[:, -1, :] / args.temperature
                        if args.top_k is not None:
                            v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                            next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)
                        curr_tokens = torch.cat((curr_tokens, next_tok), dim=1)

                    generated_ids = curr_tokens[0, seq_len:].tolist()
                    response_text = decode_fn(generated_ids)

                record["prompt_loss"] = round(loss, 4)
                record["prompt_perplexity"] = round(ppl, 2)
                record["generated_tokens_count"] = len(generated_ids)
                record["response"] = response_text
                record["latency_ms"] = round((time.time() - t0) * 1000, 2)

            else:
                # Multicontext encoding
                pos_mode = mcfg["pos_mode"]
                tok = tok_full if pos_mode == "full" else tok_coarse
                stois = mc_stois_full if pos_mode == "full" else mc_stois_coarse
                is_byte_companion = ("byte" in m_type)

                seq = tok.encode_text(prompt_text)
                n_lanes = len(stois)
                token_lists: List[List[int]] = [[] for _ in range(n_lanes)]
                for item in seq:
                    ch = item["char"]
                    indices = item["indices"]
                    for i in range(min(24, n_lanes - 1)):
                        t_char = tok.token_for(i, indices[i])
                        token_lists[i].append(stois[i].get(t_char, 0))
                    if n_lanes > 24:
                        if is_byte_companion:
                            # 256-byte fallback stream
                            code = ord(ch) if len(ch) > 0 else 0
                            byte_val = code if code < 256 else ch.encode("utf-8")[0]
                            token_lists[24].append(byte_val)
                        else:
                            # character stream
                            token_lists[24].append(stois[24].get(ch, 0))

                seq_len = len(token_lists[0])
                record["prompt_token_count"] = seq_len
                
                # Check for OOV / byte fallback
                if not is_byte_companion:
                    oov_chars = [c for c in prompt_text if c not in stois[24]]
                    record["oov_characters"] = oov_chars
                    record["num_oov_characters"] = len(oov_chars)
                    record["encoding_status"] = f"{len(oov_chars)} character lane OOV replaced with 0" if oov_chars else "Clean"
                else:
                    record["byte_fallback_active"] = True
                    record["encoding_status"] = "25-lane synchronous encoding with 256-byte fallback (0% OOV)"

                # Tensors
                tensors = [torch.tensor(lane, dtype=torch.long, device=device).unsqueeze(0) for lane in token_lists]
                vocab_sizes = getattr(model.config, "vocab_sizes", None)
                if vocab_sizes is not None and len(vocab_sizes) == len(tensors):
                    for i in range(len(tensors)):
                        tensors[i] = torch.clamp(tensors[i], 0, vocab_sizes[i] - 1)

                with torch.no_grad():
                    if seq_len > 1:
                        inp_list = [t[:, :-1] for t in tensors]
                        tgt_list = [t[:, 1:] for t in tensors]
                        logits_list, losses = model(idx=inp_list, targets=tgt_list)
                        loss = losses[-1].item() if isinstance(losses, list) else losses.item()
                        ppl = math.exp(min(loss, 50))
                    else:
                        loss = 0.0
                        ppl = 1.0

                    # Autoregressive generation
                    curr_state = [t.clone() for t in tensors]
                    block_size = model.config.block_size
                    for _ in range(args.max_new_tokens):
                        cond_list = [t if t.size(1) <= block_size else t[:, -block_size:] for t in curr_state]
                        logits_list, _ = model(idx=cond_list)
                        for i in range(len(curr_state)):
                            logit_i = logits_list[i][:, -1, :] / args.temperature
                            if args.top_k is not None:
                                v, _ = torch.topk(logit_i, min(args.top_k, logit_i.size(-1)))
                                logit_i[logit_i < v[:, [-1]]] = -float("inf")
                            prob_i = F.softmax(logit_i, dim=-1)
                            next_tok_i = torch.multinomial(prob_i, num_samples=1)
                            curr_state[i] = torch.cat((curr_state[i], next_tok_i), dim=1)

                    # Decode generated tokens
                    gen_len = args.max_new_tokens
                    chars_generated = []
                    for s_step in range(seq_len, seq_len + gen_len):
                        step_indices = [curr_state[i][0, s_step].item() for i in range(min(24, n_lanes - 1))]
                        decoded_char = tok.decode_indices(step_indices)
                        if decoded_char is None or decoded_char == "":
                            last_id = curr_state[-1][0, s_step].item()
                            if is_byte_companion:
                                try:
                                    decoded_char = bytes([last_id]).decode("utf-8", errors="replace")
                                except:
                                    decoded_char = chr(last_id) if last_id < 256 else "?"
                            else:
                                itos_char = {v: k for k, v in stois[-1].items()}
                                decoded_char = itos_char.get(last_id, "")
                        chars_generated.append(decoded_char)

                    response_text = "".join(chars_generated)

                record["prompt_loss"] = round(loss, 4)
                record["prompt_perplexity"] = round(ppl, 2)
                record["generated_tokens_count"] = args.max_new_tokens
                record["response"] = response_text
                record["latency_ms"] = round((time.time() - t0) * 1000, 2)

            model_eval_records.append(record)
            print(f"  [{p_idx:2d}/{len(prompts)}] Loss: {record['prompt_loss']:<6.4f} PPL: {record['prompt_perplexity']:<8.2f} Lat: {record['latency_ms']:<6.1f}ms | Resp: {record['response'][:30]}...")

        avg_loss = round(sum(r["prompt_loss"] for r in model_eval_records) / len(model_eval_records), 4)
        avg_ppl = round(sum(r["prompt_perplexity"] for r in model_eval_records) / len(model_eval_records), 2)
        avg_lat = round(sum(r["latency_ms"] for r in model_eval_records) / len(model_eval_records), 2)

        results_data["model_evaluations"][model_key] = {
            "display_name": mcfg["display_name"],
            "model_type": mcfg["type"],
            "checkpoint_path": mcfg["ckpt_path"],
            "records": model_eval_records,
            "summary_metrics": {
                "average_prompt_loss": avg_loss,
                "average_prompt_perplexity": avg_ppl,
                "average_generation_latency_ms": avg_lat
            }
        }
        print(f"--> Finished {mcfg['display_name']}: Avg Loss={avg_loss}, Avg PPL={avg_ppl}, Avg Latency={avg_lat}ms")

    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(results_data, fp, ensure_ascii=False, indent=2)
    print(f"\n==================================================")
    print(f" ALL EVALUATIONS COMPLETE! Saved to {args.output_json}")
    print(f"==================================================")


if __name__ == "__main__":
    main()

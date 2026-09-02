#!/usr/bin/env python3
"""OOV Contextual Safety Benchmark on Highly Irregular Unicode.

Evaluates Three-Hot Tokenizer vs Hangul Factorizer on:
1. Middle Korean (Archaic Hangul combining jamos & archaic vowels like 아래아 ㆍ).
2. Complex & Rare Hanja (e.g. 龘, 𪚥, 鬱, 爨).
3. Emoji Ligatures & Zero-Width Joiner (ZWJ) sequences (e.g. 👨‍👩‍👧‍👦, 🏳️‍🌈, 👍🏾).
4. Combining diacritics / Zalgo text (e.g. 한ᄀ̶̧͊글̸).
5. Polyglot / Mathematical script.

Metrics:
- Crash / Exception Safety (NaN, CUDA out of bounds, infinite loops).
- Tokenizer UNK Rate & Byte Integrity.
- Completion Degeneracy / Repetition Rate (collapse into single-syllable loops).
- Post-Prompt Contextual Recovery (resuming valid syllable generation).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from benchmarks.seq2seq_hangul_comparison.tokenizers import (
    EnglishBPETokenizer,
    ThreeHotSeq2SeqTokenizer,
    HangulFactorizerSeq2SeqTokenizer,
    S_BASE,
    S_COUNT,
)
from benchmarks.seq2seq_hangul_comparison.models import (
    Seq2SeqThreeHotConditional,
    Seq2SeqThreeHotIndependent,
    Seq2SeqHangulFactorizer,
)

OOV_TEST_BATTERY = [
    # 1. Middle Korean (Archaic Hangul)
    {
        "category": "Middle Korean (Archaic)",
        "name": "Hunminjeongeum Preface",
        "prompt": "ᄂᆞ랏ᄆᆞᆯᄊᆞ미",
        "desc": "Archaic syllables with 아래아 (ㆍ) and double choseong (ᄊ)",
    },
    {
        "category": "Middle Korean (Archaic)",
        "name": "Archaic Hangul Word",
        "prompt": "ᄒᆞᆫᄀᆞᆯ ᄠᅳᆮ",
        "desc": "Combining jamos outside modern precomposed Hangul block",
    },
    {
        "category": "Middle Korean (Archaic)",
        "name": "Isolated Archaic Jamos",
        "prompt": "ㆍ ᅀ ᇢ ᄯ",
        "desc": "Arae-a, Bansiot, Sun-gyeong-eum bieup, Dtan",
    },
    # 2. Complex & Rare Hanja
    {
        "category": "Complex Hanja",
        "name": "Extreme Stroke Dragons",
        "prompt": "龘 𪚥 鬱 爨",
        "desc": "48-stroke, 64-stroke, and high-complexity Hanja ideographs",
    },
    {
        "category": "Complex Hanja",
        "name": "Korean Hanja Motto",
        "prompt": "弘益人間 錦繡江山",
        "desc": "Classical 4-character idioms",
    },
    # 3. Emoji Ligatures & ZWJ
    {
        "category": "Emoji Ligature",
        "name": "Family ZWJ Sequence",
        "prompt": "👨‍👩‍👧‍👦",
        "desc": "4-person Zero-Width Joiner compound ligature (7 unicode code points)",
    },
    {
        "category": "Emoji Ligature",
        "name": "Rainbow Flag Ligature",
        "prompt": "🏳️‍🌈",
        "desc": "White flag + Variation selector + ZWJ + Rainbow",
    },
    {
        "category": "Emoji Ligature",
        "name": "Skin Tone & Profession",
        "prompt": "👍🏾 👩‍⚕️",
        "desc": "Fitzpatrick modifier and gender-profession ZWJ",
    },
    # 4. Combining Diacritics & Glitch
    {
        "category": "Glitch / Zalgo",
        "name": "Zalgo Combining Hangul",
        "prompt": "한ᄀ̶̧͊글̸ 테ᷠ스ᷢ트ᷨ",
        "desc": "Hangul syllables overlaid with multi-layer combining diacritics",
    },
    # 5. Polyglot / Math
    {
        "category": "Polyglot & Math",
        "name": "Math & Logic Notation",
        "prompt": "∀x ∈ ℝ, x² ≥ 0",
        "desc": "Mathematical quantifiers, set relations, superscripts",
    },
    {
        "category": "Polyglot & Math",
        "name": "Code / Control Characters",
        "prompt": "def func(): return NULL;\x00\x07",
        "desc": "Programming syntax with embedded null and bell escape bytes",
    },
]


def detect_looping(text: str, min_repeat: int = 4) -> bool:
    """Detects if completion collapsed into repetitive loops like '안 안 안 안' or 'ㅋㅋㅋㅋ'."""
    if not text.strip():
        return False
    # Check repeated 1-3 char patterns
    for pat_len in range(1, 4):
        matches = re.findall(r"(.{" + str(pat_len) + r"})\1{" + str(min_repeat - 1) + r",}", text)
        if matches:
            return True
    return False


def count_valid_hangul(text: str) -> int:
    return sum(1 for c in text if S_BASE <= ord(c) < S_BASE + S_COUNT)


def generate_completion_cond(
    model: Seq2SeqThreeHotConditional,
    memory: torch.Tensor,
    tok: ThreeHotSeq2SeqTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 25,
) -> Tuple[str, str]:
    prompt_ids = [(tok.sos_id, 0, 0)] + tok.encode(prompt_text, add_special_tokens=False)
    gen = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_tokens = []

    for _ in range(max_new_tokens):
        emb = model.embed_target(gen)
        mask = nn.Transformer.generate_square_subsequent_mask(gen.size(1), device=device)
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
        new_tokens.append((pi_i.item(), pi_v.item(), pi_f.item()))
        if pi_i.item() == tok.eos_id:
            break

    full_text = tok.decode(gen[0].tolist())
    continuation = tok.decode(new_tokens)
    return full_text, continuation


def generate_completion_ind(
    model: Seq2SeqThreeHotIndependent,
    memory: torch.Tensor,
    tok: ThreeHotSeq2SeqTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 25,
) -> Tuple[str, str]:
    prompt_ids = [(tok.sos_id, 0, 0)] + tok.encode(prompt_text, add_special_tokens=False)
    gen = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_tokens = []

    for _ in range(max_new_tokens):
        emb = model.embed_target(gen)
        mask = nn.Transformer.generate_square_subsequent_mask(gen.size(1), device=device)
        h_t = model.decode_step(emb, memory, tgt_mask=mask)
        last_h = h_t[:, -1, :]
        pi_i = model.head_i(last_h).argmax(dim=-1)
        pi_v = model.head_v(last_h).argmax(dim=-1)
        pi_f = model.head_f(last_h).argmax(dim=-1)
        nxt = torch.stack([pi_i, pi_v, pi_f], dim=-1).unsqueeze(1)
        gen = torch.cat([gen, nxt], dim=1)
        new_tokens.append((pi_i.item(), pi_v.item(), pi_f.item()))
        if pi_i.item() == tok.eos_id:
            break

    full_text = tok.decode(gen[0].tolist())
    continuation = tok.decode(new_tokens)
    return full_text, continuation


def generate_completion_fact(
    model: Seq2SeqHangulFactorizer,
    memory: torch.Tensor,
    tok: HangulFactorizerSeq2SeqTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 25,
) -> Tuple[str, str]:
    prompt_ids = [[tok.sos_id] + [0] * 22] + tok.encode(prompt_text, add_special_tokens=False)
    gen = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_tokens = []

    for _ in range(max_new_tokens):
        emb = model.embed_target(gen)
        mask = nn.Transformer.generate_square_subsequent_mask(gen.size(1), device=device)
        h_t = model.decode_step(emb, memory, tgt_mask=mask)
        last_h = h_t[:, -1, :]
        preds = [head(last_h).argmax(dim=-1) for head in model.heads]
        nxt = torch.stack(preds, dim=-1).unsqueeze(1)
        gen = torch.cat([gen, nxt], dim=1)
        new_tokens.append([p.item() for p in preds])
        if preds[0].item() == tok.eos_id:
            break

    full_text = tok.decode(gen[0].tolist())
    continuation = tok.decode(new_tokens)
    return full_text, continuation


def run_oov_safety_benchmark(
    ckpt_dir: Path,
    output_json: Path,
) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_cond = torch.load(ckpt_dir / "three_hot_conditional_best.pt", map_location=device, weights_only=False)
    ckpt_ind = torch.load(ckpt_dir / "three_hot_independent_best.pt", map_location=device, weights_only=False)
    ckpt_fact = torch.load(ckpt_dir / "hangul_factorizer_best.pt", map_location=device, weights_only=False)

    non_ko_vocab = ckpt_cond["non_ko_vocab"]
    src_spm = REPO_ROOT / "data/korean_seq2seq_bench/spm_en_30k.model"
    src_tok = EnglishBPETokenizer(src_spm)

    tok_th = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
    tok_hf = HangulFactorizerSeq2SeqTokenizer(non_ko_vocab)

    model_cond = Seq2SeqThreeHotConditional(src_tok.vocab_size, ckpt_cond["tgt_vocab_sizes"]).to(device)
    model_cond.load_state_dict(ckpt_cond["model_state_dict"])
    model_cond.eval()

    model_ind = Seq2SeqThreeHotIndependent(src_tok.vocab_size, ckpt_ind["tgt_vocab_sizes"]).to(device)
    model_ind.load_state_dict(ckpt_ind["model_state_dict"])
    model_ind.eval()

    model_fact = Seq2SeqHangulFactorizer(src_tok.vocab_size, ckpt_fact["tgt_vocab_sizes"]).to(device)
    model_fact.load_state_dict(ckpt_fact["model_state_dict"])
    model_fact.eval()

    # Generic encoder conditioning
    src_tensor = torch.tensor([src_tok.encode("Prompt:", add_special_tokens=True)], device=device)
    memory_cond = model_cond.encode(src_tensor)
    memory_ind = model_ind.encode(src_tensor)
    memory_fact = model_fact.encode(src_tensor)

    results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("RUNNING OOV CONTEXTUAL SAFETY EVALUATION")
    print("=" * 80)

    for case in OOV_TEST_BATTERY:
        name = case["name"]
        cat = case["category"]
        prompt = case["prompt"]
        desc = case["desc"]

        print(f"\n[{cat}] {name}: \"{prompt}\" ({desc})")

        # 1. Tokenizer Analysis
        th_enc = tok_th.encode(prompt, add_special_tokens=False)
        hf_enc = tok_hf.encode(prompt, add_special_tokens=False)

        th_unks = sum(1 for id0, _, _ in th_enc if id0 == tok_th.unk_id)
        hf_unks = sum(1 for step in hf_enc if step[0] == tok_hf.unk_id)

        th_dec = tok_th.decode(th_enc)
        hf_dec = tok_hf.decode(hf_enc)

        # 2. Model Completions
        crashed_cond = False
        crashed_ind = False
        crashed_fact = False

        try:
            full_c, cont_c = generate_completion_cond(model_cond, memory_cond, tok_th, prompt, device)
        except Exception as e:
            crashed_cond = True
            full_c, cont_c = f"CRASH: {e}", ""

        try:
            full_i, cont_i = generate_completion_ind(model_ind, memory_ind, tok_th, prompt, device)
        except Exception as e:
            crashed_ind = True
            full_i, cont_i = f"CRASH: {e}", ""

        try:
            full_f, cont_f = generate_completion_fact(model_fact, memory_fact, tok_hf, prompt, device)
        except Exception as e:
            crashed_fact = True
            full_f, cont_f = f"CRASH: {e}", ""

        looping_c = detect_looping(cont_c)
        looping_i = detect_looping(cont_i)
        looping_f = detect_looping(cont_f)

        hangul_count_c = count_valid_hangul(cont_c)
        hangul_count_i = count_valid_hangul(cont_i)
        hangul_count_f = count_valid_hangul(cont_f)

        print(f"  Three-Hot Cond: crash={crashed_cond}, loop={looping_c}, Hangul={hangul_count_c} -> \"{cont_c[:30]}\"")
        print(f"  Three-Hot Ind:  crash={crashed_ind}, loop={looping_i}, Hangul={hangul_count_i} -> \"{cont_i[:30]}\"")
        print(f"  Factorizer:     crash={crashed_fact}, loop={looping_f}, Hangul={hangul_count_f} -> \"{cont_f[:30]}\"")

        res_entry = {
            "name": name,
            "category": cat,
            "prompt": prompt,
            "description": desc,
            "prompt_length_chars": len(prompt),
            "tokens_three_hot": len(th_enc),
            "tokens_factorizer": len(hf_enc),
            "unk_three_hot": th_unks,
            "unk_factorizer": hf_unks,
            "decoded_prompt_three_hot": th_dec,
            "decoded_prompt_factorizer": hf_dec,
            "conditional": {
                "crashed": crashed_cond,
                "looping": looping_c,
                "continuation": cont_c,
                "valid_hangul_count": hangul_count_c,
            },
            "independent": {
                "crashed": crashed_ind,
                "looping": looping_i,
                "continuation": cont_i,
                "valid_hangul_count": hangul_count_i,
            },
            "factorizer": {
                "crashed": crashed_fact,
                "looping": looping_f,
                "continuation": cont_f,
                "valid_hangul_count": hangul_count_f,
            },
        }
        results.append(res_entry)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSafety benchmark results written to {output_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOV Contextual Safety on Irregular Unicode.")
    parser.add_argument("--ckpt_dir", type=str, default="out_seq2seq_bench")
    parser.add_argument("--output_json", type=str, default="out_seq2seq_bench/oov_safety_results.json")
    args = parser.parse_args()

    run_oov_safety_benchmark(Path(args.ckpt_dir), Path(args.output_json))


if __name__ == "__main__":
    main()

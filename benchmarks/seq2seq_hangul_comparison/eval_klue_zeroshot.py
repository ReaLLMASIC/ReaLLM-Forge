#!/usr/bin/env python3
"""Zero-Shot Probability Benchmark on KLUE-NER and KLUE-DP.

Evaluates Three-Hot Tokenizer vs. Hangul Factorizer on:
1. KLUE-NER (Named Entity Recognition):
   - Overall sentence BPJ on formal journalistic Korean.
   - Entity-token BPJ vs. Non-entity (O) token BPJ.
   - Per-category Entity BPJ (Person, Organization, Location, Date, Time, Quantity).
   - Zero-shot entity type classification probability & accuracy (인물, 단체, 장소, 날짜, 시간, 수량).
2. KLUE-DP (Dependency Parsing):
   - Overall syntactic sentence BPJ.
   - Syntactic role BPJ breakdown: Subject (NP_SBJ), Object (NP_OBJ), Modifier (NP_MOD), Adverbial (NP_AJT), Predicate (VP).
   - Zero-shot dependency relation classification probability & accuracy (주어, 목적어, 수식어, 부사어, 서술어).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm

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
from benchmarks.seq2seq_hangul_comparison.evaluate import calculate_bpj

NER_TAG_MAP = {
    0: "DT", 1: "DT",
    2: "LC", 3: "LC",
    4: "OG", 5: "OG",
    6: "PS", 7: "PS",
    8: "QT", 9: "QT",
    10: "TI", 11: "TI",
    12: "O",
}

NER_LABEL_NAMES = {
    "PS": "인물",
    "OG": "단체",
    "LC": "장소",
    "DT": "날짜",
    "TI": "시간",
    "QT": "수량",
}

DP_RELATION_NAMES = {
    "NP_SBJ": "주어",
    "NP_OBJ": "목적어",
    "NP_MOD": "수식어",
    "NP_AJT": "부사어",
    "VP": "서술어",
}


def clean_ner_sentence(annotated: str) -> str:
    # Removes XML-like entity tags: <경찰:OG> -> 경찰
    return re.sub(r"<([^:>]+):[A-Z]+>", r"\1", annotated)


def compute_sequence_log_prob_cond(
    model: Seq2SeqThreeHotConditional,
    memory: torch.Tensor,
    tok: ThreeHotSeq2SeqTokenizer,
    prompt_prefix: str,
    target_candidate: str,
    device: torch.device,
) -> float:
    """Computes log P(target_candidate | prompt_prefix) under Three-Hot Conditional."""
    prompt_ids = [(tok.sos_id, 0, 0)] + tok.encode(prompt_prefix, add_special_tokens=False)
    cand_ids = tok.encode(target_candidate, add_special_tokens=False)
    if not cand_ids:
        return 0.0

    full_ids = prompt_ids + cand_ids
    tgt = torch.tensor([full_ids], dtype=torch.long, device=device)
    tgt_input = tgt[:, :-1]
    tgt_target = tgt[:, 1:]

    with torch.no_grad():
        tgt_emb = model.embed_target(tgt_input)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
        h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

        b, t_len, d = h_t.size()
        h0 = model.h0.expand(b, t_len, d)

        h_i = torch.tanh(model.W_e(h_t) + model.W_h(h0))
        logp_i = nn.functional.log_softmax(model.head_i(h_i), dim=-1)

        embi = model.emb_i(tgt_target[:, :, 0])
        h_v = torch.tanh(model.W_e(embi) + model.W_h(h_i))
        logp_v = nn.functional.log_softmax(model.head_v(h_v), dim=-1)

        embv = model.emb_v(tgt_target[:, :, 1])
        h_f = torch.tanh(model.W_e(embv) + model.W_h(h_v))
        logp_f = nn.functional.log_softmax(model.head_f(h_f), dim=-1)

    start_idx = len(prompt_ids) - 1
    total_logp = 0.0
    for pos in range(start_idx, tgt_target.size(1)):
        t_i = tgt_target[0, pos, 0].item()
        t_v = tgt_target[0, pos, 1].item()
        t_f = tgt_target[0, pos, 2].item()

        lp = logp_i[0, pos, t_i].item()
        if t_v > 0:
            lp += logp_v[0, pos, t_v].item()
        if t_f > 0:
            lp += logp_f[0, pos, t_f].item()
        total_logp += lp

    return total_logp


def compute_sequence_log_prob_ind(
    model: Seq2SeqThreeHotIndependent,
    memory: torch.Tensor,
    tok: ThreeHotSeq2SeqTokenizer,
    prompt_prefix: str,
    target_candidate: str,
    device: torch.device,
) -> float:
    """Computes log P(target_candidate | prompt_prefix) under Three-Hot Independent."""
    prompt_ids = [(tok.sos_id, 0, 0)] + tok.encode(prompt_prefix, add_special_tokens=False)
    cand_ids = tok.encode(target_candidate, add_special_tokens=False)
    if not cand_ids:
        return 0.0

    full_ids = prompt_ids + cand_ids
    tgt = torch.tensor([full_ids], dtype=torch.long, device=device)
    tgt_input = tgt[:, :-1]
    tgt_target = tgt[:, 1:]

    with torch.no_grad():
        tgt_emb = model.embed_target(tgt_input)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
        h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

        logp_i = nn.functional.log_softmax(model.head_i(h_t), dim=-1)
        logp_v = nn.functional.log_softmax(model.head_v(h_t), dim=-1)
        logp_f = nn.functional.log_softmax(model.head_f(h_t), dim=-1)

    start_idx = len(prompt_ids) - 1
    total_logp = 0.0
    for pos in range(start_idx, tgt_target.size(1)):
        t_i = tgt_target[0, pos, 0].item()
        t_v = tgt_target[0, pos, 1].item()
        t_f = tgt_target[0, pos, 2].item()

        lp = logp_i[0, pos, t_i].item()
        if t_v > 0:
            lp += logp_v[0, pos, t_v].item()
        if t_f > 0:
            lp += logp_f[0, pos, t_f].item()
        total_logp += lp

    return total_logp


def compute_sequence_log_prob_fact(
    model: Seq2SeqHangulFactorizer,
    memory: torch.Tensor,
    tok: HangulFactorizerSeq2SeqTokenizer,
    prompt_prefix: str,
    target_candidate: str,
    device: torch.device,
) -> float:
    """Computes log P(target_candidate | prompt_prefix) under Hangul Factorizer."""
    prompt_ids = [[tok.sos_id] + [0] * 22] + tok.encode(prompt_prefix, add_special_tokens=False)
    cand_ids = tok.encode(target_candidate, add_special_tokens=False)
    if not cand_ids:
        return 0.0

    full_ids = prompt_ids + cand_ids
    tgt = torch.tensor([full_ids], dtype=torch.long, device=device)
    tgt_input = tgt[:, :-1]
    tgt_target = tgt[:, 1:]

    with torch.no_grad():
        tgt_emb = model.embed_target(tgt_input)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
        h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

        # Logits across all heads
        logits_list = [head(h_t) for head in model.heads]
        logp_list = [nn.functional.log_softmax(lg, dim=-1) for lg in logits_list]

    start_idx = len(prompt_ids) - 1
    total_logp = 0.0
    for pos in range(start_idx, tgt_target.size(1)):
        # Lane 0 (Script / non-ko)
        t0 = tgt_target[0, pos, 0].item()
        lp = logp_list[0][0, pos, t0].item()

        # If Korean syllable (lane 0 == 4), include primary jamo lanes 1, 2, 3
        if t0 == tok.hangul_flag_id:
            for k in (1, 2, 3):
                tk = tgt_target[0, pos, k].item()
                lp += logp_list[k][0, pos, tk].item()

        total_logp += lp

    return total_logp


def evaluate_klue_ner(
    model: nn.Module,
    tok: Any,
    src_tok: EnglishBPETokenizer,
    dataset,
    device: torch.device,
    is_factorizer: bool,
    is_conditional: bool,
    max_samples: int = 300,
) -> Dict[str, Any]:
    model.eval()

    # Generic encoder memory
    src_tensor = torch.tensor([src_tok.encode("Task: Named Entity Recognition", add_special_tokens=True)], device=device)
    with torch.no_grad():
        memory = model.encode(src_tensor)

    total_sentence_nll = 0.0
    total_sentence_jamos = 0

    entity_nll = 0.0
    entity_jamos = 0

    o_nll = 0.0
    o_jamos = 0

    cat_nll: Dict[str, float] = {k: 0.0 for k in NER_LABEL_NAMES.keys()}
    cat_jamos: Dict[str, int] = {k: 0 for k in NER_LABEL_NAMES.keys()}

    correct_class_predictions = 0
    total_class_evaluations = 0
    total_true_logp = 0.0

    eval_count = min(len(dataset), max_samples)
    print(f"Evaluating {eval_count} KLUE-NER samples...")

    for idx in tqdm(range(eval_count), desc="KLUE-NER"):
        item = dataset[idx]
        tokens = item["tokens"]  # character tokens
        tags = item["ner_tags"]  # tag per character
        raw_sentence = clean_ner_sentence(item["sentence"])

        if not raw_sentence.strip():
            continue

        # 1. Per-character / Per-jamo NLL
        if is_factorizer:
            seq = tok.encode(raw_sentence, add_special_tokens=True)
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with torch.no_grad():
                tgt_emb = model.embed_target(tgt_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
                h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)
                logits_list = [head(h_t) for head in model.heads]

            # Sequence length alignment (minus SOS and EOS)
            # tgt_target matches characters 0..len(tokens)-1
            for pos in range(min(len(tokens), tgt_target.size(1) - 1)):
                tag_id = tags[pos]
                tag_group = NER_TAG_MAP.get(tag_id, "O")
                is_hangul = tgt_target[0, pos, 0].item() == tok.hangul_flag_id

                if is_hangul:
                    loss_i = nn.functional.cross_entropy(logits_list[1][:, pos], tgt_target[:, pos, 1]).item()
                    loss_v = nn.functional.cross_entropy(logits_list[2][:, pos], tgt_target[:, pos, 2]).item()
                    loss_f = nn.functional.cross_entropy(logits_list[3][:, pos], tgt_target[:, pos, 3]).item()
                    char_nll = loss_i + loss_v + loss_f
                    num_j = 3

                    total_sentence_nll += char_nll
                    total_sentence_jamos += num_j

                    if tag_group == "O":
                        o_nll += char_nll
                        o_jamos += num_j
                    else:
                        entity_nll += char_nll
                        entity_jamos += num_j
                        if tag_group in cat_nll:
                            cat_nll[tag_group] += char_nll
                            cat_jamos[tag_group] += num_j
        else:
            seq = tok.encode(raw_sentence, add_special_tokens=True)
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with torch.no_grad():
                tgt_emb = model.embed_target(tgt_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
                h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

                if is_conditional:
                    b, t_len, d = h_t.size()
                    h0 = model.h0.expand(b, t_len, d)
                    h_i = torch.tanh(model.W_e(h_t) + model.W_h(h0))
                    embi = model.emb_i(tgt_target[:, :, 0])
                    h_v = torch.tanh(model.W_e(embi) + model.W_h(h_i))
                    embv = model.emb_v(tgt_target[:, :, 1])
                    h_f = torch.tanh(model.W_e(embv) + model.W_h(h_v))

                    logits_i = model.head_i(h_i)
                    logits_v = model.head_v(h_v)
                    logits_f = model.head_f(h_f)
                else:
                    logits_i = model.head_i(h_t)
                    logits_v = model.head_v(h_t)
                    logits_f = model.head_f(h_t)

            for pos in range(min(len(tokens), tgt_target.size(1) - 1)):
                tag_id = tags[pos]
                tag_group = NER_TAG_MAP.get(tag_id, "O")
                is_hangul = tgt_target[0, pos, 1].item() != 0

                if is_hangul:
                    loss_i = nn.functional.cross_entropy(logits_i[:, pos], tgt_target[:, pos, 0]).item()
                    loss_v = nn.functional.cross_entropy(logits_v[:, pos], tgt_target[:, pos, 1]).item()
                    loss_f = nn.functional.cross_entropy(logits_f[:, pos], tgt_target[:, pos, 2]).item()
                    char_nll = loss_i + loss_v + loss_f
                    num_j = 3

                    total_sentence_nll += char_nll
                    total_sentence_jamos += num_j

                    if tag_group == "O":
                        o_nll += char_nll
                        o_jamos += num_j
                    else:
                        entity_nll += char_nll
                        entity_jamos += num_j
                        if tag_group in cat_nll:
                            cat_nll[tag_group] += char_nll
                            cat_jamos[tag_group] += num_j

        # 2. Zero-Shot Entity Classification Ranking
        # Find first entity span in the sentence
        cur_entity_chars = []
        cur_entity_tag = None
        for char, tag_id in zip(tokens, tags):
            grp = NER_TAG_MAP.get(tag_id, "O")
            if grp != "O":
                cur_entity_chars.append(char)
                cur_entity_tag = grp
                if len(cur_entity_chars) >= 2:
                    break

        if cur_entity_chars and cur_entity_tag in NER_LABEL_NAMES:
            entity_str = "".join(cur_entity_chars).strip()
            true_label_kor = NER_LABEL_NAMES[cur_entity_tag]
            prompt = f"문장: {raw_sentence[:40]} | 개체명 '{entity_str}'의 유형: "

            cand_scores: Dict[str, float] = {}
            for tag_key, label_kor in NER_LABEL_NAMES.items():
                if is_factorizer:
                    score = compute_sequence_log_prob_fact(model, memory, tok, prompt, label_kor, device)
                elif is_conditional:
                    score = compute_sequence_log_prob_cond(model, memory, tok, prompt, label_kor, device)
                else:
                    score = compute_sequence_log_prob_ind(model, memory, tok, prompt, label_kor, device)
                cand_scores[label_kor] = score

            best_pred = max(cand_scores, key=cand_scores.get)
            if best_pred == true_label_kor:
                correct_class_predictions += 1
            total_true_logp += cand_scores[true_label_kor]
            total_class_evaluations += 1

    overall_bpj = calculate_bpj(total_sentence_nll, total_sentence_jamos)
    ent_bpj = calculate_bpj(entity_nll, entity_jamos)
    bg_bpj = calculate_bpj(o_nll, o_jamos)

    per_cat_bpj = {
        cat: calculate_bpj(cat_nll[cat], cat_jamos[cat])
        for cat in NER_LABEL_NAMES.keys()
    }

    acc = (correct_class_predictions / max(1, total_class_evaluations)) * 100.0
    mean_true_logp = total_true_logp / max(1, total_class_evaluations)

    return {
        "overall_bpj": round(overall_bpj, 4),
        "entity_bpj": round(ent_bpj, 4),
        "non_entity_bpj": round(bg_bpj, 4),
        "per_category_bpj": per_cat_bpj,
        "zero_shot_accuracy": round(acc, 2),
        "mean_log_prob": round(mean_true_logp, 4),
        "evaluated_spans": total_class_evaluations,
    }


def evaluate_klue_dp(
    model: nn.Module,
    tok: Any,
    src_tok: EnglishBPETokenizer,
    dataset,
    device: torch.device,
    is_factorizer: bool,
    is_conditional: bool,
    max_samples: int = 300,
) -> Dict[str, Any]:
    model.eval()
    src_tensor = torch.tensor([src_tok.encode("Task: Dependency Parsing", add_special_tokens=True)], device=device)
    with torch.no_grad():
        memory = model.encode(src_tensor)

    total_sentence_nll = 0.0
    total_sentence_jamos = 0

    role_nll = {r: 0.0 for r in DP_RELATION_NAMES.keys()}
    role_jamos = {r: 0 for r in DP_RELATION_NAMES.keys()}

    correct_dp_predictions = 0
    total_dp_evaluations = 0
    total_true_logp = 0.0

    eval_count = min(len(dataset), max_samples)
    print(f"Evaluating {eval_count} KLUE-DP samples...")

    for idx in tqdm(range(eval_count), desc="KLUE-DP"):
        item = dataset[idx]
        sentence = item["sentence"]
        word_forms = item["word_form"]
        heads = item["head"]
        deprels = item["deprel"]

        if not sentence.strip():
            continue

        # 1. Syntactic BPJ breakdown by word
        if is_factorizer:
            seq = tok.encode(sentence, add_special_tokens=True)
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with torch.no_grad():
                tgt_emb = model.embed_target(tgt_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
                h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)
                logits_list = [head(h_t) for head in model.heads]

            # Overall sentence BPJ
            ko_mask = tgt_target[0, :, 0] == tok.hangul_flag_id
            if ko_mask.any():
                loss_i = nn.functional.cross_entropy(logits_list[1][0], tgt_target[0, :, 1], reduction="none")[ko_mask].sum().item()
                loss_v = nn.functional.cross_entropy(logits_list[2][0], tgt_target[0, :, 2], reduction="none")[ko_mask].sum().item()
                loss_f = nn.functional.cross_entropy(logits_list[3][0], tgt_target[0, :, 3], reduction="none")[ko_mask].sum().item()
                total_sentence_nll += loss_i + loss_v + loss_f
                total_sentence_jamos += ko_mask.sum().item() * 3

            # Map word spans to roles
            char_cursor = 0
            for word, rel in zip(word_forms, deprels):
                w_pos = sentence.find(word, char_cursor)
                if w_pos != -1:
                    char_cursor = w_pos + len(word)
                    if rel in role_nll:
                        w_len = len(word)
                        span_slice = slice(w_pos, min(w_pos + w_len, tgt_target.size(1)))
                        span_mask = tgt_target[0, span_slice, 0] == tok.hangul_flag_id
                        if span_mask.any():
                            li = nn.functional.cross_entropy(logits_list[1][0, span_slice], tgt_target[0, span_slice, 1], reduction="none")[span_mask].sum().item()
                            lv = nn.functional.cross_entropy(logits_list[2][0, span_slice], tgt_target[0, span_slice, 2], reduction="none")[span_mask].sum().item()
                            lf = nn.functional.cross_entropy(logits_list[3][0, span_slice], tgt_target[0, span_slice, 3], reduction="none")[span_mask].sum().item()
                            role_nll[rel] += li + lv + lf
                            role_jamos[rel] += span_mask.sum().item() * 3
        else:
            seq = tok.encode(sentence, add_special_tokens=True)
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            with torch.no_grad():
                tgt_emb = model.embed_target(tgt_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1), device=device)
                h_t = model.decode_step(tgt_emb, memory, tgt_mask=tgt_mask)

                if is_conditional:
                    b, t_len, d = h_t.size()
                    h0 = model.h0.expand(b, t_len, d)
                    h_i = torch.tanh(model.W_e(h_t) + model.W_h(h0))
                    embi = model.emb_i(tgt_target[:, :, 0])
                    h_v = torch.tanh(model.W_e(embi) + model.W_h(h_i))
                    embv = model.emb_v(tgt_target[:, :, 1])
                    h_f = torch.tanh(model.W_e(embv) + model.W_h(h_v))

                    logits_i = model.head_i(h_i)
                    logits_v = model.head_v(h_v)
                    logits_f = model.head_f(h_f)
                else:
                    logits_i = model.head_i(h_t)
                    logits_v = model.head_v(h_t)
                    logits_f = model.head_f(h_t)

            ko_mask = tgt_target[0, :, 1] != 0
            if ko_mask.any():
                loss_i = nn.functional.cross_entropy(logits_i[0], tgt_target[0, :, 0], reduction="none")[ko_mask].sum().item()
                loss_v = nn.functional.cross_entropy(logits_v[0], tgt_target[0, :, 1], reduction="none")[ko_mask].sum().item()
                loss_f = nn.functional.cross_entropy(logits_f[0], tgt_target[0, :, 2], reduction="none")[ko_mask].sum().item()
                total_sentence_nll += loss_i + loss_v + loss_f
                total_sentence_jamos += ko_mask.sum().item() * 3

            char_cursor = 0
            for word, rel in zip(word_forms, deprels):
                w_pos = sentence.find(word, char_cursor)
                if w_pos != -1:
                    char_cursor = w_pos + len(word)
                    if rel in role_nll:
                        w_len = len(word)
                        span_slice = slice(w_pos, min(w_pos + w_len, tgt_target.size(1)))
                        span_mask = tgt_target[0, span_slice, 1] != 0
                        if span_mask.any():
                            li = nn.functional.cross_entropy(logits_i[0, span_slice], tgt_target[0, span_slice, 0], reduction="none")[span_mask].sum().item()
                            lv = nn.functional.cross_entropy(logits_v[0, span_slice], tgt_target[0, span_slice, 1], reduction="none")[span_mask].sum().item()
                            lf = nn.functional.cross_entropy(logits_f[0, span_slice], tgt_target[0, span_slice, 2], reduction="none")[span_mask].sum().item()
                            role_nll[rel] += li + lv + lf
                            role_jamos[rel] += span_mask.sum().item() * 3

        # 2. Zero-Shot Dependency Relation Probability Ranking
        # Pick first salient relation (Subject or Object)
        for w_idx, (word, head_idx, rel) in enumerate(zip(word_forms, heads, deprels)):
            if rel in DP_RELATION_NAMES and 1 <= head_idx <= len(word_forms):
                head_word = word_forms[head_idx - 1]
                true_rel_kor = DP_RELATION_NAMES[rel]
                prompt = f"문장: {sentence[:35]} | '{word}'와 '{head_word}'의 문법 관계: "

                cand_scores: Dict[str, float] = {}
                for rel_key, rel_kor in DP_RELATION_NAMES.items():
                    if is_factorizer:
                        score = compute_sequence_log_prob_fact(model, memory, tok, prompt, rel_kor, device)
                    elif is_conditional:
                        score = compute_sequence_log_prob_cond(model, memory, tok, prompt, rel_kor, device)
                    else:
                        score = compute_sequence_log_prob_ind(model, memory, tok, prompt, rel_kor, device)
                    cand_scores[rel_kor] = score

                best_pred = max(cand_scores, key=cand_scores.get)
                if best_pred == true_rel_kor:
                    correct_dp_predictions += 1
                total_true_logp += cand_scores[true_rel_kor]
                total_dp_evaluations += 1
                break  # evaluate 1 salient relation per sentence

    overall_bpj = calculate_bpj(total_sentence_nll, total_sentence_jamos)
    per_role_bpj = {
        role: calculate_bpj(role_nll[role], role_jamos[role])
        for role in DP_RELATION_NAMES.keys()
    }

    acc = (correct_dp_predictions / max(1, total_dp_evaluations)) * 100.0
    mean_true_logp = total_true_logp / max(1, total_dp_evaluations)

    return {
        "overall_bpj": round(overall_bpj, 4),
        "per_role_bpj": per_role_bpj,
        "zero_shot_accuracy": round(acc, 2),
        "mean_log_prob": round(mean_true_logp, 4),
        "evaluated_relations": total_dp_evaluations,
    }


def run_klue_benchmark(
    ckpt_dir: Path,
    output_json: Path,
    max_samples: int = 300,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading KLUE datasets...")
    ds_ner = load_dataset("klue", "ner", split=f"validation[:{max_samples}]")
    ds_dp = load_dataset("klue", "dp", split=f"validation[:{max_samples}]")

    src_spm = REPO_ROOT / "data/korean_seq2seq_bench/spm_en_30k.model"
    src_tok = EnglishBPETokenizer(src_spm)

    architectures = [
        ("three_hot_conditional", "Three-Hot (Conditional RNN, EACL 2023)", False, True),
        ("three_hot_independent", "Three-Hot (Independent Heads, Song et al.)", False, False),
        ("hangul_factorizer", "Hangul Factorizer (23-Lane Multi-Head)", True, False),
    ]

    results: Dict[str, Any] = {}

    for arch_key, arch_name, is_factorizer, is_conditional in architectures:
        ckpt_path = ckpt_dir / f"{arch_key}_best.pt"
        print(f"\n=======================================================")
        print(f"Evaluating {arch_name} on KLUE...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        non_ko_vocab = ckpt["non_ko_vocab"]
        tgt_vocab_sizes = ckpt["tgt_vocab_sizes"]

        if is_factorizer:
            tgt_tok = HangulFactorizerSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqHangulFactorizer(src_tok.vocab_size, tgt_vocab_sizes).to(device)
        elif is_conditional:
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotConditional(src_tok.vocab_size, tgt_vocab_sizes).to(device)
        else:
            tgt_tok = ThreeHotSeq2SeqTokenizer(non_ko_vocab)
            model = Seq2SeqThreeHotIndependent(src_tok.vocab_size, tgt_vocab_sizes).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        # 1. KLUE-NER
        ner_res = evaluate_klue_ner(
            model, tgt_tok, src_tok, ds_ner, device, is_factorizer, is_conditional, max_samples=max_samples
        )

        # 2. KLUE-DP
        dp_res = evaluate_klue_dp(
            model, tgt_tok, src_tok, ds_dp, device, is_factorizer, is_conditional, max_samples=max_samples
        )

        results[arch_key] = {
            "name": arch_name,
            "ner": ner_res,
            "dp": dp_res,
        }

        print(f"  -> NER Overall BPJ: {ner_res['overall_bpj']} | Entity BPJ: {ner_res['entity_bpj']} | Non-Entity BPJ: {ner_res['non_entity_bpj']}")
        print(f"  -> NER Zero-Shot Acc: {ner_res['zero_shot_accuracy']}% | Mean LogP: {ner_res['mean_log_prob']}")
        print(f"  -> DP Overall BPJ: {dp_res['overall_bpj']} | Subject BPJ: {dp_res['per_role_bpj'].get('NP_SBJ')} | Predicate BPJ: {dp_res['per_role_bpj'].get('VP')}")
        print(f"  -> DP Zero-Shot Acc: {dp_res['zero_shot_accuracy']}% | Mean LogP: {dp_res['mean_log_prob']}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nKLUE zero-shot benchmark results saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot probabilities on KLUE-NER and KLUE-DP.")
    parser.add_argument("--ckpt_dir", type=str, default="out_seq2seq_bench")
    parser.add_argument("--output_json", type=str, default="out_seq2seq_bench/klue_zeroshot_results.json")
    parser.add_argument("--max_samples", type=int, default=300)
    args = parser.parse_args()

    run_klue_benchmark(Path(args.ckpt_dir), Path(args.output_json), max_samples=args.max_samples)


if __name__ == "__main__":
    main()

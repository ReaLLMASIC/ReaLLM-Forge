#!/usr/bin/env python3
"""Evaluation and metric computation for English-to-Korean Seq2Seq models.

Implements:
1. Paper-compliant canonicalization:
   - Decomposes Hangul syllables into Unicode Compatibility Jamo (0x3131-0x3163).
   - Retains non-Korean characters as-is.
   - Strips all punctuation characters.
2. Metrics:
   - BPJ (Bits-Per-Jamo): Cross-entropy NLL / (ln 2 * N_jamo).
   - BLEU: Word-level 4-gram BLEU over whitespace-delimited jamo words.
   - chrF: Character n-gram F-score on jamo strings with char_order=18.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple
from jamo import h2j, j2hcj
import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def canonicalize(text: str) -> str:
    """Canonicalize text according to EACL 2023 Appendix A & Section 4.

    1. Decompose Hangul into Compatibility Jamo
    2. Strip punctuation
    3. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    # 1. Decompose to Compatibility Jamo
    decomposed = j2hcj(h2j(text))
    # 2. Strip punctuation
    stripped = re.sub(r"[^\w\s]", "", decomposed)
    # 3. Normalize whitespace
    return re.sub(r"\s+", " ", stripped).strip()


def calculate_bpj(total_nll: float, total_jamos: int) -> float:
    """Compute Bits-Per-Jamo from total negative log-likelihood."""
    if total_jamos <= 0:
        return 0.0
    return total_nll / (math.log(2.0) * total_jamos)


def evaluate_corpus_metrics(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> Dict[str, float]:
    """Computes BLEU and chrF on canonicalized hypotheses and references."""
    canon_hyps = [canonicalize(h) for h in hypotheses]
    canon_refs = [canonicalize(r) for r in references]

    # Ensure non-empty lines for metrics
    filtered_hyps = []
    filtered_refs = []
    for h, r in zip(canon_hyps, canon_refs):
        h_str = h if h.strip() else "<empty>"
        r_str = r if r.strip() else "<empty>"
        filtered_hyps.append(h_str)
        filtered_refs.append(r_str)

    # BLEU (word 4-gram)
    bleu_res = sacrebleu.corpus_bleu(
        filtered_hyps,
        [filtered_refs],
        smooth_method="exp",
        lowercase=False,
        use_effective_order=True,
    )

    # chrF with char_order=18 (representing 6 syllables)
    chrf_res = sacrebleu.corpus_chrf(
        filtered_hyps,
        [filtered_refs],
        char_order=18,
    )

    return {
        "BLEU": float(bleu_res.score),
        "chrF": float(chrf_res.score),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    is_factorizer: bool = False,
    is_conditional: bool = False,
    max_eval_samples: Optional[int] = None,
    max_gen_len: int = 100,
) -> Dict[str, float]:
    """Evaluates BPJ, BLEU, and chrF on a validation/test dataloader."""
    model.eval()
    total_jamo_nll = 0.0
    total_jamo_count = 0

    hypotheses: List[str] = []
    references: List[str] = []

    samples_processed = 0

    for batch in tqdm(val_loader, desc="Evaluating"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        ref_texts = batch["ref_text"]

        b_size = src.size(0)

        # 1. Compute BPJ (Loss on teacher-forced targets)
        # Shift targets for next-token prediction: input tgt[:, :-1], target tgt[:, 1:]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        if is_factorizer:
            logits_list = model(src, tgt_input)
            # tgt_target shape: (B, T, 23)
            # Korean syllables are flagged with hangul_flag_id (4) in Lane 0
            ko_mask = tgt_target[:, :, 0] == 4
            if ko_mask.any():
                loss_i = torch.nn.functional.cross_entropy(
                    logits_list[1].transpose(1, 2), tgt_target[:, :, 1], reduction="none"
                )[ko_mask].sum().item()
                loss_v = torch.nn.functional.cross_entropy(
                    logits_list[2].transpose(1, 2), tgt_target[:, :, 2], reduction="none"
                )[ko_mask].sum().item()
                loss_f = torch.nn.functional.cross_entropy(
                    logits_list[3].transpose(1, 2), tgt_target[:, :, 3], reduction="none"
                )[ko_mask].sum().item()
                num_syllables = ko_mask.sum().item()
                total_jamo_nll += loss_i + loss_v + loss_f
                total_jamo_count += num_syllables * 3
        else:
            logits_i, logits_v, logits_f = model(src, tgt_input)
            # In Three-Hot, Korean syllables have non-zero medial vowel (Lane 1 != 0)
            ko_mask = tgt_target[:, :, 1] != 0
            if ko_mask.any():
                loss_i = torch.nn.functional.cross_entropy(
                    logits_i.transpose(1, 2), tgt_target[:, :, 0], reduction="none"
                )[ko_mask].sum().item()
                loss_v = torch.nn.functional.cross_entropy(
                    logits_v.transpose(1, 2), tgt_target[:, :, 1], reduction="none"
                )[ko_mask].sum().item()
                loss_f = torch.nn.functional.cross_entropy(
                    logits_f.transpose(1, 2), tgt_target[:, :, 2], reduction="none"
                )[ko_mask].sum().item()
                num_syllables = ko_mask.sum().item()
                total_jamo_nll += loss_i + loss_v + loss_f
                total_jamo_count += num_syllables * 3

        # 2. Generate translations for generation metrics
        gen_tokens = model.generate(src, max_len=max_gen_len)  # (B, T_out, lanes)
        gen_list = gen_tokens.cpu().tolist()
        for idx in range(b_size):
            pred_text = tgt_tokenizer.decode(gen_list[idx])
            hypotheses.append(pred_text)
            references.append(ref_texts[idx])

        samples_processed += b_size
        if max_eval_samples and samples_processed >= max_eval_samples:
            break

    bpj = calculate_bpj(total_jamo_nll, total_jamo_count)
    gen_metrics = evaluate_corpus_metrics(hypotheses, references)

    return {
        "BPJ": bpj,
        "BLEU": gen_metrics["BLEU"],
        "chrF": gen_metrics["chrF"],
        "samples": len(hypotheses),
    }

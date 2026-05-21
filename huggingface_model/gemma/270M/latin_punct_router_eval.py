"""Build a manual LM-head router for Gemma 270M and evaluate on OPUS-100 en-es.

Base groups:
1) latin: tokens containing Latin-script characters.
2) punct: punctuation-only tokens (Unicode punctuation + common ES/EN punctuation).
3) other: everything else (including byte/special tokens).

Routing modes:
- three_way: latin vs punct vs other.
- latin_punct_vs_other: (latin U punct) vs other.
- latin_vs_punct_only: latin vs punct (never routes to other).
- latin_punct_only: only score within (latin U punct), no routing decision.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_ES_EN_PUNCT = {
    ".", ",", "!", "?", ":", ";", "'", '"', "-", "—", "–", "(", ")", "[", "]", "{", "}",
    "¡", "¿", "…", "«", "»", "‹", "›", "`", "´", "/", "\\", "|", "@", "#", "&", "*", "%",
    "_",
}
BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
FEW_SHOT_EXAMPLES = [
    ("Good morning.", "Buenos días."),
    ("Where is the train station?", "¿Dónde está la estación de tren?"),
    ("I would like a glass of water.", "Me gustaría un vaso de agua."),
]
ANSI_RESET = "\033[0m"
ANSI_GEN = "\033[92m"
ANSI_USER = "\033[96m"


@dataclass
class EvalStats:
    total_target_tokens: int = 0
    top1_correct_routed: int = 0
    top1_correct_full: int = 0
    route_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class ExamplePair:
    english: str
    reference: str
    full_output: str
    routed_output: str


@dataclass
class QuantConfig:
    bits: int
    mode: str  # vector_symmetric | vector_asymmetric | group32_symmetric | group32_asymmetric


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _is_latin_char(ch: str) -> bool:
    if not ch:
        return False
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return "LATIN" in name


def _is_common_punctuation(ch: str) -> bool:
    if ch in COMMON_ES_EN_PUNCT:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_punctuation_only_text(text: str) -> bool:
    if not text:
        return False
    return all(c.isspace() or _is_common_punctuation(c) for c in text)


def _build_token_groups(tokenizer: AutoTokenizer, vocab_size: int) -> Dict[str, List[int]]:
    groups = {"latin": [], "punct": [], "other": [], "byte": []}
    raw_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

    for token_id, raw_token in enumerate(raw_tokens):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if BYTE_TOKEN_RE.match(raw_token or ""):
            groups["byte"].append(token_id)
        if any(_is_latin_char(c) for c in decoded):
            groups["latin"].append(token_id)
        elif _is_punctuation_only_text(decoded):
            groups["punct"].append(token_id)
        else:
            groups["other"].append(token_id)
    return groups


def _trim_latin_token_ids(
    tokenizer: AutoTokenizer,
    latin_ids: Sequence[int],
    trim_percent: float,
    trim_strategy: str = "longest_bytes",
) -> List[int]:
    if trim_percent <= 0:
        return list(latin_ids)
    if trim_percent >= 100:
        return []

    trim_count = int(len(latin_ids) * (trim_percent / 100.0))
    if trim_strategy == "longest_bytes":
        scored = []
        for token_id in latin_ids:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            byte_len = len(decoded.encode("utf-8"))
            scored.append((byte_len, token_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [token_id for _, token_id in scored[trim_count:]]
    elif trim_strategy == "highest_id":
        sorted_ids = sorted(latin_ids, reverse=True)
        keep = sorted_ids[trim_count:]
    else:
        raise ValueError(f"Unknown trim_strategy: {trim_strategy}")
    return keep


def _print_group_table(groups: Dict[str, Sequence[int]], vocab_size: int) -> None:
    headers = ["Group", "Raw token count", "% of vocab"]
    rows = []
    for key in ("latin", "punct", "other", "byte (subset of other)"):
        src_key = "byte" if key.startswith("byte") else key
        count = len(groups[src_key])
        pct = 100.0 * count / vocab_size
        rows.append((key, count, f"{pct:.2f}%"))

    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print("\nToken routing groups for vocabulary")
    print(line)
    print("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line)


def _compute_group_prototypes(
    lm_head_weight: torch.Tensor,
    route_groups: Dict[str, Sequence[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, id_list in route_groups.items():
        if not id_list:
            raise ValueError(f"Routing group '{key}' is empty; cannot compute average prototype.")
        ids = torch.tensor(id_list, dtype=torch.long, device=lm_head_weight.device)
        avg = lm_head_weight.index_select(0, ids).mean(dim=0)
        out[key] = torch.nn.functional.normalize(avg, dim=0).to(device)
    return out


def _router_scores(
    hidden_last: torch.Tensor,
    prototypes: Dict[str, torch.Tensor],
    ordered_names: Sequence[str],
    route_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    mat = torch.stack([prototypes[name] for name in ordered_names], dim=0).to(dtype=torch.float32)
    if route_scales is not None:
        mat = mat * route_scales.to(dtype=torch.float32).unsqueeze(-1)
    hidden = hidden_last.to(dtype=torch.float32)
    return torch.matmul(hidden, mat.T)


def _build_route_groups(
    base_groups: Dict[str, Sequence[int]],
    route_mode: str,
    tokenizer: AutoTokenizer | None = None,
    latin_trim_percent: float = 0.0,
    latin_trim_strategy: str = "longest_bytes",
) -> Dict[str, List[int]]:
    latin_ids = list(base_groups["latin"])
    if latin_trim_percent > 0:
        if tokenizer is None:
            raise ValueError("tokenizer is required when latin_trim_percent > 0")
        latin_ids = _trim_latin_token_ids(tokenizer, latin_ids, latin_trim_percent, latin_trim_strategy)

    if route_mode == "three_way":
        return {
            "latin": latin_ids,
            "punct": list(base_groups["punct"]),
            "other": list(base_groups["other"]),
        }
    if route_mode == "latin_punct_vs_other":
        latin_punct = sorted(set(latin_ids) | set(base_groups["punct"]))
        return {
            "latin_punct": latin_punct,
            "other": list(base_groups["other"]),
        }
    if route_mode == "latin_vs_punct_only":
        return {
            "latin": latin_ids,
            "punct": list(base_groups["punct"]),
        }
    if route_mode == "latin_punct_only":
        latin_punct = sorted(set(latin_ids) | set(base_groups["punct"]))
        return {
            "latin_punct_only": latin_punct,
        }
    raise ValueError(f"Unknown route_mode: {route_mode}")


def _build_3shot_prompt(english_text: str) -> str:
    lines = [
        "Translate English to Spanish.",
        "",
    ]
    for src, tgt in FEW_SHOT_EXAMPLES:
        lines.append(f"English: {src}")
        lines.append(f"Spanish: {tgt}")
        lines.append("")
    lines.append(f"English: {english_text}")
    lines.append("Spanish:")
    return "\n".join(lines)


def _evaluate_router(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    byte_ids: Sequence[int],
    prototypes: Dict[str, torch.Tensor],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    device: torch.device,
    byte_fallback: bool,
    route_scales: torch.Tensor | None = None,
) -> EvalStats:
    stats = EvalStats()
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
    lm_head_weight = _normalize_rows(model.lm_head.weight.detach())

    route_names = list(route_groups.keys())
    byte_set = set(byte_ids)
    group_ids = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in route_groups.items()
    }

    for ex in ds.select(range(min(max_samples, len(ds)))):
        en = ex["translation"]["en"]
        es = ex["translation"]["es"]
        prompt = _build_3shot_prompt(en)

        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(es, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if target_ids.numel() == 0:
            continue

        steps = min(max_target_tokens, target_ids.size(1))
        running = prompt_ids

        for idx in range(steps):
            with torch.no_grad():
                out = model(running, output_hidden_states=True, use_cache=False)
            hidden_last = out.hidden_states[-1][:, -1, :]  # post-final layernorm state
            if len(route_names) == 1:
                chosen = route_names[0]
            else:
                route = _router_scores(hidden_last, prototypes, route_names, route_scales).argmax(dim=-1).item()
                chosen = route_names[route]
            stats.route_counts[chosen] += 1

            candidate_ids_list = list(route_groups[chosen])
            if byte_fallback and chosen != "other":
                candidate_ids_list = sorted(set(candidate_ids_list) | byte_set)
            if not candidate_ids_list:
                candidate_ids_list = list(byte_set)
            candidate_ids = torch.tensor(candidate_ids_list, dtype=torch.long, device=device)

            candidate_weight = lm_head_weight.index_select(0, candidate_ids)
            logits = torch.matmul(torch.nn.functional.normalize(hidden_last, dim=-1), candidate_weight.T)
            pred_local = torch.argmax(logits, dim=-1)
            pred_id = candidate_ids[pred_local].item()
            full_logits = torch.matmul(torch.nn.functional.normalize(hidden_last, dim=-1), lm_head_weight.T)
            pred_id_full = torch.argmax(full_logits, dim=-1).item()
            gold_id = target_ids[0, idx].item()

            stats.total_target_tokens += 1
            if pred_id == gold_id:
                stats.top1_correct_routed += 1
            if pred_id_full == gold_id:
                stats.top1_correct_full += 1

            next_gold = target_ids[:, idx : idx + 1]
            running = torch.cat([running, next_gold], dim=1)

    return stats


def _next_token_id(
    hidden_last: torch.Tensor,
    lm_head_weight: torch.Tensor,
    route_names: Sequence[str],
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_set: set[int],
    byte_fallback: bool,
    routed: bool,
    device: torch.device,
    route_scales: torch.Tensor | None = None,
) -> int:
    hidden_norm = torch.nn.functional.normalize(hidden_last, dim=-1)
    if not routed:
        logits_full = torch.matmul(hidden_norm, lm_head_weight.T)
        return torch.argmax(logits_full, dim=-1).item()

    if len(route_names) == 1:
        chosen = route_names[0]
    else:
        route = _router_scores(hidden_last, prototypes, route_names, route_scales).argmax(dim=-1).item()
        chosen = route_names[route]
    candidate_ids_list = list(route_groups[chosen])
    if byte_fallback and chosen != "other":
        candidate_ids_list = sorted(set(candidate_ids_list) | byte_set)
    if not candidate_ids_list:
        candidate_ids_list = list(byte_set)

    candidate_ids = torch.tensor(candidate_ids_list, dtype=torch.long, device=device)
    candidate_weight = lm_head_weight.index_select(0, candidate_ids)
    logits = torch.matmul(hidden_norm, candidate_weight.T)
    pred_local = torch.argmax(logits, dim=-1)
    return candidate_ids[pred_local].item()


def _highlight_generated(full_text: str, prompt: str, color: str = ANSI_GEN) -> str:
    if full_text.startswith(prompt):
        return f"{prompt}{color}{full_text[len(prompt):]}{ANSI_RESET}"
    return f"{color}{full_text}{ANSI_RESET}"


def _generate_translation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    english_text: str,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    max_new_tokens: int,
    routed: bool,
    byte_fallback: bool,
    device: torch.device,
    route_scales: torch.Tensor | None = None,
) -> tuple[str, str]:
    prompt = _build_3shot_prompt(english_text)
    running = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)

    route_names = list(route_groups.keys())
    byte_set = set(byte_ids)
    lm_head_weight = _normalize_rows(model.lm_head.weight.detach())
    eos = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(running, output_hidden_states=True, use_cache=False)
        hidden_last = out.hidden_states[-1][:, -1, :]
        next_id = _next_token_id(
            hidden_last=hidden_last,
            lm_head_weight=lm_head_weight,
            route_names=route_names,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_set=byte_set,
            byte_fallback=byte_fallback,
            routed=routed,
            device=device,
            route_scales=route_scales,
        )
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        running = torch.cat([running, next_token], dim=1)
        if eos is not None and next_id == eos:
            break

    full_text = tokenizer.decode(running[0], skip_special_tokens=True)
    if "Spanish:" in full_text:
        generated = full_text.split("Spanish:", 1)[1].strip()
    else:
        generated = full_text.strip()
    return generated, _highlight_generated(full_text, prompt)


def _print_validation_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    example_split: str,
    num_examples: int,
    max_new_tokens: int,
    byte_fallback: bool,
    device: torch.device,
    route_scales: torch.Tensor | None = None,
) -> None:
    if num_examples <= 0:
        return
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=example_split)
    count = min(num_examples, len(ds))

    print("\nValidation examples (before=full LM head, after=routed LM head)")
    for idx, ex in enumerate(ds.select(range(count)), start=1):
        en = ex["translation"]["en"]
        es_ref = ex["translation"]["es"]
        pred_full, pred_full_colored = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=False,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        pred_routed, pred_routed_colored = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=True,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        print(f"\nExample {idx}")
        print(f"EN: {en}")
        print(f"REF: {es_ref}")
        print(f"Before (full): {pred_full}")
        print(f"After  (routed): {pred_routed}")
        print(f"Before (full, generated highlighted): {pred_full_colored}")
        print(f"After  (routed, generated highlighted): {pred_routed_colored}")


def _run_chat_mode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    example_max_new_tokens: int,
    byte_fallback: bool,
    device: torch.device,
    route_scales: torch.Tensor | None = None,
) -> None:
    print("\nChat mode: enter English text and get Spanish translations.")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input(f"\n{ANSI_USER}EN>{ANSI_RESET} ").strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        pred_full, pred_full_colored = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=user_text,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=example_max_new_tokens,
            routed=False,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        pred_routed, pred_routed_colored = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=user_text,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=example_max_new_tokens,
            routed=True,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        print(f"{ANSI_USER}EN input:{ANSI_RESET} {ANSI_USER}{user_text}{ANSI_RESET}")
        print(f"ES (full):   {pred_full}")
        print(f"ES (routed): {pred_routed}")
        print(f"ES (full, generated highlighted):   {pred_full_colored}")
        print(f"ES (routed, generated highlighted): {pred_routed_colored}")


def _collect_example_pairs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    route_groups: Dict[str, Sequence[int]],
    prototypes: Dict[str, torch.Tensor],
    byte_ids: Sequence[int],
    example_split: str,
    num_examples: int,
    max_new_tokens: int,
    byte_fallback: bool,
    device: torch.device,
    route_scales: torch.Tensor | None = None,
) -> List[ExamplePair]:
    pairs: List[ExamplePair] = []
    if num_examples <= 0:
        return pairs
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=example_split)
    count = min(num_examples, len(ds))
    for ex in ds.select(range(count)):
        en = ex["translation"]["en"]
        ref = ex["translation"]["es"]
        pred_full, _ = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=False,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        pred_routed, _ = _generate_translation(
            model=model,
            tokenizer=tokenizer,
            english_text=en,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            max_new_tokens=max_new_tokens,
            routed=True,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=route_scales,
        )
        pairs.append(ExamplePair(english=en, reference=ref, full_output=pred_full, routed_output=pred_routed))
    return pairs


def _ascii_prefix_before_newline(text: str) -> str:
    return text.split("\n", 1)[0].encode("ascii", errors="ignore").decode("ascii")


def _ascii_diff_score(pred: str, reference: str) -> float:
    pred_ascii = _ascii_prefix_before_newline(pred)
    ref_ascii = _ascii_prefix_before_newline(reference)
    if not pred_ascii and not ref_ascii:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(a=pred_ascii, b=ref_ascii).ratio()


def _latin_token_json_records(tokenizer: AutoTokenizer, latin_ids: Sequence[int]) -> List[dict]:
    records = []
    for token_id in latin_ids:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        records.append(
            {
                "id": int(token_id),
                "token": decoded,
                "length_bytes": len(decoded.encode("utf-8")),
            }
        )
    records.sort(key=lambda x: x["length_bytes"], reverse=True)
    return records


def _run_latin_trim_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_groups: Dict[str, Sequence[int]],
    route_mode: str,
    byte_ids: Sequence[int],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    example_split: str,
    sweep_examples: int,
    example_max_new_tokens: int,
    byte_fallback: bool,
    device: torch.device,
    report_dir: str,
    sweep_max_percent: int,
    sweep_step_percent: int,
    latin_trim_strategy: str,
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    percents = list(range(0, sweep_max_percent + 1, sweep_step_percent))
    rows = []
    examples_by_percent: Dict[int, List[ExamplePair]] = {}

    print("\nLatin trim sweep report")
    print(f"- percents: {percents}")
    for pct in percents:
        trimmed_latin_ids = _trim_latin_token_ids(
            tokenizer, base_groups["latin"], float(pct), trim_strategy=latin_trim_strategy
        )
        route_groups = _build_route_groups(
            base_groups=base_groups,
            route_mode=route_mode,
            tokenizer=tokenizer,
            latin_trim_percent=float(pct),
            latin_trim_strategy=latin_trim_strategy,
        )
        prototypes = _compute_group_prototypes(model.lm_head.weight.detach(), route_groups, device)
        stats = _evaluate_router(
            model=model,
            tokenizer=tokenizer,
            route_groups=route_groups,
            byte_ids=byte_ids,
            prototypes=prototypes,
            split=split,
            max_samples=max_samples,
            max_target_tokens=max_target_tokens,
            device=device,
            byte_fallback=byte_fallback,
            route_scales=None,
        )
        acc_routed = 100.0 * stats.top1_correct_routed / max(1, stats.total_target_tokens)
        acc_full = 100.0 * stats.top1_correct_full / max(1, stats.total_target_tokens)
        rows.append((pct, acc_full, acc_routed, stats.total_target_tokens, len(trimmed_latin_ids)))
        examples_by_percent[pct] = _collect_example_pairs(
            model=model,
            tokenizer=tokenizer,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=byte_ids,
            example_split=example_split,
            num_examples=sweep_examples,
            max_new_tokens=example_max_new_tokens,
            byte_fallback=byte_fallback,
            device=device,
            route_scales=None,
        )
        ascii_full = []
        ascii_routed = []
        for ex in examples_by_percent[pct]:
            ascii_full.append(_ascii_diff_score(ex.full_output, ex.reference))
            ascii_routed.append(_ascii_diff_score(ex.routed_output, ex.reference))
        avg_ascii_diff_full = sum(ascii_full) / max(1, len(ascii_full))
        avg_ascii_diff_routed = sum(ascii_routed) / max(1, len(ascii_routed))
        rows[-1] = (
            rows[-1][0],
            rows[-1][1],
            rows[-1][2],
            rows[-1][3],
            rows[-1][4],
            avg_ascii_diff_full,
            avg_ascii_diff_routed,
        )
        print(f"- trim={pct}% full_acc={acc_full:.2f}% routed_acc={acc_routed:.2f}% tokens={stats.total_target_tokens}")
        percent_path = os.path.join(report_dir, f"latin_trim_{pct:02d}.txt")
        with open(percent_path, "w", encoding="utf-8") as pf:
            pf.write(
                f"trim={pct}% | full_top1={acc_full:.2f}% | routed_top1={acc_routed:.2f}% | "
                f"eval_tokens={stats.total_target_tokens} | latin_vocab_count={len(trimmed_latin_ids)}\n"
            )
            pf.write(
                f"avg_ascii_diff_before_newline_full={avg_ascii_diff_full:.4f} | "
                f"avg_ascii_diff_before_newline_routed={avg_ascii_diff_routed:.4f}\n\n"
            )
            for ex in examples_by_percent[pct]:
                pf.write(f"EN: {ex.english}\n")
                pf.write(f"REF: {ex.reference}\n")
                pf.write(f"ES(full): {ex.full_output}\n")
                pf.write(f"ES(routed): {ex.routed_output}\n")
                pf.write("\n")
        print(f"  wrote per-percent details: {percent_path}")
        latin_json_path = os.path.join(report_dir, f"latin_trim_{pct:02d}_latin_tokens.json")
        latin_records = _latin_token_json_records(tokenizer, trimmed_latin_ids)
        with open(latin_json_path, "w", encoding="utf-8") as jf:
            json.dump(latin_records, jf, ensure_ascii=False, indent=2)
        print(f"  wrote per-percent latin token json: {latin_json_path}")

    csv_path = os.path.join(report_dir, "latin_trim_sweep.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "latin_trim_percent",
                "top1_full_percent",
                "top1_routed_percent",
                "eval_tokens",
                "latin_vocab_count",
                "avg_ascii_diff_before_newline_full",
                "avg_ascii_diff_before_newline_routed",
            ]
        )
        writer.writerows(rows)

    fig_path = os.path.join(report_dir, "latin_trim_sweep_accuracy.png")
    x = [r[0] for r in rows]
    y_full = [r[1] for r in rows]
    y_routed = [r[2] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_full, marker="o", label="Full LM head")
    plt.plot(x, y_routed, marker="o", label="Routed")
    plt.xlabel("Latin trim percent (longest UTF-8 tokens removed)")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Latin trim sweep: top-1 accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    report_path = os.path.join(report_dir, "latin_trim_sweep_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Latin trim sweep final report\n")
        f.write("============================\n\n")
        for pct, acc_full, acc_routed, tok_count, latin_count, ascii_full, ascii_routed in rows:
            f.write(
                f"trim={pct}% | full_top1={acc_full:.2f}% | routed_top1={acc_routed:.2f}% | "
                f"eval_tokens={tok_count} | latin_vocab_count={latin_count} | "
                f"ascii_diff_full={ascii_full:.4f} | ascii_diff_routed={ascii_routed:.4f}\n"
            )
            for ex in examples_by_percent[pct]:
                f.write(f"  EN: {ex.english}\n")
                f.write(f"  REF: {ex.reference}\n")
                f.write(f"  ES(full): {ex.full_output}\n")
                f.write(f"  ES(routed): {ex.routed_output}\n")
            f.write("\n")

    print(f"- wrote CSV: {csv_path}")
    print(f"- wrote graph: {fig_path}")
    print(f"- wrote report: {report_path}")


def _quantize_tensor(x: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    qmin = 0
    qmax = (1 << bits) - 1
    if symmetric:
        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / float((1 << (bits - 1)) - 1)
        q = torch.round(x / scale).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        return q * scale

    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / float(qmax - qmin)).clamp_min(1e-8)
    zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return (q - zero_point) * scale


def _quantize_weights(weight: torch.Tensor, cfg: QuantConfig, group_size: int) -> torch.Tensor:
    mode = cfg.mode
    if mode == "vector_symmetric":
        return _quantize_tensor(weight, cfg.bits, symmetric=True)
    if mode == "vector_asymmetric":
        return _quantize_tensor(weight, cfg.bits, symmetric=False)
    if mode in {"group32_symmetric", "group32_asymmetric"}:
        symmetric = mode.endswith("symmetric")
        chunks = []
        for start in range(0, weight.size(1), group_size):
            chunk = weight[:, start : start + group_size]
            chunks.append(_quantize_tensor(chunk, cfg.bits, symmetric=symmetric))
        return torch.cat(chunks, dim=-1)
    raise ValueError(f"Unknown quantization mode: {mode}")


def _evaluate_candidate_weight_top1(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    candidate_ids: torch.Tensor,
    candidate_weight_normalized: torch.Tensor,
    split: str,
    max_samples: int,
    max_target_tokens: int,
    device: torch.device,
) -> float:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
    correct = 0
    total = 0
    for ex in ds.select(range(min(max_samples, len(ds)))):
        en = ex["translation"]["en"]
        es = ex["translation"]["es"]
        prompt = _build_3shot_prompt(en)
        running = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(es, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if target_ids.numel() == 0:
            continue
        steps = min(max_target_tokens, target_ids.size(1))
        for idx in range(steps):
            with torch.no_grad():
                out = model(running, output_hidden_states=True, use_cache=False)
            hidden_last = torch.nn.functional.normalize(out.hidden_states[-1][:, -1, :], dim=-1)
            logits = torch.matmul(hidden_last, candidate_weight_normalized.T)
            pred_local = torch.argmax(logits, dim=-1)
            pred_id = candidate_ids[pred_local].item()
            gold_id = target_ids[0, idx].item()
            if pred_id == gold_id:
                correct += 1
            total += 1
            running = torch.cat([running, target_ids[:, idx : idx + 1]], dim=1)
    return 100.0 * correct / max(1, total)


def _evaluate_two_pass_candidate_weight_top1(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    candidate_ids: torch.Tensor,
    first_pass_weight_normalized: torch.Tensor,
    second_pass_weight: torch.Tensor,
    second_pass_topn: int,
    split: str,
    max_samples: int,
    max_target_tokens: int,
    device: torch.device,
) -> float:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
    correct = 0
    total = 0
    topn = max(1, min(second_pass_topn, candidate_ids.numel()))
    for ex in ds.select(range(min(max_samples, len(ds)))):
        en = ex["translation"]["en"]
        es = ex["translation"]["es"]
        prompt = _build_3shot_prompt(en)
        running = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(es, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if target_ids.numel() == 0:
            continue
        steps = min(max_target_tokens, target_ids.size(1))
        for idx in range(steps):
            with torch.no_grad():
                out = model(running, output_hidden_states=True, use_cache=False)
            hidden_last = torch.nn.functional.normalize(out.hidden_states[-1][:, -1, :], dim=-1)
            coarse_logits = torch.matmul(hidden_last, first_pass_weight_normalized.T)
            shortlist_local = torch.topk(coarse_logits, k=topn, dim=-1).indices[0]
            shortlist_weight = second_pass_weight.index_select(0, shortlist_local)
            rerank_logits = torch.matmul(hidden_last.to(shortlist_weight.dtype), shortlist_weight.T).to(torch.float32)
            pred_local = shortlist_local[torch.argmax(rerank_logits, dim=-1).item()]
            pred_id = candidate_ids[pred_local].item()
            gold_id = target_ids[0, idx].item()
            if pred_id == gold_id:
                correct += 1
            total += 1
            running = torch.cat([running, target_ids[:, idx : idx + 1]], dim=1)
    return 100.0 * correct / max(1, total)


def _run_quantization_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_groups: Dict[str, Sequence[int]],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    group_size: int,
    report_dir: str,
    device: torch.device,
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    candidate_list = sorted(set(base_groups["latin"]) | set(base_groups["punct"]) | set(base_groups["byte"]))
    candidate_ids = torch.tensor(candidate_list, dtype=torch.long, device=device)
    base_weight = model.lm_head.weight.detach().to(device)
    base_candidate_weight = base_weight.index_select(0, candidate_ids)

    configs = [
        QuantConfig(bits=b, mode=m)
        for b in [8, 6, 5, 4, 3]
        for m in ["vector_symmetric", "vector_asymmetric", "group32_symmetric", "group32_asymmetric"]
    ]

    rows = []
    fp_weight_norm = torch.nn.functional.normalize(base_candidate_weight, dim=-1)
    fp_acc = _evaluate_candidate_weight_top1(
        model=model,
        tokenizer=tokenizer,
        candidate_ids=candidate_ids,
        candidate_weight_normalized=fp_weight_norm,
        split=split,
        max_samples=max_samples,
        max_target_tokens=max_target_tokens,
        device=device,
    )
    rows.append(("none", "none", 0, fp_acc))
    print(f"\nQuantization sweep (latin+punct+byte candidate set): baseline={fp_acc:.2f}%")

    for cfg in configs:
        quant_w = _quantize_weights(base_candidate_weight, cfg, group_size=group_size)
        quant_w_norm = torch.nn.functional.normalize(quant_w, dim=-1)
        acc = _evaluate_candidate_weight_top1(
            model=model,
            tokenizer=tokenizer,
            candidate_ids=candidate_ids,
            candidate_weight_normalized=quant_w_norm,
            split=split,
            max_samples=max_samples,
            max_target_tokens=max_target_tokens,
            device=device,
        )
        rows.append((cfg.mode, "group32" if cfg.mode.startswith("group32") else "vector", cfg.bits, acc))
        print(f"- {cfg.mode} @ {cfg.bits}-bit: {acc:.2f}%")

    csv_path = os.path.join(report_dir, "quantization_sweep.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "granularity", "bits", "top1_percent"])
        writer.writerows(rows)

    fig_path = os.path.join(report_dir, "quantization_sweep_accuracy.png")
    labels = [f"{m}-{b}" if b > 0 else "baseline" for m, _, b, _ in rows]
    vals = [acc for _, _, _, acc in rows]
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=70, ha="right")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Quantization sweep on latin+punct+byte candidate set")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"- wrote quantization CSV: {csv_path}")
    print(f"- wrote quantization graph: {fig_path}")


def _run_two_pass_trim_sweep(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, base_groups: Dict[str, Sequence[int]], split: str,
    max_samples: int, max_target_tokens: int, byte_fallback: bool, device: torch.device, report_dir: str,
    sweep_max_percent: int, sweep_step_percent: int, latin_trim_strategy: str, first_pass_configs: Sequence[QuantConfig],
    first_pass_group_size: int, second_pass_topn_values: Sequence[int], second_pass_dtype: str,
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    percents = list(range(0, sweep_max_percent + 1, sweep_step_percent))
    rows = []
    base_weight = model.lm_head.weight.detach().to(device)
    second_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(second_pass_dtype, torch.float32)
    untrimmed_candidate = sorted(
        set(base_groups["latin"]) | set(base_groups["punct"]) | (set(base_groups["byte"]) if byte_fallback else set())
    )
    for pct in percents:
        latin_trimmed = _trim_latin_token_ids(tokenizer, base_groups["latin"], float(pct), trim_strategy=latin_trim_strategy)
        trimmed_candidate = sorted(set(latin_trimmed) | set(base_groups["punct"]) | (set(base_groups["byte"]) if byte_fallback else set()))
        for variant_name, candidate in [("trimmed", trimmed_candidate), ("untrimmed", untrimmed_candidate)]:
            candidate_ids = torch.tensor(candidate, dtype=torch.long, device=device)
            fp_weight = base_weight.index_select(0, candidate_ids)
            second = fp_weight.to(second_dtype)
            for cfg in first_pass_configs:
                coarse = _quantize_weights(fp_weight, cfg, group_size=first_pass_group_size)
                coarse_norm = torch.nn.functional.normalize(coarse, dim=-1)
                for topn in second_pass_topn_values:
                    acc = _evaluate_two_pass_candidate_weight_top1(
                        model=model, tokenizer=tokenizer, candidate_ids=candidate_ids,
                        first_pass_weight_normalized=coarse_norm, second_pass_weight=second, second_pass_topn=topn,
                        split=split, max_samples=max_samples, max_target_tokens=max_target_tokens, device=device,
                    )
                    rows.append((pct, variant_name, cfg.mode, cfg.bits, topn, len(candidate), acc))
                    print(
                        f"- two-pass {variant_name} trim={pct}% first={cfg.mode}/{cfg.bits}b "
                        f"topn={topn} candidates={len(candidate)} top1={acc:.2f}%"
                    )
    csv_path = os.path.join(report_dir, "two_pass_trim_sweep.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "latin_trim_percent",
                "candidate_variant",
                "first_pass_mode",
                "first_pass_bits",
                "second_pass_topn",
                "candidate_vocab_count",
                "top1_two_pass_percent",
            ]
        )
        w.writerows(rows)


def _train_route_scales(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prototypes: Dict[str, torch.Tensor],
    route_groups: Dict[str, Sequence[int]],
    split: str,
    max_samples: int,
    max_target_tokens: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> torch.Tensor:
    route_names = list(route_groups.keys())
    if route_names != ["latin", "punct", "other"]:
        raise ValueError("Scalar training currently supports only --route_mode three_way.")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    token_to_group = torch.full((tokenizer.vocab_size,), fill_value=2, dtype=torch.long, device=device)
    token_to_group[torch.tensor(route_groups["latin"], dtype=torch.long, device=device)] = 0
    token_to_group[torch.tensor(route_groups["punct"], dtype=torch.long, device=device)] = 1
    token_to_group[torch.tensor(route_groups["other"], dtype=torch.long, device=device)] = 2

    scales = torch.nn.Parameter(torch.ones(len(route_names), device=device))
    optimizer = torch.optim.Adam([scales], lr=lr)
    ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)

    print("\nTraining route scales with frozen model weights")
    print(f"- init scales: {[round(x, 4) for x in scales.detach().cpu().tolist()]}")
    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = 0
        for ex in ds.select(range(min(max_samples, len(ds)))):
            en = ex["translation"]["en"]
            es = ex["translation"]["es"]
            prompt = _build_3shot_prompt(en)

            prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
            target_ids = tokenizer(es, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            if target_ids.numel() == 0:
                continue
            steps = min(max_target_tokens, target_ids.size(1))
            running = prompt_ids

            for idx in range(steps):
                with torch.no_grad():
                    out = model(running, output_hidden_states=True, use_cache=False)
                hidden_last = out.hidden_states[-1][:, -1, :]
                route_logits = _router_scores(hidden_last, prototypes, route_names, scales)
                gold_id = target_ids[0, idx]
                target_group = token_to_group[gold_id].unsqueeze(0)
                loss = F.cross_entropy(route_logits, target_group)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1
                running = torch.cat([running, target_ids[:, idx : idx + 1]], dim=1)

        avg_loss = total_loss / max(1, total_steps)
        print(
            f"- epoch {epoch + 1}/{epochs}: avg_route_ce={avg_loss:.4f}, "
            f"scales={[round(x, 4) for x in scales.detach().cpu().tolist()]}"
        )
    return scales.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Gemma 270M token router eval (OPUS-100 en-es).")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--split", type=str, default="train[:1%]")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_target_tokens", type=int, default=64)
    parser.add_argument(
        "--route_mode",
        type=str,
        default="three_way",
        choices=["three_way", "latin_punct_vs_other", "latin_vs_punct_only", "latin_punct_only"],
        help=(
            "Routing mode: 3-way (latin/punct/other), 2-way ((latin+punct)/other), "
            "latin vs punct only (never route to other), or latin+punct only (no routing)."
        ),
    )
    parser.add_argument(
        "--byte_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include byte tokens in non-other candidate sets as fallback.",
    )
    parser.add_argument("--example_split", type=str, default="validation[:20]")
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--example_max_new_tokens", type=int, default=64)
    parser.add_argument("--chat_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_route_scales", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_split", type=str, default="train[:1%]")
    parser.add_argument("--train_max_samples", type=int, default=100)
    parser.add_argument("--train_max_target_tokens", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--train_lr", type=float, default=1e-2)
    parser.add_argument("--latin_trim_percent", type=float, default=0.0)
    parser.add_argument(
        "--latin_trim_strategy",
        type=str,
        default="longest_bytes",
        choices=["longest_bytes", "highest_id"],
        help="Trim latin tokens by descending UTF-8 byte length or descending token id.",
    )
    parser.add_argument("--latin_trim_sweep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latin_trim_sweep_max", type=int, default=80)
    parser.add_argument("--latin_trim_sweep_step", type=int, default=10)
    parser.add_argument("--sweep_examples", type=int, default=2)
    parser.add_argument("--report_dir", type=str, default="latin_trim_reports")
    parser.add_argument("--quantization_sweep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--two_pass_trim_sweep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--two_pass_first_bits", type=int, default=4)
    parser.add_argument("--two_pass_first_mode", type=str, default="group32_asymmetric")
    parser.add_argument(
        "--two_pass_first_configs",
        type=str,
        default=(
            "group32_asymmetric:4,group32_symmetric:4,vector_asymmetric:4,vector_symmetric:4,"
            "group32_asymmetric:3,group32_symmetric:3,vector_asymmetric:3,vector_symmetric:3"
        ),
        help="Comma-separated first-pass configs in mode:bits form.",
    )
    parser.add_argument("--two_pass_first_group_size", type=int, default=32)
    parser.add_argument("--two_pass_second_topn", type=int, default=1000)
    parser.add_argument("--two_pass_second_topn_values", type=str, default="1,10,100,1000,10000")
    parser.add_argument("--two_pass_second_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--two_pass_report_dir", type=str, default="two_pass_trim_reports")
    parser.add_argument("--quant_group_size", type=int, default=32)
    parser.add_argument("--quant_report_dir", type=str, default="quantization_reports")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.to(device)
    model.eval()

    vocab_size = tokenizer.vocab_size
    base_groups = _build_token_groups(tokenizer, vocab_size)
    _print_group_table(base_groups, vocab_size)

    route_groups = _build_route_groups(
        base_groups=base_groups,
        route_mode=args.route_mode,
        tokenizer=tokenizer,
        latin_trim_percent=args.latin_trim_percent,
        latin_trim_strategy=args.latin_trim_strategy,
    )
    prototypes = _compute_group_prototypes(model.lm_head.weight.detach(), route_groups, device)
    learned_scales = None
    if args.train_route_scales:
        learned_scales = _train_route_scales(
            model=model,
            tokenizer=tokenizer,
            prototypes=prototypes,
            route_groups=route_groups,
            split=args.train_split,
            max_samples=args.train_max_samples,
            max_target_tokens=args.train_max_target_tokens,
            epochs=args.train_epochs,
            lr=args.train_lr,
            device=device,
        )
    stats = _evaluate_router(
        model=model,
        tokenizer=tokenizer,
        route_groups=route_groups,
        byte_ids=base_groups["byte"],
        prototypes=prototypes,
        split=args.split,
        max_samples=args.max_samples,
        max_target_tokens=args.max_target_tokens,
        device=device,
        byte_fallback=args.byte_fallback,
        route_scales=learned_scales,
    )

    acc_routed = 100.0 * stats.top1_correct_routed / max(1, stats.total_target_tokens)
    acc_full = 100.0 * stats.top1_correct_full / max(1, stats.total_target_tokens)
    print("\nRouter evaluation (teacher-forced next-token prediction)")
    print(f"- Routing mode: {args.route_mode}")
    print(f"- Byte fallback: {args.byte_fallback}")
    print(
        "- Route scales: {}".format(
            [round(x, 4) for x in (learned_scales.detach().cpu().tolist() if learned_scales is not None else [1.0] * len(route_groups))]
        )
    )
    print(f"- Evaluated tokens: {stats.total_target_tokens}")
    print(f"- Top-1 accuracy without routing (full LM head): {acc_full:.2f}%")
    print(f"- Top-1 accuracy with routing: {acc_routed:.2f}%")
    route_total = max(1, sum(stats.route_counts.values()))
    route_summary = " ".join(
        f"{name}={100.0 * count / route_total:.2f}%" for name, count in stats.route_counts.items()
    )
    print(f"- Route distribution: {route_summary}")

    _print_validation_examples(
        model=model,
        tokenizer=tokenizer,
        route_groups=route_groups,
        prototypes=prototypes,
        byte_ids=base_groups["byte"],
        example_split=args.example_split,
        num_examples=args.num_examples,
        max_new_tokens=args.example_max_new_tokens,
        byte_fallback=args.byte_fallback,
        device=device,
        route_scales=learned_scales,
    )
    if args.latin_trim_sweep:
        _run_latin_trim_sweep(
            model=model,
            tokenizer=tokenizer,
            base_groups=base_groups,
            route_mode=args.route_mode,
            byte_ids=base_groups["byte"],
            split=args.split,
            max_samples=args.max_samples,
            max_target_tokens=args.max_target_tokens,
            example_split=args.example_split,
            sweep_examples=args.sweep_examples,
            example_max_new_tokens=args.example_max_new_tokens,
            byte_fallback=args.byte_fallback,
            device=device,
            report_dir=args.report_dir,
            sweep_max_percent=args.latin_trim_sweep_max,
            sweep_step_percent=args.latin_trim_sweep_step,
            latin_trim_strategy=args.latin_trim_strategy,
        )
    if args.quantization_sweep:
        _run_quantization_sweep(
            model=model,
            tokenizer=tokenizer,
            base_groups=base_groups,
            split=args.split,
            max_samples=args.max_samples,
            max_target_tokens=args.max_target_tokens,
            group_size=args.quant_group_size,
            report_dir=args.quant_report_dir,
            device=device,
        )
    if args.two_pass_trim_sweep:
        second_topn_values = [int(x.strip()) for x in args.two_pass_second_topn_values.split(",") if x.strip()]
        if not second_topn_values:
            second_topn_values = [args.two_pass_second_topn]
        first_pass_configs = []
        for item in args.two_pass_first_configs.split(","):
            item = item.strip()
            if not item:
                continue
            mode, bits = item.split(":")
            first_pass_configs.append(QuantConfig(bits=int(bits), mode=mode.strip()))
        if not first_pass_configs:
            first_pass_configs = [QuantConfig(bits=args.two_pass_first_bits, mode=args.two_pass_first_mode)]
        _run_two_pass_trim_sweep(
            model=model,
            tokenizer=tokenizer,
            base_groups=base_groups,
            split=args.split,
            max_samples=args.max_samples,
            max_target_tokens=args.max_target_tokens,
            byte_fallback=args.byte_fallback,
            device=device,
            report_dir=args.two_pass_report_dir,
            sweep_max_percent=args.latin_trim_sweep_max,
            sweep_step_percent=args.latin_trim_sweep_step,
            latin_trim_strategy=args.latin_trim_strategy,
            first_pass_configs=first_pass_configs,
            first_pass_group_size=args.two_pass_first_group_size,
            second_pass_topn_values=second_topn_values,
            second_pass_dtype=args.two_pass_second_dtype,
        )
    if args.chat_mode:
        _run_chat_mode(
            model=model,
            tokenizer=tokenizer,
            route_groups=route_groups,
            prototypes=prototypes,
            byte_ids=base_groups["byte"],
            example_max_new_tokens=args.example_max_new_tokens,
            byte_fallback=args.byte_fallback,
            device=device,
            route_scales=learned_scales,
        )


if __name__ == "__main__":
    main()

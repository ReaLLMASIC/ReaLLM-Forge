#!/usr/bin/env python3
"""CJK tri-parallel translation fine-tuning and evaluation pipeline.

This module intentionally wraps the repository's existing prompt seq2seq
training path instead of replacing it.  The JSONL task files are the canonical
human-readable surface; the optional binary datasets under data/ are generated
only to feed train_seq2seq.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
LANG_MAP = {"zho_Hans": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko", "zh": "zh", "ja": "ja", "ko": "ko"}
ORIGINAL_LANG_CODES = {"zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang"}
LANG_LABELS = {"zh": "C", "ja": "J", "ko": "K"}
LANG_NAMES = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean"}
DIRECTIONS = ("zh_to_ja", "ja_to_zh", "zh_to_ko", "ko_to_zh", "ja_to_ko", "ko_to_ja")
FAMILIES = ("tiktoken", "byte", "ipa")
SCORE_COLUMNS = [
    "run_id", "model_variant", "tokenizer_family", "text_representation", "checkpoint_path",
    "eval_dataset", "split_or_benchmark", "direction", "src_lang", "tgt_lang", "metric_scope",
    "f1", "exact_match", "bleu", "chrf", "num_examples", "selected_by_dev",
    "is_best_within_family", "prediction_file", "score_file", "status", "notes",
]
PROMPT_TEMPLATE = (
    "Translate the following sentence from {src_lang_name} to {tgt_lang_name}.\n"
    "{src_label}: {src_text}\n"
    "{tgt_label}:"
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def stable_hash(payload: Any, n: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:n]


def triad_id(translations: dict[str, str]) -> str:
    return "triad-" + stable_hash({lang: norm_text(translations[lang]) for lang in ("zh", "ja", "ko")}, 24)


def split_for_record(record_id: str, train: float = 0.90, dev: float = 0.05) -> str:
    value = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    if value < train:
        return "train"
    if value < train + dev:
        return "dev"
    return "test"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validate_jsonl(tmp)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mark_success(path: Path) -> None:
    atomic_write_text(path, now() + "\n")


def write_run_state(data_dir: Path, step: str, status: str = "completed", extra: dict[str, Any] | None = None) -> None:
    state_path = data_dir / "run_state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.setdefault("steps", {})
    state["steps"][step] = {"status": status, "updated_at": now(), **(extra or {})}
    state["updated_at"] = now()
    atomic_write_json(state_path, state)


def validate_jsonl(path: Path) -> int:
    count = 0
    seen_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if "id" in row:
                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate id {row['id']} in {path}:{line_no}")
                seen_ids.add(row["id"])
            count += 1
    return count


def load_tri_records(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = json.loads(input_path.read_text(encoding="utf-8"))
    for key in ("source", "languages", "records"):
        if key not in root:
            raise ValueError(f"{input_path} is missing root key {key!r}")
    source = root["source"]
    languages = root["languages"]
    lang_lookup = {LANG_MAP.get(code): code for code in languages if LANG_MAP.get(code)}
    missing = [lang for lang in ("zh", "ja", "ko") if lang not in lang_lookup]
    if missing:
        raise ValueError(f"{input_path} is missing required languages after mapping: {missing}")
    rows = []
    skipped = Counter()
    for idx, record in enumerate(root["records"]):
        translations = record.get("translations")
        if not isinstance(translations, dict):
            skipped["missing_translations"] += 1
            continue
        normalized = {}
        empty = False
        for lang in ("zh", "ja", "ko"):
            raw = translations.get(lang_lookup[lang])
            if not isinstance(raw, str):
                empty = True
                skipped[f"missing_{lang}"] += 1
                break
            if not raw.strip():
                empty = True
                skipped[f"empty_{lang}"] += 1
                break
            normalized[lang] = raw
        if empty:
            continue
        rows.append({
            "record_id": triad_id(normalized),
            "translations_raw": normalized,
            "original_lang_codes": {lang: lang_lookup[lang] for lang in ("zh", "ja", "ko")},
            "provenance": [{"dataset": source, "input_file": str(input_path), "record_index": idx}],
        })
    return rows, {"source": source, "raw_records": len(root["records"]), "complete_records": len(rows), "skipped": dict(skipped)}


def cmd_prepare(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    success = out_dir / "_SUCCESS.prepare"
    if success.exists() and not args.force:
        print(f"prepare already complete: {success}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_state(out_dir, "prepare", "running")
    by_id: dict[str, dict[str, Any]] = {}
    input_stats = []
    duplicate_count = 0
    source_counts = Counter()
    skipped_total = Counter()
    for ds in args.datasets:
        path = Path(ds)
        if not path.exists() and str(path) == "ntrex/cjk_sentence_pairs.json":
            path = Path("data/ntrex/data/cjk_sentence_pairs.json")
        rows, stats = load_tri_records(path)
        input_stats.append({"input_file": str(path), **stats})
        skipped_total.update(stats["skipped"])
        for row in rows:
            rid = row["record_id"]
            source_counts[row["provenance"][0]["dataset"]] += 1
            if rid in by_id:
                duplicate_count += 1
                by_id[rid]["provenance"].extend(row["provenance"])
            else:
                by_id[rid] = row
    split_rows = {"train": [], "dev": [], "test": []}
    for row in sorted(by_id.values(), key=lambda r: r["record_id"]):
        split_rows[split_for_record(row["record_id"])].append(row)
    for split, rows in split_rows.items():
        write_jsonl(out_dir / "canonical" / f"{split}.records.jsonl", rows)
    directed_counts = {split: len(rows) * len(DIRECTIONS) for split, rows in split_rows.items()}
    metadata = {
        "input_file_paths": args.datasets,
        "resolved_input_files": [s["input_file"] for s in input_stats],
        "benchmark_input_path": "benchmarks_easy.csv",
        "source_dataset_counts": dict(source_counts),
        "input_stats": input_stats,
        "complete_tri_record_count": len(by_id),
        "skipped_record_counts_by_reason": dict(skipped_total),
        "duplicate_count": duplicate_count,
        "train_dev_test_tri_record_counts": {split: len(rows) for split, rows in split_rows.items()},
        "expected_directed_task_counts_per_split": directed_counts,
        "counts_per_direction": {direction: {split: len(rows) for split, rows in split_rows.items()} for direction in DIRECTIONS},
        "language_mapping": {"zho_Hans": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko"},
        "language_labels": LANG_LABELS,
        "language_names": LANG_NAMES,
        "prompt_template": PROMPT_TEMPLATE,
        "benchmark_adapter_information": None,
    }
    atomic_write_json(out_dir / "metadata.json", metadata)
    write_run_state(out_dir, "prepare", "completed", {"complete_tri_record_count": len(by_id)})
    mark_success(success)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def split_direction(direction: str) -> tuple[str, str]:
    src, _, tgt = direction.partition("_to_")
    if src not in LANG_NAMES or tgt not in LANG_NAMES:
        raise ValueError(f"Unknown direction: {direction}")
    return src, tgt


def make_prompt(src_lang: str, tgt_lang: str, src_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        src_lang_name=LANG_NAMES[src_lang],
        tgt_lang_name=LANG_NAMES[tgt_lang],
        src_label=LANG_LABELS[src_lang],
        tgt_label=LANG_LABELS[tgt_lang],
        src_text=src_text,
    )


class IpaConverter:
    def __init__(self) -> None:
        self.available = False
        self.error = None
        self._cache: dict[tuple[str, str], str] = {}
        try:
            self.zh_mod = self._load("zh_to_ipa", REPO_ROOT / "data/template/utils/zh_to_ipa.py")
            self.ja_mod = self._load("ja2ipa", REPO_ROOT / "data/template/utils/ja2ipa.py")
            self.ko_mod = self._load("ko_en_to_ipa", REPO_ROOT / "data/template/utils/ko_en_to_ipa.py")
            self.available = True
        except Exception as exc:  # optional dependencies may be absent
            self.error = str(exc)

    @staticmethod
    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def convert(self, lang: str, text: str) -> str:
        key = (lang, text)
        if key in self._cache:
            return self._cache[key]
        if not self.available:
            raise RuntimeError(f"IPA conversion unavailable: {self.error}")
        if lang == "zh":
            # The repository's Chinese IPA utility can fail on valid CJK text
            # when dragonmapper emits tone-bearing syllables it cannot parse.
            # Preserve those spans literally so the custom tokenizer's byte
            # fallback handles them instead of emitting an error string.
            pieces = []
            for ch in text:
                out = self.zh_mod.transcribe_chinese(ch)
                pieces.append(ch if out.startswith("Error in transcribing Chinese:") else out)
            converted = "".join(pieces)
            self._cache[key] = converted
            return converted
        if lang == "ja":
            converted = self.ja_mod.hiragana_to_ipa(self.ja_mod.to_hiragana(text))
            self._cache[key] = converted
            return converted
        if lang == "ko":
            stats = {}
            converted = self.ko_mod.transcribe_plain_text(text, wrapper=False, stats=stats)
            self._cache[key] = converted
            return converted
        raise ValueError(lang)


def iter_directed(records: list[dict[str, Any]], split: str, family: str, ipa: IpaConverter | None = None) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        for direction in DIRECTIONS:
            src, tgt = split_direction(direction)
            src_text = rec["translations_raw"][src]
            tgt_text = rec["translations_raw"][tgt]
            if family == "ipa":
                assert ipa is not None
                src_text = ipa.convert(src, src_text)
                tgt_text = ipa.convert(tgt, tgt_text)
            out.append({
                "id": f"{rec['record_id']}:{direction}",
                "direction": direction,
                "prompt": make_prompt(src, tgt, src_text),
                "completion": tgt_text,
            })
    return out


def cmd_render(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    family = args.tokenizer_family
    success = data_dir / "tasks" / family / "_SUCCESS.render"
    if success.exists() and not args.force:
        print(f"render already complete: {success}")
        return
    write_run_state(data_dir, f"render:{family}", "running")
    ipa = IpaConverter() if family == "ipa" else None
    if family == "ipa" and not ipa.available:
        write_run_state(data_dir, f"render:{family}", "skipped", {"error": ipa.error})
        raise RuntimeError(f"IPA rendering unavailable: {ipa.error}")
    counts = {}
    for split in ("train", "dev", "test"):
        records = read_jsonl(data_dir / "canonical" / f"{split}.records.jsonl")
        rows = iter_directed(records, split, family, ipa)
        write_jsonl(data_dir / "tasks" / family / f"{split}.jsonl", rows)
        validate_rendered_jsonl(data_dir / "tasks" / family / f"{split}.jsonl")
        counts[split] = len(rows)
    write_run_state(data_dir, f"render:{family}", "completed", {"counts": counts})
    mark_success(success)
    print(json.dumps({"tokenizer_family": family, "counts": counts}, ensure_ascii=False, indent=2))


def validate_rendered_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if set(row) != {"id", "direction", "prompt", "completion"}:
                raise ValueError(f"{path}:{line_no} must contain only id,direction,prompt,completion")
            if row["direction"] not in DIRECTIONS:
                raise ValueError(f"{path}:{line_no} bad direction {row['direction']}")
            if not row["completion"]:
                raise ValueError(f"{path}:{line_no} empty completion")
            if row["prompt"].rstrip().endswith(row["completion"]):
                raise ValueError(f"{path}:{line_no} target appears at prompt end")
            count += 1
    return count


def inspect_benchmark(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    col_map = {
        "zh": "Chinese (Simplified)" if "Chinese (Simplified)" in fields else None,
        "ja": "Japanese (Natural Polite)" if "Japanese (Natural Polite)" in fields else None,
        "ko": "Korean (Natural Polite)" if "Korean (Natural Polite)" in fields else None,
    }
    compatible = all(col_map.values())
    return {
        "benchmark": str(path),
        "num_rows": len(rows),
        "columns": fields,
        "source_language_columns": col_map,
        "target_language_columns": col_map,
        "source_text_or_prompt_columns": col_map,
        "reference_target_text_columns": col_map,
        "direction": "implicit_from_all_cjk_language_columns",
        "format": "raw_tri_parallel_rows" if compatible else "unknown",
        "text_representation": "orthographic",
        "dataset_task_identifier_columns": [c for c in ("#", "Focus / Type") if c in fields],
        "compatible_with_cjk_translation_task": compatible,
        "adapter_required": True,
        "error": None if compatible else "Cannot identify Chinese, Japanese, and Korean reference columns.",
    }


def cmd_inspect_benchmark(args: argparse.Namespace) -> None:
    schema = inspect_benchmark(Path(args.benchmark))
    atomic_write_json(Path(args.out), schema)
    print(json.dumps(schema, ensure_ascii=False, indent=2))


def cmd_adapt_benchmark(args: argparse.Namespace) -> None:
    schema = inspect_benchmark(Path(args.benchmark))
    if not schema["compatible_with_cjk_translation_task"]:
        raise ValueError(schema["error"])
    out = Path(args.out)
    rows_out = []
    with Path(args.benchmark).open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            tri = {
                "zh": row["Chinese (Simplified)"],
                "ja": row["Japanese (Natural Polite)"],
                "ko": row["Korean (Natural Polite)"],
            }
            for direction in DIRECTIONS:
                src, tgt = split_direction(direction)
                if not tri[src].strip() or not tri[tgt].strip():
                    continue
                ex_id = f"benchmarks_easy-row-{row.get('#') or idx}:{direction}"
                rows_out.append({
                    "id": ex_id,
                    "benchmark_row_id": row.get("#") or str(idx),
                    "record_id": f"benchmarks_easy-row-{row.get('#') or idx}",
                    "direction": direction,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "prompt": make_prompt(src, tgt, tri[src]),
                    "completion": tri[tgt],
                    "dataset": "benchmarks_easy",
                    "task": row.get("Focus / Type", ""),
                })
    write_jsonl(out, rows_out)
    schema["adapted_output"] = str(out)
    schema["adapted_num_examples"] = len(rows_out)
    atomic_write_json(out.parent / "benchmark_schema.json", schema)
    print(json.dumps({"out": str(out), "num_examples": len(rows_out)}, ensure_ascii=False, indent=2))


def source_text_from_prompt(example: dict[str, Any]) -> str:
    if "source_text" in example:
        return example["source_text"]
    src = example.get("src_lang")
    if not src and example.get("direction") in DIRECTIONS:
        src, _ = split_direction(example["direction"])
    if not src:
        raise ValueError(f"Cannot infer source language for benchmark example {example.get('id')}")
    prefix = f"{LANG_LABELS[src]}: "
    for line in example.get("prompt", "").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    raise ValueError(f"Cannot extract source text from benchmark prompt for {example.get('id')}")


def render_benchmark_examples_for_family(
    examples: list[dict[str, Any]],
    family: str,
    ipa_eval_mode: str = "native",
) -> list[dict[str, Any]]:
    if family != "ipa":
        return examples
    ipa = IpaConverter()
    if not ipa.available:
        raise RuntimeError(f"IPA rendering unavailable: {ipa.error}")
    rendered = []
    for ex in examples:
        direction = ex["direction"]
        src, tgt = split_direction(direction)
        src_text = ipa.convert(src, source_text_from_prompt(ex))
        if ipa_eval_mode == "source_only":
            tgt_text = ex["completion"]
            text_representation = "ipa_source_orthographic_target"
        else:
            tgt_text = ipa.convert(tgt, ex["completion"])
            text_representation = "ipa"
        rendered.append({
            **ex,
            "prompt": make_prompt(src, tgt, src_text),
            "completion": tgt_text,
            "orthographic_prompt": ex["prompt"],
            "orthographic_completion": ex["completion"],
            "text_representation": text_representation,
        })
    return rendered


def render_split_examples_for_eval(data_dir: Path, split: str, family: str, ipa_eval_mode: str) -> list[dict[str, Any]]:
    if family != "ipa" or ipa_eval_mode != "source_only":
        return read_jsonl(data_dir / "tasks" / family / f"{split}.jsonl")
    ipa = IpaConverter()
    if not ipa.available:
        raise RuntimeError(f"IPA rendering unavailable: {ipa.error}")
    examples = []
    for rec in read_jsonl(data_dir / "canonical" / f"{split}.records.jsonl"):
        for direction in DIRECTIONS:
            src, tgt = split_direction(direction)
            src_text = ipa.convert(src, rec["translations_raw"][src])
            tgt_text = rec["translations_raw"][tgt]
            examples.append({
                "id": f"{rec['record_id']}:{direction}",
                "direction": direction,
                "prompt": make_prompt(src, tgt, src_text),
                "completion": tgt_text,
                "text_representation": "ipa_source_orthographic_target",
            })
    return examples


def discover_variants(search_roots: list[str]) -> list[dict[str, Any]]:
    variants = []
    for root in search_roots:
        for meta_path in Path(root).glob("**/meta.pkl"):
            ckpt = meta_path.parent / "ckpt.pt"
            if not ckpt.exists():
                continue
            try:
                meta = pickle.loads(meta_path.read_bytes())
            except Exception as exc:
                variants.append({"model_variant": meta_path.parent.name, "status": "skipped", "reason": f"bad meta.pkl: {exc}"})
                continue
            tokenizer = meta.get("tokenizer")
            if tokenizer == "tiktoken":
                family, rep = "tiktoken", "orthographic"
            elif tokenizer == "byte":
                family, rep = "byte", "orthographic"
            elif tokenizer == "custom_char_with_byte_fallback":
                chars = "".join(str(c) for c in meta.get("custom_chars", []))
                if any(ch in chars for ch in ("ɑ", "ɕ", "ɯ", "ː")):
                    family, rep = "ipa", "ipa"
                else:
                    variants.append({"model_variant": meta_path.parent.name, "status": "skipped", "reason": "custom byte-fallback tokenizer is not identifiable as IPA"})
                    continue
            else:
                variants.append({"model_variant": meta_path.parent.name, "status": "skipped", "reason": f"unsupported tokenizer {tokenizer!r}"})
                continue
            variants.append({
                "model_variant": meta_path.parent.name,
                "run_id": stable_hash({"variant": meta_path.parent.name, "ckpt": str(ckpt)}, 12),
                "checkpoint": str(ckpt),
                "base_meta": str(meta_path),
                "tokenizer_family": family,
                "text_representation": rep,
                "status": "available",
            })
    return sorted(variants, key=lambda v: (v.get("tokenizer_family", ""), v["model_variant"]))


def cmd_discover(args: argparse.Namespace) -> None:
    variants = discover_variants(args.search_roots)
    if args.out:
        atomic_write_json(Path(args.out), variants)
    print(json.dumps(variants, ensure_ascii=False, indent=2))


def load_variant(name: str, search_roots: list[str]) -> dict[str, Any]:
    selected_path = REPO_ROOT / "cjk_translation" / "selected_base_checkpoints.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_matches = [
            row for row in selected
            if name in {row.get("variation_key"), row.get("scoped_variation_key"), row.get("run_name")}
        ]
        if selected_matches:
            if len(selected_matches) > 1:
                raise ValueError(f"Multiple selected variants match {name!r}")
            row = dict(selected_matches[0])
            model_variant = row.get("variation_key") or row.get("run_name")
            row["model_variant"] = model_variant
            row["run_id"] = stable_hash({"variant": model_variant, "ckpt": row["checkpoint"]}, 12)
            row["status"] = "available"
            return row
    matches = [v for v in discover_variants(search_roots) if v["model_variant"] == name and v.get("status") == "available"]
    if not matches:
        raise ValueError(f"No valid model variant named {name!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple variants named {name!r}; use unique directory names or reduce --search-roots")
    return matches[0]


def checkpoint_block_size(checkpoint_path: Path) -> int:
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        block_size = int(checkpoint["model_args"]["block_size"])
    except Exception as exc:
        raise ValueError(f"{checkpoint_path} has no model_args['block_size']") from exc
    return block_size


def build_seq2seq_meta(base_meta_path: Path, family: str) -> tuple[dict[str, Any], Callable[[str], list[int]]]:
    import tiktoken
    with base_meta_path.open("rb") as f:
        base_meta = pickle.load(f)
    special_tokens = []
    if family == "tiktoken":
        base_enc = tiktoken.get_encoding(base_meta["tiktoken_encoding"])
        special = {tok: base_enc.n_vocab + i for i, tok in enumerate(special_tokens)}
        meta = {"vocab_size": base_enc.n_vocab, "tokenizer": "tiktoken", "tiktoken_encoding": base_meta["tiktoken_encoding"], "special_tokens": special}
        return meta, lambda text: base_enc.encode(text, allowed_special=set(), disallowed_special=())
    if family == "byte":
        return {"vocab_size": 256, "tokenizer": "byte"}, lambda text: list(text.encode("utf-8"))
    # custom_char_with_byte_fallback
    custom = list(base_meta.get("custom_chars", []))
    stoi = {bytes([b]): b for b in range(256)}
    itos = {b: bytes([b]) for b in range(256)}
    for idx, token in enumerate(custom, start=256):
        stoi[token] = idx
        itos[idx] = token
    ordered = sorted(custom, key=lambda t: len(t.encode("utf-8")), reverse=True)
    token_bytes = {token: token.encode("utf-8") for token in custom}
    def encode(text: str) -> list[int]:
        data = text.encode("utf-8")
        ids: list[int] = []
        i = 0
        while i < len(data):
            for token in ordered:
                tb = token_bytes[token]
                if data[i:i + len(tb)] == tb:
                    ids.append(stoi[token])
                    i += len(tb)
                    break
            else:
                ids.append(data[i])
                i += 1
        return ids
    return {"vocab_size": 256 + len(custom), "tokenizer": "custom_char_with_byte_fallback", "custom_chars": custom, "stoi": stoi, "itos": itos}, encode


def validate_task_jsonl(data_dir: Path, family: str) -> dict[str, Any]:
    summary = {}
    for split in ("train", "dev", "test"):
        path = data_dir / "tasks" / family / f"{split}.jsonl"
        rows = read_jsonl(path)
        keys = {"id", "direction", "prompt", "completion"}
        bad = [idx for idx, row in enumerate(rows, start=1) if set(row) != keys]
        if bad:
            raise ValueError(f"{path} has rows with unexpected keys; first bad row: {bad[0]}")
        directions = Counter(row["direction"] for row in rows)
        missing = [direction for direction in DIRECTIONS if directions[direction] == 0]
        if missing:
            raise ValueError(f"{path} is missing directions: {missing}")
        summary[split] = {"rows": len(rows), "directions": dict(directions)}
    return summary


def write_seq2seq_dataset(
    data_dir: Path,
    family: str,
    base_meta: Path,
    dataset_name: str,
    force: bool = False,
    checkpoint_path: Path | None = None,
) -> Path:
    """Build a GPT SFT prompt/completion dataset with shifted loss masks."""
    dataset_dir = REPO_ROOT / "data" / dataset_name
    success = dataset_dir / "_SUCCESS.cjk_seq2seq"
    if success.exists() and not force:
        return dataset_dir
    meta, encode = build_seq2seq_meta(base_meta, family)
    task_summary = validate_task_jsonl(data_dir, family)
    block_size = checkpoint_block_size(checkpoint_path) if checkpoint_path else None
    dataset_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    masks = {}
    pairs = {}
    dtypes = {}
    skip_summary = {}
    for split, bin_name in (("train", "train"), ("dev", "val"), ("test", "test")):
        token_ids: list[int] = []
        loss_mask: list[int] = []
        pair_rows = []
        split_skips = {
            "total_input_examples": 0,
            "kept_examples": 0,
            "skipped_examples": 0,
            "prompt_plus_target_over_block": 0,
            "prompt_over_block": 0,
            "empty_prompt_or_target": 0,
            "kept_counts_by_direction": Counter(),
            "skipped_counts_by_direction": Counter(),
            "suspicious_direction_loss": False,
        }
        for row in read_jsonl(data_dir / "tasks" / family / f"{split}.jsonl"):
            prompt_ids = encode(row["prompt"])
            target_ids = encode(row["completion"] + "\n")
            split_skips["total_input_examples"] += 1
            skip_reason = None
            if not prompt_ids or not target_ids:
                skip_reason = "empty_prompt_or_target"
            elif block_size is not None and len(prompt_ids) >= block_size:
                skip_reason = "prompt_over_block"
            elif block_size is not None and len(prompt_ids) + len(target_ids) > block_size:
                skip_reason = "prompt_plus_target_over_block"
            if skip_reason is not None:
                split_skips[skip_reason] += 1
                split_skips["skipped_examples"] += 1
                split_skips["skipped_counts_by_direction"][row["direction"]] += 1
                continue
            token_ids.extend(prompt_ids)
            loss_mask.extend([0] * len(prompt_ids))
            token_ids.extend(target_ids)
            loss_mask.extend([1] * len(target_ids))
            split_skips["kept_examples"] += 1
            split_skips["kept_counts_by_direction"][row["direction"]] += 1
            pair_rows.append({"id": row["id"], "direction": row["direction"], "prompt": row["prompt"], "completion": row["completion"]})
        dtype = np.uint32 if meta["vocab_size"] > 65535 else np.uint16
        np.asarray(token_ids, dtype=dtype).tofile(dataset_dir / f"{bin_name}.bin")
        np.asarray(loss_mask, dtype=np.uint8).tofile(dataset_dir / f"{bin_name}_loss_mask.bin")
        if len(token_ids) != len(loss_mask):
            raise ValueError(f"{bin_name}.bin and {bin_name}_loss_mask.bin would have different lengths")
        if split == "dev":
            shutil.copy2(dataset_dir / "val_loss_mask.bin", dataset_dir / "dev_loss_mask.bin")
        write_jsonl(dataset_dir / f"{split}_pairs.jsonl", pair_rows)
        if split == "dev":
            shutil.copy2(dataset_dir / "dev_pairs.jsonl", dataset_dir / "val_pairs.jsonl")
        arrays[split] = len(token_ids)
        masks[split] = len(loss_mask)
        pairs[split] = len(pair_rows)
        dtypes[bin_name] = np.dtype(dtype).name
        for direction in DIRECTIONS:
            total_dir = split_skips["kept_counts_by_direction"][direction] + split_skips["skipped_counts_by_direction"][direction]
            if total_dir and split_skips["skipped_counts_by_direction"][direction] / total_dir > 0.20:
                split_skips["suspicious_direction_loss"] = True
        split_skips["kept_counts_by_direction"] = dict(split_skips["kept_counts_by_direction"])
        split_skips["skipped_counts_by_direction"] = dict(split_skips["skipped_counts_by_direction"])
        skip_summary[split] = split_skips
    with (dataset_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    atomic_write_json(dataset_dir / "manifest.json", {
        "dataset_name": dataset_name,
        "source": "cjk_translation_pipeline",
        "tokenizer_family": family,
        "base_meta": str(base_meta),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "block_size": block_size,
        "prompt_target_budget_rule": "len(encode(prompt)) + len(encode(completion + '\\n')) <= block_size",
        "token_counts": arrays,
        "loss_mask_token_counts": masks,
        "pair_counts": pairs,
        "task_jsonl_validation": task_summary,
        "skip_counts": skip_summary,
        "dtype_by_bin": dtypes,
        "loss_masking": "prompt tokens masked, completion tokens unmasked",
    })
    mark_success(success)
    return dataset_dir


def update_status(run_dir: Path, payload: dict[str, Any]) -> None:
    status_path = run_dir / "status.json"
    current = {}
    if status_path.exists():
        current = json.loads(status_path.read_text(encoding="utf-8"))
    current.update(payload)
    current["updated_at"] = now()
    atomic_write_json(status_path, current)


def append_manifest(runs_dir: Path, row: dict[str, Any]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "manifest.jsonl"
    rows = []
    if path.exists():
        rows = read_jsonl(path)
        rows = [r for r in rows if r.get("model_variant") != row.get("model_variant")]
    rows.append(row)
    write_jsonl(path, rows)


def cmd_finetune(args: argparse.Namespace) -> None:
    variant = load_variant(args.model_variant, args.search_roots)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    success = out_dir / "_SUCCESS.finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    arg_payload = {k: v for k, v in vars(args).items() if k != "func"}
    run_config = {**variant, **arg_payload, "run_id": variant["run_id"]}
    atomic_write_json(out_dir / "run_config.json", run_config)
    append_manifest(data_dir / "runs", {**variant, "run_dir": str(out_dir), "status": "registered", "updated_at": now()})
    update_status(out_dir, {
        "model_variant": args.model_variant, "tokenizer_family": variant["tokenizer_family"],
        "text_representation": variant["text_representation"], "status": "running",
        "current_step": "finetune", "completed_steps": [], "failed_steps": [],
        "best_checkpoint": None, "latest_checkpoint": None,
        "dev_score_file": str(out_dir / "dev_scores.json"),
        "test_score_file": str(out_dir / "test_scores.json"),
        "benchmark_score_file": str(out_dir / "benchmark_easy_scores.json"),
        "benchmark_prediction_file": str(out_dir / "benchmark_easy_predictions.jsonl"),
        "error": None,
    })
    if success.exists() and not args.force and not args.dry_run:
        update_status(out_dir, {"status": "completed", "current_step": "finetune", "best_checkpoint": str(out_dir / "ckpt.pt"), "latest_checkpoint": str(out_dir / "ckpt.pt")})
        print(f"finetune already complete: {success}")
        return
    dataset_name = f"cjk_translation_{variant['tokenizer_family']}_{stable_hash(variant['checkpoint'], 8)}"
    dataset_dir = write_seq2seq_dataset(
        data_dir,
        variant["tokenizer_family"],
        Path(variant["base_meta"]),
        dataset_name,
        force=args.force,
        checkpoint_path=Path(variant["checkpoint"]),
    )
    shutil.copy2(dataset_dir / "meta.pkl", out_dir / "meta.pkl")
    cmd = [
        sys.executable, "train_cjk_sft.py",
        "--dataset", dataset_name,
        "--init_checkpoint", variant["checkpoint"],
        "--out_dir", str(out_dir),
        "--sft_loss_mask",
        "--max_iters", str(args.max_iters),
        "--eval_interval", str(args.eval_interval),
        "--eval_iters", str(args.eval_iters),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--optimizer", args.optimizer,
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--cosine_t_max", str(args.cosine_t_max),
        "--cosine_eta_min", str(args.cosine_eta_min),
        "--adamw_weight_decay", str(args.adamw_weight_decay),
        "--adamw_betas", str(args.adamw_betas[0]), str(args.adamw_betas[1]),
        "--grad_clip", str(args.grad_clip),
        "--loss_fn", args.loss_fn,
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    atomic_write_text(out_dir / "logs" / "finetune_command.txt", " ".join(cmd) + "\n")
    if args.dry_run:
        update_status(out_dir, {"status": "skipped", "current_step": "finetune", "error": "dry_run"})
        print(" ".join(cmd))
        return
    with (out_dir / "logs" / "finetune.log").open("a", encoding="utf-8") as log:
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            update_status(out_dir, {"status": "failed", "failed_steps": ["finetune"], "error": str(exc)})
            append_manifest(data_dir / "runs", {**variant, "run_dir": str(out_dir), "status": "failed", "error": str(exc), "updated_at": now()})
            raise
    best = out_dir / "ckpt.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    shutil.copy2(best, out_dir / "checkpoints" / "ckpt.pt")
    mark_success(success)
    update_status(out_dir, {"status": "completed", "current_step": "finetune", "completed_steps": ["finetune"], "best_checkpoint": str(best), "latest_checkpoint": str(best), "error": None})
    append_manifest(data_dir / "runs", {**variant, "run_dir": str(out_dir), "status": "completed", "updated_at": now()})


def char_f1(pred: str, ref: str) -> float:
    pred_chars = list(norm_text(pred))
    ref_chars = list(norm_text(ref))
    if not pred_chars and not ref_chars:
        return 1.0
    if not pred_chars or not ref_chars:
        return 0.0
    pc, rc = Counter(pred_chars), Counter(ref_chars)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def corpus_bleu(preds: list[str], refs: list[str]) -> float:
    try:
        from benchmarks.bleu import corpus_bleu as repo_bleu
        return float(repo_bleu(preds, [refs]))
    except Exception:
        clipped = [0] * 4
        total = [0] * 4
        ref_len = hyp_len = 0
        for ref, hyp in zip(refs, preds):
            rt = list(norm_text(ref))
            ht = list(norm_text(hyp))
            ref_len += len(rt); hyp_len += len(ht)
            for order in range(1, 5):
                h = Counter(tuple(ht[i:i+order]) for i in range(max(len(ht) - order + 1, 0)))
                r = Counter(tuple(rt[i:i+order]) for i in range(max(len(rt) - order + 1, 0)))
                total[order-1] += max(len(ht) - order + 1, 0)
                clipped[order-1] += sum(min(c, r.get(k, 0)) for k, c in h.items())
        if hyp_len == 0:
            return 0.0
        precisions = [(c + 1.0) / (t + 1.0) for c, t in zip(clipped, total)]
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
        return 100.0 * bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)


def corpus_chrf(preds: list[str], refs: list[str]) -> float | None:
    try:
        import sacrebleu
        return float(sacrebleu.corpus_chrf(preds, [refs]).score)
    except Exception:
        return None


def score_predictions(rows: list[dict[str, Any]], eval_dataset: str, split_or_benchmark: str, checkpoint: str) -> dict[str, Any]:
    by_dir = defaultdict(list)
    for row in rows:
        by_dir[row.get("direction", "unknown")].append(row)
    direction_scores = []
    for direction, items in sorted(by_dir.items()):
        preds = [i["prediction"] for i in items]
        refs = [i["reference"] for i in items]
        src, tgt = ("", "")
        if direction in DIRECTIONS:
            src, tgt = split_direction(direction)
        direction_scores.append({
            "eval_dataset": eval_dataset,
            "split_or_benchmark": split_or_benchmark,
            "direction": direction,
            "src_lang": src,
            "tgt_lang": tgt,
            "metric_scope": "direction",
            "f1": sum(char_f1(p, r) for p, r in zip(preds, refs)) / max(len(items), 1),
            "exact_match": sum(norm_text(p) == norm_text(r) for p, r in zip(preds, refs)) / max(len(items), 1),
            "bleu": corpus_bleu(preds, refs),
            "chrf": corpus_chrf(preds, refs),
            "num_examples": len(items),
        })
    macro = {
        "eval_dataset": eval_dataset,
        "split_or_benchmark": split_or_benchmark,
        "direction": "macro_avg",
        "src_lang": "",
        "tgt_lang": "",
        "metric_scope": "macro_avg" if split_or_benchmark != "benchmarks_easy.csv" else "benchmark_overall",
        "num_examples": sum(s["num_examples"] for s in direction_scores),
    }
    for metric in ("f1", "exact_match", "bleu", "chrf"):
        vals = [s[metric] for s in direction_scores if s[metric] is not None]
        macro[metric] = sum(vals) / len(vals) if vals else None
    return {"checkpoint_path": checkpoint, "scores_by_direction": direction_scores, "macro_avg": macro}


def cmd_eval(args: argparse.Namespace) -> None:
    # This command supports metric-only evaluation for existing predictions, and
    # model generation for checkpoints produced by train_cjk_sft.py.
    out = Path(args.out)
    pred_out = Path(args.predictions_out)
    if out.exists() and pred_out.exists() and not args.force:
        print(f"eval already complete: {out}")
        return
    if args.benchmark and not args.mock_predictions and not args.predictions_in and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Benchmark evaluation requires a real checkpoint: {args.checkpoint}")
    if args.benchmark:
        adapted = Path(args.data_dir) / "benchmark_adapted" / "benchmarks_easy.jsonl"
        if not adapted.exists():
            cmd_adapt_benchmark(argparse.Namespace(benchmark=args.benchmark, out=str(adapted)))
        examples = read_jsonl(adapted)
        run_config = json.loads((Path(args.checkpoint).parent / "run_config.json").read_text(encoding="utf-8")) if (Path(args.checkpoint).parent / "run_config.json").exists() else {}
        family = args.tokenizer_family or run_config.get("tokenizer_family")
        if family:
            examples = render_benchmark_examples_for_family(examples, family, args.ipa_eval_mode)
        eval_dataset, split_or_benchmark = "benchmarks_easy", "benchmarks_easy.csv"
    else:
        if not args.split:
            raise ValueError("--split is required unless --benchmark is passed")
        run_config = json.loads((Path(args.checkpoint).parent / "run_config.json").read_text(encoding="utf-8")) if (Path(args.checkpoint).parent / "run_config.json").exists() else {}
        family = args.tokenizer_family or run_config.get("tokenizer_family")
        if not family:
            raise ValueError("Cannot infer tokenizer family; pass --tokenizer-family")
        examples = render_split_examples_for_eval(Path(args.data_dir), args.split, family, args.ipa_eval_mode)
        eval_dataset, split_or_benchmark = args.split, args.split
    if args.predictions_in:
        pred_rows = read_jsonl(Path(args.predictions_in))
    elif args.mock_predictions:
        pred_rows = [{**ex, "prediction": ex["completion"], "reference": ex["completion"]} for ex in examples]
    else:
        pred_rows = generate_predictions(args, examples)
    write_jsonl(pred_out, pred_rows)
    result = score_predictions(pred_rows, eval_dataset, split_or_benchmark, str(args.checkpoint))
    result["prediction_file"] = str(pred_out)
    if family == "ipa":
        result["ipa_eval_mode"] = args.ipa_eval_mode
    atomic_write_json(out, result)
    marker = out.parent / f"_SUCCESS.eval_{'benchmarks_easy' if args.benchmark else args.split}"
    mark_success(marker)
    if (out.parent / "status.json").exists():
        step = f"eval_{'benchmarks_easy' if args.benchmark else args.split}"
        st = json.loads((out.parent / "status.json").read_text(encoding="utf-8"))
        completed = sorted(set(st.get("completed_steps", []) + [step]))
        status_value = "completed" if step == "eval_benchmarks_easy" or st.get("status") == "completed" else "partial"
        update_status(out.parent, {"status": status_value, "current_step": step, "completed_steps": completed, "error": None})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def generate_until_stop(model, start_ids, max_new_tokens: int, decode: Callable[[list[int]], str], temperature: float, top_k: int | None):
    import torch
    from torch.nn import functional as F
    idx = start_ids
    generated_ids: list[int] = []
    generated_text = ""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        next_id = int(idx_next[0, 0].item())
        generated_ids.append(next_id)
        idx = torch.cat((idx, idx_next), dim=1)
        generated_text = decode(generated_ids)
        if generated_text.endswith("\n"):
            break
    return idx, generated_text


def generate_predictions(args: argparse.Namespace, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from contextlib import nullcontext
    from gpt_conf import GPTConfig
    from model import GPT
    from sample import get_tokenizer_functions
    ckpt = Path(args.checkpoint)
    meta_path = ckpt.parent / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.pkl next to checkpoint: {meta_path}")
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    encode, decode = get_tokenizer_functions(meta)
    if meta.get("tokenizer") == "tiktoken" and args.tiktoken_decode_mode == "bytes":
        import tiktoken
        enc = tiktoken.get_encoding(meta["tiktoken_encoding"])

        def decode(token_ids: list[int]) -> str:
            data = b"".join(enc.decode_single_token_bytes(token_id) for token_id in token_ids)
            return data.decode("utf-8", errors="replace")

    checkpoint = torch.load(ckpt, map_location=args.device)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" or args.dtype == "float32" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    rows = []
    torch.manual_seed(1337)
    with torch.no_grad():
        for idx, ex in enumerate(examples):
            prompt_ids = encode(ex["prompt"])
            remaining = model.config.block_size - len(prompt_ids)
            if remaining <= 0:
                rows.append({**ex, "prediction": "", "reference": ex["completion"], "skip_reason": "prompt_over_block"})
                continue
            max_new_tokens = min(args.max_new_tokens, remaining)
            start_ids = torch.tensor(prompt_ids, dtype=torch.long, device=args.device)[None, ...]
            with ctx:
                generated = generate_until_stop(
                    model, start_ids, max_new_tokens=max_new_tokens,
                    decode=decode, temperature=args.temperature, top_k=args.top_k,
                )[1]
            rows.append({**ex, "prediction": generated.split("\n", 1)[0].strip(), "reference": ex["completion"]})
            if args.max_examples is not None and idx + 1 >= args.max_examples:
                break
    return rows


def cmd_aggregate(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    rows = []
    for run_config_path in runs_dir.glob("*/run_config.json"):
        run_dir = run_config_path.parent
        config = json.loads(run_config_path.read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8")) if (run_dir / "status.json").exists() else {"status": "unknown"}
        score_files = [("dev", run_dir / "dev_scores.json"), ("test", run_dir / "test_scores.json"), ("benchmarks_easy", run_dir / "benchmark_easy_scores.json")]
        for eval_name, score_file in score_files:
            if not score_file.exists():
                continue
            score = json.loads(score_file.read_text(encoding="utf-8"))
            pred_file = score.get("prediction_file", "")
            for s in score.get("scores_by_direction", []) + [score.get("macro_avg", {})]:
                if not s:
                    continue
                rows.append({
                    "run_id": config.get("run_id", ""),
                    "model_variant": config.get("model_variant", run_dir.name),
                    "tokenizer_family": config.get("tokenizer_family", ""),
                    "text_representation": config.get("text_representation", ""),
                    "checkpoint_path": score.get("checkpoint_path", config.get("checkpoint", "")),
                    "eval_dataset": s.get("eval_dataset", eval_name),
                    "split_or_benchmark": s.get("split_or_benchmark", eval_name),
                    "direction": s.get("direction", ""),
                    "src_lang": s.get("src_lang", ""),
                    "tgt_lang": s.get("tgt_lang", ""),
                    "metric_scope": s.get("metric_scope", "direction"),
                    "f1": s.get("f1"),
                    "exact_match": s.get("exact_match"),
                    "bleu": s.get("bleu"),
                    "chrf": s.get("chrf"),
                    "num_examples": s.get("num_examples"),
                    "selected_by_dev": True,
                    "is_best_within_family": False,
                    "prediction_file": pred_file,
                    "score_file": str(score_file),
                    "status": status.get("status", "completed"),
                    "notes": "IPA native-representation metrics are not directly comparable to orthographic metrics." if config.get("tokenizer_family") == "ipa" else "",
                })
    best_by_family = {}
    dev_macros = [r for r in rows if r["split_or_benchmark"] == "dev" and r["metric_scope"] == "macro_avg" and r["status"] in {"completed", "partial"}]
    for family in sorted(set(r["tokenizer_family"] for r in dev_macros)):
        candidates = [r for r in dev_macros if r["tokenizer_family"] == family]
        metric = "chrf" if any(r.get("chrf") is not None for r in candidates) else "f1"
        best = max(candidates, key=lambda r: (r.get(metric) is not None, r.get(metric) or -1))
        best_by_family[family] = {"model_variant": best["model_variant"], "selection_metric": f"dev_macro_avg_{metric}", "score": best.get(metric)}
        for row in rows:
            if row["tokenizer_family"] == family and row["model_variant"] == best["model_variant"]:
                row["is_best_within_family"] = True
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(Path(args.out_csv).with_suffix(".csv.tmp"), args.out_csv)
    atomic_write_json(Path(args.out_json), rows)
    atomic_write_json(Path(args.best_out), best_by_family)
    mark_success(runs_dir / "_SUCCESS.aggregate_scores")
    print(json.dumps({"rows": len(rows), "best_by_family": best_by_family}, ensure_ascii=False, indent=2))


def cmd_resume(args: argparse.Namespace) -> None:
    state = json.loads(Path(args.state).read_text(encoding="utf-8")) if Path(args.state).exists() else {}
    manifest = read_jsonl(Path(args.runs_dir) / "manifest.jsonl") if (Path(args.runs_dir) / "manifest.jsonl").exists() else []
    print(json.dumps({"state": state, "manifest_records": len(manifest), "incomplete_runs": [r for r in manifest if r.get("status") != "completed"]}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CJK translation task pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-cjk-translation")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser("inspect-cjk-benchmark")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_inspect_benchmark)
    p = sub.add_parser("adapt-cjk-benchmark")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_adapt_benchmark)
    p = sub.add_parser("render-cjk-translation")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer-family", choices=FAMILIES, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("discover-cjk-models")
    p.add_argument("--search-roots", nargs="+", default=["out_multidata_single", "out_multidata_base", "out_multidata_base_byte"])
    p.add_argument("--out")
    p.set_defaults(func=cmd_discover)
    p = sub.add_parser("finetune-cjk-translation")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--search-roots", nargs="+", default=["out_multidata_single", "out_multidata_base", "out_multidata_base_byte"])
    p.add_argument("--resume", default="auto")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--lr-scheduler", choices=["none", "cosine", "exponential", "step", "plateau"], default="cosine")
    p.add_argument("--cosine-t-max", type=int, default=5000)
    p.add_argument("--cosine-eta-min", type=float, default=3e-6)
    p.add_argument("--adamw-weight-decay", type=float, default=0.1)
    p.add_argument("--adamw-betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-fn", default="cross_entropy")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_finetune)
    p = sub.add_parser("eval-cjk-translation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="cjk_translation")
    p.add_argument("--split", choices=["dev", "test", "train"])
    p.add_argument("--benchmark")
    p.add_argument("--tokenizer-family", choices=FAMILIES)
    p.add_argument("--ipa-eval-mode", choices=["native", "source_only"], default="native")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions-out", required=True)
    p.add_argument("--predictions-in")
    p.add_argument("--mock-predictions", action="store_true")
    p.add_argument("--resume", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--tiktoken-decode-mode", choices=["text", "bytes"], default="text")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("aggregate-cjk-scores")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--best-out", required=True)
    p.set_defaults(func=cmd_aggregate)
    p = sub.add_parser("resume-cjk-translation")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--state", required=True)
    p.set_defaults(func=cmd_resume)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""In-context-learning evaluation for the CJK translation task.

This evaluates the pretrained checkpoints selected by cjk_translation without
fine-tuning.  Prompts are built from the same rendered task JSONL files as the
SFT pipeline, with a configurable number of same-direction demonstrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

import cjk_translation_pipeline as cjk
from gpt_conf import GPTConfig
from model import GPT
from sample import get_tokenizer_functions


DEFAULT_CONFIG: dict[str, Any] = {
    "data_dir": "cjk_translation",
    "out_dir": "cjk_ICL/runs",
    "selected_json": "cjk_translation/selected_base_checkpoints.json",
    "families": ["tiktoken", "byte", "ipa"],
    "model_variants": [],
    "shot_counts": [0, 1],
    "eval_splits": ["dev", "test"],
    "benchmark": "benchmarks_easy.csv",
    "max_examples": None,
    "max_new_tokens": 128,
    "temperature": 1.0,
    "top_k": 1,
    "device": "cuda",
    "dtype": "bfloat16",
    "seed": 1337,
    "force": False,
    "tiktoken_decode_mode": "text",
    "drop_shots_over_block": True,
}


def load_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        user = json.loads(path.read_text(encoding="utf-8"))
        config.update(user)
    return config


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
    os.replace(tmp, path)


def stable_index(key: str, modulo: int) -> int:
    return int(cjk.hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % modulo


def load_selected(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = json.loads(Path(config["selected_json"]).read_text(encoding="utf-8"))
    families = set(config["families"])
    model_variants = set(config.get("model_variants") or [])
    out = []
    for row in rows:
        if row.get("tokenizer_family") not in families:
            continue
        names = {row.get("variation_key"), row.get("scoped_variation_key"), row.get("run_name")}
        if model_variants and not (names & model_variants):
            continue
        out.append(row)
    return out


def examples_for_eval(data_dir: Path, split: str | None, benchmark: str | None, family: str) -> tuple[list[dict[str, Any]], str, str, str]:
    if benchmark:
        adapted = data_dir / "benchmark_adapted" / "benchmarks_easy.jsonl"
        if not adapted.exists():
            cjk.cmd_adapt_benchmark(argparse.Namespace(benchmark=benchmark, out=str(adapted)))
        examples = cjk.read_jsonl(adapted)
        examples = cjk.render_benchmark_examples_for_family(examples, family, ipa_eval_mode="native")
        return examples, "benchmarks_easy", "benchmarks_easy.csv", "benchmark_easy"
    if split is None:
        raise ValueError("split or benchmark is required")
    examples = cjk.render_split_examples_for_eval(data_dir, split, family, ipa_eval_mode="native")
    return examples, split, split, split


def train_examples_by_direction(data_dir: Path, family: str) -> dict[str, list[dict[str, Any]]]:
    rows = cjk.read_jsonl(data_dir / "tasks" / family / "train.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["direction"]].append(row)
    return grouped


def demo_text(example: dict[str, Any]) -> str:
    return f"{example['prompt']} {example['completion']}\n\n"


def build_icl_prompt(
    query: dict[str, Any],
    demos_by_direction: dict[str, list[dict[str, Any]]],
    shot_count: int,
    encode: Callable[[str], list[int]],
    block_size: int,
    max_new_tokens: int,
    drop_shots_over_block: bool,
) -> tuple[str, int, list[str], str | None]:
    demos = demos_by_direction.get(query["direction"], [])
    selected: list[dict[str, Any]] = []
    if shot_count > 0:
        if not demos:
            return query["prompt"], 0, [], "no_demos_for_direction"
        start = stable_index(f"{query['id']}:{shot_count}", len(demos))
        seen_ids = {query["id"]}
        cursor = start
        while len(selected) < shot_count and len(seen_ids) <= len(demos):
            candidate = demos[cursor % len(demos)]
            cursor += 1
            if candidate["id"] in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate["id"])
    prefix = "".join(demo_text(ex) for ex in selected)
    prompt = prefix + query["prompt"]
    allowed = block_size - max_new_tokens
    if len(encode(query["prompt"])) > allowed:
        return query["prompt"], len(selected), [ex["id"] for ex in selected], "query_over_block"
    if len(encode(prompt)) <= allowed:
        return prompt, len(selected), [ex["id"] for ex in selected], None
    if not drop_shots_over_block:
        return prompt, len(selected), [ex["id"] for ex in selected], "prompt_over_block"
    while selected and len(encode("".join(demo_text(ex) for ex in selected) + query["prompt"])) > allowed:
        selected.pop(0)
    prompt = "".join(demo_text(ex) for ex in selected) + query["prompt"]
    return prompt, len(selected), [ex["id"] for ex in selected], "dropped_shots_over_block" if len(selected) < shot_count else None


def build_decode(meta: dict[str, Any], base_decode: Callable[[list[int]], str], mode: str) -> Callable[[list[int]], str]:
    if meta.get("tokenizer") != "tiktoken" or mode != "bytes":
        return base_decode
    import tiktoken
    enc = tiktoken.get_encoding(meta["tiktoken_encoding"])

    def decode(token_ids: list[int]) -> str:
        data = b"".join(enc.decode_single_token_bytes(token_id) for token_id in token_ids)
        return data.decode("utf-8", errors="replace")

    return decode


def load_model(checkpoint_path: Path, meta_path: Path, device: str):
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    encode, decode = get_tokenizer_functions(meta)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, meta, encode, decode


def generate_until_stop(model, start_ids, max_new_tokens: int, decode: Callable[[list[int]], str], temperature: float, top_k: int | None):
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
    return generated_text, generated_ids


def generate_predictions(
    model,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    examples: list[dict[str, Any]],
    demos_by_direction: dict[str, list[dict[str, Any]]],
    shot_count: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    device = config["device"]
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    ctx = nullcontext() if device_type == "cpu" or config["dtype"] == "float32" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    rows = []
    max_examples = config.get("max_examples")
    torch.manual_seed(int(config["seed"]))
    with torch.no_grad():
        for idx, ex in enumerate(examples):
            if max_examples is not None and idx >= int(max_examples):
                break
            prompt, actual_shots, demo_ids, skip_reason = build_icl_prompt(
                ex,
                demos_by_direction,
                shot_count,
                encode,
                model.config.block_size,
                int(config["max_new_tokens"]),
                bool(config["drop_shots_over_block"]),
            )
            if skip_reason == "query_over_block" or (skip_reason == "prompt_over_block" and not config["drop_shots_over_block"]):
                rows.append({
                    **ex,
                    "icl_prompt": prompt,
                    "configured_shots": shot_count,
                    "actual_shots": actual_shots,
                    "demo_ids": demo_ids,
                    "prediction": "",
                    "reference": ex["completion"],
                    "skip_reason": skip_reason,
                })
                continue
            prompt_ids = encode(prompt)
            remaining = model.config.block_size - len(prompt_ids)
            max_new_tokens = min(int(config["max_new_tokens"]), remaining)
            start_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
            with ctx:
                generated, generated_ids = generate_until_stop(
                    model,
                    start_ids,
                    max_new_tokens=max_new_tokens,
                    decode=decode,
                    temperature=float(config["temperature"]),
                    top_k=config["top_k"],
                )
            rows.append({
                **ex,
                "icl_prompt": prompt,
                "configured_shots": shot_count,
                "actual_shots": actual_shots,
                "demo_ids": demo_ids,
                "prediction": generated.split("\n", 1)[0].strip(),
                "reference": ex["completion"],
                "skip_reason": skip_reason,
                "generated_token_ids": generated_ids,
            })
    return rows


def run_variant(row: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    family = row["tokenizer_family"]
    data_dir = Path(config["data_dir"])
    out_root = Path(config["out_dir"])
    variant = row["variation_key"]
    checkpoint = Path(row["checkpoint"])
    meta_path = Path(row["base_meta"])
    demos_by_direction = train_examples_by_direction(data_dir, family)
    model, meta, encode, decode = load_model(checkpoint, meta_path, config["device"])
    decode = build_decode(meta, decode, config["tiktoken_decode_mode"])
    result_rows = []
    for shot_count in config["shot_counts"]:
        shot_count = int(shot_count)
        run_dir = out_root / variant / f"shots_{shot_count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_config = {
            "model_variant": variant,
            "checkpoint": str(checkpoint),
            "base_meta": str(meta_path),
            "tokenizer_family": family,
            "text_representation": row.get("text_representation"),
            "shot_count": shot_count,
            "prompt_format": "same prompt as SFT; demonstrations are prompt + completion followed by a blank line",
            "ipa_mode": "native_source_ipa_to_target_ipa" if family == "ipa" else None,
            "config": config,
        }
        atomic_write_json(run_dir / "prompt_config.json", prompt_config)
        eval_specs: list[tuple[list[dict[str, Any]], str, str, str]] = []
        for split in config["eval_splits"]:
            eval_specs.append(examples_for_eval(data_dir, split, None, family))
        if config.get("benchmark"):
            eval_specs.append(examples_for_eval(data_dir, None, config["benchmark"], family))
        for examples, eval_dataset, split_or_benchmark, stem in eval_specs:
            pred_path = run_dir / f"{stem}_predictions.jsonl"
            score_path = run_dir / f"{stem}_scores.json"
            if pred_path.exists() and score_path.exists() and not config["force"]:
                score = json.loads(score_path.read_text(encoding="utf-8"))
            else:
                preds = generate_predictions(model, encode, decode, examples, demos_by_direction, shot_count, config)
                write_jsonl(pred_path, preds)
                score = cjk.score_predictions(preds, eval_dataset, split_or_benchmark, str(checkpoint))
                score.update({
                    "prediction_file": str(pred_path),
                    "score_file": str(score_path),
                    "model_variant": variant,
                    "tokenizer_family": family,
                    "shot_count": shot_count,
                    "checkpoint_path": str(checkpoint),
                    "base_meta": str(meta_path),
                    "ipa_mode": "native_source_ipa_to_target_ipa" if family == "ipa" else None,
                })
                atomic_write_json(score_path, score)
            macro = score["macro_avg"]
            result_rows.append({
                "model_variant": variant,
                "tokenizer_family": family,
                "shot_count": shot_count,
                "eval_dataset": eval_dataset,
                "split_or_benchmark": split_or_benchmark,
                "checkpoint_path": str(checkpoint),
                "prediction_file": str(pred_path),
                "score_file": str(score_path),
                "f1": macro.get("f1"),
                "exact_match": macro.get("exact_match"),
                "bleu": macro.get("bleu"),
                "chrf": macro.get("chrf"),
                "num_examples": macro.get("num_examples"),
            })
    return result_rows


def write_aggregate(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "all_icl_scores.json"
    csv_path = out_dir / "all_icl_scores.csv"
    atomic_write_json(json_path, rows)
    columns = [
        "model_variant", "tokenizer_family", "shot_count", "eval_dataset", "split_or_benchmark",
        "checkpoint_path", "f1", "exact_match", "bleu", "chrf", "num_examples",
        "prediction_file", "score_file",
    ]
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})
    os.replace(tmp, csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CJK in-context-learning evaluation.")
    parser.add_argument("--config", default="cjk_ICL/config.example.json")
    parser.add_argument("--families", nargs="+", choices=cjk.FAMILIES)
    parser.add_argument("--model-variants", nargs="+")
    parser.add_argument("--shot-counts", nargs="+", type=int)
    parser.add_argument("--eval-splits", nargs="+", choices=["dev", "test", "train"])
    parser.add_argument("--benchmark")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    for key in ("families", "model_variants", "shot_counts", "eval_splits"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    for key in ("benchmark", "max_examples", "max_new_tokens", "temperature", "top_k", "device", "dtype"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.force:
        config["force"] = True
    selected = load_selected(config)
    if not selected:
        raise ValueError("No selected checkpoints match the requested families/model variants")
    out_dir = Path(config["out_dir"])
    atomic_write_json(out_dir / "resolved_config.json", config)
    all_rows = []
    started = time.time()
    for idx, row in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {row['tokenizer_family']} {row['variation_key']}", flush=True)
        all_rows.extend(run_variant(row, config))
        write_aggregate(out_dir, all_rows)
    atomic_write_json(out_dir / "run_summary.json", {
        "completed_at": cjk.now(),
        "elapsed_seconds": time.time() - started,
        "num_model_variants": len(selected),
        "num_score_rows": len(all_rows),
        "aggregate_csv": str(out_dir / "all_icl_scores.csv"),
        "aggregate_json": str(out_dir / "all_icl_scores.json"),
    })
    print(json.dumps({"aggregate_csv": str(out_dir / "all_icl_scores.csv"), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

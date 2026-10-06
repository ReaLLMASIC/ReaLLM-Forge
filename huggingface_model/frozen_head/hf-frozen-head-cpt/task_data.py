"""GSM8K SFT preparation, response-only labels, and transparent numerical scoring."""
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import check_identity, digest, file_sha, read_json, write_json
from prepare_data import prepare

DEFAULT_PROMPT = "Solve the problem. Show your work and end with #### followed by the final numerical answer.\nQuestion: {question}\nAnswer:\n"
NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"


def final_number(text, strict=True):
    """Decimal normalization: no executable expression parsing; exact numeric equality."""
    if strict:
        match = re.findall(r"####\s*\$?\s*(" + NUMBER + r")", text)
    else:
        marker = final_number(text, strict=True)
        if marker is not None:
            return marker
        match = re.findall(NUMBER, text)
    if not match:
        return None
    try:
        number = Decimal(match[-1].replace(",", ""))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def score_answer(prediction, answer):
    gold = final_number(answer)
    if gold is None:
        raise ValueError("Reference solution has no valid #### final answer")
    strict, relaxed = final_number(prediction), final_number(prediction, strict=False)
    return {"strict_correct": strict is not None and strict == gold,
            "relaxed_correct": relaxed is not None and relaxed == gold,
            "format_valid": strict is not None, "gold": str(gold),
            "strict_number": str(strict) if strict is not None else None,
            "relaxed_number": str(relaxed) if relaxed is not None else None}


def encode_example(tokenizer, question, answer, max_length, prompt_template=DEFAULT_PROMPT):
    prompt = prompt_template.format(question=question)
    # Explicitly concatenate the exact inference prompt tokens with response tokens.
    # This avoids prefix-length mistakes caused by BPE merges across the boundary.
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if not p or not a or len(p) + len(a) > max_length:
        return None  # Do not truncate the final answer away.
    return {"question": question, "answer": answer, "prompt": prompt,
            "input_ids": p + a, "labels": [-100] * len(p) + a,
            "prompt_ids": p, "question_hash": hashlib.sha256(question.strip().encode()).hexdigest()}


def collate_examples(examples, pad_id, device="cpu"):
    n = max(len(x["input_ids"]) for x in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        length = len(row["input_ids"])
        ids[i, :length] = torch.tensor(row["input_ids"])
        labels[i, :length] = torch.tensor(row["labels"])
        mask[i, :length] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def task_manifest(c):
    return read_json(Path(c["output"]) / "data" / "task_manifest.json")


def task_examples(c, split):
    path = Path(c["output"]) / "data" / f"math_{split}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def prepare_task(c):
    prepare(c)  # The task config supplies only corpus A, for language retention/replay.
    root = Path(c["output"]) / "data"
    spec = c["sft"]
    check_identity(root / "task_identity.json", {"sft": spec, "seq_len": c["seq_len"]})
    if (root / "task_manifest.json").exists():
        m = task_manifest(c)
        for split, checksum in m["hashes"].items():
            if file_sha(root / f"math_{split}.jsonl") != checksum:
                raise RuntimeError("Prepared task data changed")
        print("Reusing verified task examples", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    if spec.get("local_jsonl"):
        mapping = spec["local_jsonl"]
        source = {"local_hashes": {str(Path(v).resolve()): file_sha(v) for v in mapping.values()}}
        pools = {s: [json.loads(x) for x in Path(p).read_text().splitlines()] for s, p in mapping.items()}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        sha = HfApi().dataset_info(spec.get("id", "openai/gsm8k"), revision=spec.get("revision", "main")).sha
        source = {"id": spec.get("id", "openai/gsm8k"), "revision": sha, "config": spec.get("config", "main")}
        ds = load_dataset(source["id"], source["config"], revision=sha)
        pools = {"train": list(ds["train"]), "test": list(ds["test"])}
    selected = {s: [] for s in ("train", "validation", "test")}
    seen, skipped_length, duplicates = set(), 0, 0
    # Reserve official test questions first, even if malformed/overlong, so none can enter training.
    reserved_test = {x["question"].strip() for x in pools["test"]}
    explicit_val = "validation" in pools
    reserved_val = {x["question"].strip() for x in pools.get("validation", [])}
    for original_split, rows in pools.items():
        for raw in rows:
            question, answer = raw["question"].strip(), raw["answer"].strip()
            qhash = hashlib.sha256(question.encode()).hexdigest()
            if qhash in seen or (original_split != "test" and question in reserved_test) or (
                    original_split == "train" and question in reserved_val):
                duplicates += 1
                continue
            seen.add(qhash)
            if original_split == "train" and not explicit_val:
                split = "validation" if int(qhash[:16], 16) / 2**64 < spec.get("validation_fraction", 0.1) else "train"
            else:
                split = original_split
            if final_number(answer) is None:
                raise ValueError("Reference answer must end with a GSM8K-style #### numerical answer")
            row = encode_example(tokenizer, question, answer, c["seq_len"] + 1, spec.get("prompt", DEFAULT_PROMPT))
            if row is None:
                skipped_length += 1
            else:
                selected[split].append(row)
    hashes, sizes = {}, {}
    for split, rows in selected.items():
        rows.sort(key=lambda r: r["question_hash"])
        limit = spec.get(f"{split}_limit")
        if limit:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"No task examples in {split}; check sequence length and source splits")
        path = root / f"math_{split}.jsonl"
        tmp = path.with_suffix(".partial")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
        tmp.replace(path)
        hashes[split], sizes[split] = file_sha(path), len(rows)
    write_json(root / "task_manifest.json", {"source": source, "hashes": hashes, "sizes": sizes,
               "skipped_overlong": skipped_length, "skipped_duplicates": duplicates,
               "prompt": spec.get("prompt", DEFAULT_PROMPT), "scoring_version": "decimal-final-number-v1",
               "test_scope": "Official GSM8K test split, optionally restricted by length and deterministic subset limit; report these restrictions."})
    print(f"Task split counts: {sizes}; overlong skipped: {skipped_length}", flush=True)

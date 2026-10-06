"""Deterministic fresh synthetic episodes and bounded, document-separated token files."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer


class ByteTokenizer:
    """Offline test tokenizer, never used for pretrained experiments."""
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [b+3 for b in text.encode()]
    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes([i-3 for i in ids if 3 <= i < 259]).decode(errors="replace")


def tokenizer_for(cfg, checkpoint=None):
    if cfg.tiny: return ByteTokenizer()
    local = Path(checkpoint)/"tokenizer" if checkpoint is not None else None
    if local is not None and local.exists():
        return AutoTokenizer.from_pretrained(local, local_files_only=True)
    return AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, trust_remote_code=False)


def chat_parts(tokenizer, enabled):
    if not enabled or isinstance(tokenizer, ByteTokenizer): return [], []
    marker = "SLERP_UNIQUE_BODY_MARKER_8d39af"
    text = tokenizer.apply_chat_template([{"role":"user", "content":marker}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.count(marker) != 1: raise ValueError("Cannot delimit this tokenizer's chat template")
    before, after = text.split(marker)
    return tokenizer.encode(before, add_special_tokens=False), tokenizer.encode(after, add_special_tokens=False)


@dataclass
class Episode:
    prompt: list
    answer: list
    target: str
    metadata: dict

    def tensors(self, device):
        seq = self.prompt + self.answer
        ids = torch.tensor([seq[:-1]], device=device)
        labels = torch.tensor([seq[1:]], device=device)
        if self.metadata.get("task") != "natural":
            labels[:, :len(self.prompt)-1] = -100
        return ids, labels


def synthetic(tokenizer, total_length, seed, index, split="train", task="recall",
              fact_count=8, position=0.1, window=2048, chunk=256, chat_template=False):
    """Length includes answer; training and eval use disjoint namespaces/templates."""
    rng = random.Random(f"{seed}:{split}:{index}:{task}:{fact_count}")
    prefix = "t" if split == "train" else "z"
    keys = [prefix + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=8)) for _ in range(fact_count)]
    vals = ["".join(rng.choices("0123456789", k=6)) for _ in keys]
    if fact_count < 1 or not 0 <= position <= 1: raise ValueError("Invalid fact count/position")
    query_index = rng.randrange(fact_count)
    key, target = keys[query_index], vals[query_index]
    if split == "train":
        records = [f"\nRecord: {k} = {v}.\n" for k, v in zip(keys, vals)]
        question = f"\nWhat is the value of {key}? Reply with the value only.\nAnswer: "
    else:
        records = [f"\nEntry [{k}]: {v}.\n" for k, v in zip(keys, vals)]
        question = f"\nReturn only the stored value associated with [{key}].\nAnswer: "
    evidence_indices = [query_index]
    if task == "update":
        target = "".join(rng.choices("0123456789", k=6))
        records.append(f"\nCorrection: {key} = {target}. This replaces the previous value.\n")
        evidence_indices = [len(records)-1]
        question = f"\nWhat is the latest value of {key}? Reply with the value only.\nAnswer: "
    elif task == "trace":
        if fact_count < 2:
            raise ValueError("Trace requires at least two records")
        referenced = rng.choice([j for j in range(fact_count) if j != query_index])
        records[query_index] = f"\n{key} refers to {keys[referenced]}.\n"
        target = vals[referenced]
        evidence_indices = [query_index, referenced]
        question = f"\nFollow {key} to its referenced key. Return that key's value only.\nAnswer: "
    elif task == "multi":
        n = min(3, len(keys))
        evidence_indices = rng.sample(range(len(keys)), n)
        target = ", ".join(vals[j] for j in evidence_indices)
        question = "\nReturn the values of " + ", ".join(keys[j] for j in evidence_indices) + " in that order, comma separated.\nAnswer: "
    elif task not in {"recall", "capacity"}:
        raise ValueError(f"Unknown task {task}")
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    answer = enc(target + "\n")
    if tokenizer.eos_token_id is not None: answer.append(tokenizer.eos_token_id)
    chat_before, chat_after = chat_parts(tokenizer, chat_template)
    header = chat_before + enc("Read the records and retain their values. The question comes at the end.\n")
    q = enc(question) + chat_after
    blocks = [enc(r) for r in records]
    budget = total_length - len(answer) - len(header) - len(q)
    free = budget - sum(map(len, blocks))
    if free < 0:
        raise ValueError(f"{fact_count} facts do not fit {total_length} tokens")
    # One controlled region with distractors between records; every exact token span recorded.
    before = int(free * position)
    middle = int((free-before) * 0.15)
    after = free - before - middle
    fillers = [enc(text) for text in [
        " The quiet garden has trees and a path. Nothing is recorded here.",
        " A visitor describes a river, hills, and clouds crossing the sky.",
        " The schedule contains no stored key values. This paragraph is background.",
        " Someone moved the books from a table to a shelf by the window."]]
    def fill(n):
        result = []
        while len(result) < n: result.extend(rng.choice(fillers))
        return result[:n]
    prompt = header + fill(before)
    spans = []
    for i, block in enumerate(blocks):
        start = len(prompt)
        prompt += block
        spans.append([start, len(prompt)])
        if i < len(blocks)-1:
            n = middle // max(1, len(blocks)-1) + (i < middle % max(1, len(blocks)-1))
            prompt += fill(n)
    if len(blocks) == 1:
        prompt += fill(middle)
    prompt += fill(after) + q
    assert len(prompt) + len(answer) == total_length
    # At the final prompt token: number of canonical evictions performed.
    dropped = max(0, ((len(prompt)-1-window)//chunk + 1) * chunk)
    relevant = [spans[i] for i in evidence_indices]
    metadata = {"example_id": f"{split}-{seed}-{index}-{task}-{fact_count}-{position}",
        "split": split, "seed": seed, "index": index, "task": task,
        "generator_version":"0.2.0", "query_record_index":query_index,
        "evidence_record_indices":evidence_indices, "chat_template":chat_template,
        "total_length": total_length, "prompt_length": len(prompt), "fact_count": fact_count,
        "position_fraction": position, "evidence_spans": relevant,
        "distance": len(prompt)-max(b for a,b in relevant),
        "all_evidence_evicted": all(b <= dropped for a,b in relevant),
        "any_evidence_evicted": any(b <= dropped for a,b in relevant),
        "prompt_sha256": hashlib.sha256(np.asarray(prompt,dtype=np.uint32).tobytes()).hexdigest()}
    return Episode(prompt, answer, target, metadata)


class TokenDocuments:
    def __init__(self, path, split):
        path = Path(path)
        self.manifest=json.loads((path/f"{split}.manifest.json").read_text())
        self.tokens = np.memmap(path / f"{split}.bin", dtype=np.uint32, mode="r")
        if len(self.tokens)!=self.manifest["tokens"]:
            raise ValueError("Token file length disagrees with completed preparation manifest")
        self.rows = [json.loads(s) for s in (path/f"{split}.index.jsonl").read_text().splitlines()]

    def verify_tokenizer(self,cfg):
        if self.manifest["model"]!=cfg.model_id or self.manifest["model_revision"]!=cfg.revision:
            raise ValueError("Prepared data tokenizer/model revision differs from run")

    def episode(self, length, seed, index):
        candidates = [r for r in self.rows if r["length"] >= length]
        if not candidates:
            raise ValueError(f"No prepared document has {length} tokens; prepare longer books")
        rng = random.Random(f"natural:{seed}:{index}")
        row = rng.choice(candidates)
        start = row["offset"] + rng.randrange(row["length"]-length+1)
        ids = self.tokens[start:start+length].astype(np.int64).tolist()
        return Episode(ids[:-1], ids[-1:], "", {"task":"natural", "book_id":row["book_id"],
            "offset":start, "total_length":length, "split":row["split"]})


def prepare_documents(cfg, out, token_limit, split, revision, min_length=32768):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from .storage import disk_guard
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out/f"{split}.bin", out/f"{split}.index.jsonl"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Prepared split already exists; choose a new output directory")
    dataset_id = "emozilla/pg19"
    sha = HfApi().dataset_info(dataset_id, revision=revision).sha
    cfg.revision = HfApi().model_info(cfg.model_id, revision=cfg.revision).sha
    tokenizer = tokenizer_for(cfg)
    stream = load_dataset(dataset_id, split=split, revision=sha, streaming=True)
    count, books = 0, 0
    # No full corpus or Arrow cache; capped token output, one book at a time.
    with paths[0].open("wb") as data, paths[1].open("w") as idx:
        for source_index, row in enumerate(stream):
            disk_guard(cfg)
            text = row["text"]
            clipped = text[:2_000_000]  # explicit host-RAM cap per document
            ids = tokenizer.encode(clipped, add_special_tokens=False)
            original_count=len(ids)
            ids = ids[:min(262144, token_limit-count)]
            if len(ids) < min_length:
                if token_limit-count < min_length:
                    break
                continue
            arr = np.asarray(ids, dtype=np.uint32)
            arr.tofile(data)
            book_id = str(row.get("id", row.get("short_book_title", source_index)))
            meta = {"book_id":book_id, "source_index":source_index, "split":split,
                "offset":count, "length":len(ids), "source_sha256":hashlib.sha256(text.encode()).hexdigest(),
                "truncated":len(text)>len(clipped) or len(ids)<original_count}
            idx.write(json.dumps(meta)+"\n")
            count += len(ids); books += 1
            print(json.dumps({"prepared_tokens":count,"books":books}), flush=True)
            if count >= token_limit:
                break
    source_hashes={json.loads(s)["source_sha256"] for s in paths[1].read_text().splitlines()}
    for other_index in out.glob("*.index.jsonl"):
        if other_index!=paths[1]:
            other_hashes={json.loads(s)["source_sha256"] for s in other_index.read_text().splitlines()}
            if source_hashes & other_hashes:
                raise ValueError("Source document overlap across prepared splits")
    (out/f"{split}.manifest.json").write_text(json.dumps({"dataset":dataset_id,"revision":sha,
        "split":split,"tokens":count,"books":books,"model":cfg.model_id,"model_revision":cfg.revision,
        "tokenizer_class":type(tokenizer).__name__,"format":"uint32-little-endian"}, indent=2)+"\n")
    if count == 0:
        raise RuntimeError("No suitable books were prepared")


def training_episode(cfg, tokenizer, index, documents=None):
    rng = random.Random(f"episode:{cfg.seed}:{index}")
    length = rng.choices(cfg.lengths, weights=cfg.length_weights, k=1)[0]
    if documents is not None and rng.random() < cfg.natural_fraction:
        return documents.episode(length, cfg.seed, index)
    task = rng.choice(cfg.tasks)
    # Tiny smoke examples still contain facts, distractors and a late query.
    count = 2 if cfg.tiny else rng.choice(cfg.fact_counts)
    return synthetic(tokenizer, length, cfg.seed, index, task=task, fact_count=count,
        position=rng.choice(cfg.positions), window=cfg.window, chunk=cfg.chunk,
        chat_template=cfg.chat_template)

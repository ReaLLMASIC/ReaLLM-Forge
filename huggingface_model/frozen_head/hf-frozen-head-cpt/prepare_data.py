"""Pin Hugging Face revisions; stream, split, and pack data once for all arms."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import check_identity, digest, file_sha, read_json, write_json


def resolve_model(c):
    if Path(c["model"]).is_dir():
        root = Path(c["model"])
        files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model"))
        return {"id": str(root.resolve()), "revision": None,
                "local_hashes": {p.name: file_sha(p) for p in files}}
    from huggingface_hub import HfApi
    info = HfApi().model_info(c["model"], revision=c["revision"])
    return {"id": c["model"], "revision": info.sha}


def route_record(record, spec):
    """Hash complete documents/repositories BEFORE tokenization, never random windows."""
    text = record[spec.get("text_field", "text")]
    group_key = spec.get("group_field")
    key = str(record[group_key]) if group_key else text
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    v = spec.get("validation_fraction", 0.05)
    t = spec.get("test_fraction", 0.05)
    if v <= 0 or t <= 0 or v + t >= 1:
        raise ValueError("Need positive validation/test fractions with sum < 1")
    return "validation" if value < v else "test" if value < v + t else "train"


class Packer:
    def __init__(self, path, seq_len, token_budget, vocab_size):
        self.path, self.seq_len = Path(path), seq_len
        self.limit = math.ceil(token_budget / seq_len)
        self.f = open(path, "wb")
        self.buffer, self.blocks, self.documents = [], 0, 0
        self.counts = np.zeros(vocab_size, dtype=np.int64)

    @property
    def full(self):
        return self.blocks >= self.limit

    def add(self, ids):
        if self.full:
            return
        self.documents += 1
        self.buffer.extend(ids)
        consumed = 0
        while len(self.buffer) - consumed >= self.seq_len + 1 and not self.full:
            block = np.asarray(self.buffer[consumed:consumed + self.seq_len + 1], dtype="<u4")
            block.tofile(self.f)
            self.counts += np.bincount(block[1:], minlength=len(self.counts))
            self.blocks += 1
            consumed += self.seq_len
        self.buffer = self.buffer[consumed:] if not self.full else []

    def close(self):
        self.f.close()
        if self.blocks == 0:
            raise ValueError(f"No complete blocks in {self.path}. Increase data limits or reduce sequence length.")
        np.save(self.path.with_suffix(".counts.npy"), self.counts)
        return {"blocks": self.blocks, "target_tokens": self.blocks * self.seq_len,
                "documents": self.documents, "requested_blocks": self.limit,
                "sha256": file_sha(self.path), "counts_sha256": file_sha(self.path.with_suffix(".counts.npy"))}


def iter_records(spec, split, revision):
    if "local_jsonl" in spec:
        mapping = spec["local_jsonl"]
        path = mapping[split] if isinstance(mapping, dict) else mapping
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    from datasets import load_dataset
    ds = load_dataset(spec["id"], name=spec.get("config"), split=split,
                      revision=revision, streaming=True)
    # Shuffle documents with a bounded buffer; the prepared files lock their exact order.
    if spec.get("shuffle_buffer", 0):
        ds = ds.shuffle(seed=spec.get("shuffle_seed", 1729), buffer_size=spec["shuffle_buffer"])
    yield from ds


def prepare(c):
    root = Path(c["output"])
    root.mkdir(parents=True, exist_ok=True)
    identity = {"model": c["model"], "revision": c["revision"], "data": c["data"], "seq_len": c["seq_len"]}
    check_identity(root / "data_identity.json", identity)
    target = root / "data"
    if (target / "manifest.json").exists():
        manifest = read_json(target / "manifest.json")
        for domain in c["data"]:
            for split in ("train", "validation", "test"):
                path = target / f"{domain}_{split}.bin"
                meta = manifest["corpora"][domain]["splits"][split]
                if file_sha(path) != meta["sha256"] or file_sha(path.with_suffix(".counts.npy")) != meta["counts_sha256"]:
                    raise RuntimeError(f"Prepared data changed: {path}")
        print(f"Reusing verified prepared data: {target}", flush=True)
        return
    tmp = root / "data.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model = resolve_model(c)
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"], trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token; choose/configure one explicitly")
    tokenizer.save_pretrained(tmp / "tokenizer")
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    manifest = {"version": 1, "model": model, "seq_len": c["seq_len"],
                "tokenizer_vocab_size": vocab_size, "eos_token_id": tokenizer.eos_token_id,
                "tokenizer_digest": digest(tokenizer.get_vocab()), "corpora": {}}
    # Cross-split and cross-corpus exact-text dedup. Small pilot corpora keep this bounded.
    seen = set()
    for domain in c["data"]:
        spec = c["data"][domain]
        local = "local_jsonl" in spec
        if local:
            mapping = spec["local_jsonl"]
            files = mapping.values() if isinstance(mapping, dict) else [mapping]
            source = {"local_hashes": {str(Path(p).resolve()): file_sha(p) for p in files}}
            revision = None
        else:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info(spec["id"], revision=spec.get("revision", "main")).sha
            source = {"id": spec["id"], "revision": revision, "config": spec.get("config")}
        writers = {s: Packer(tmp / f"{domain}_{s}.bin", c["seq_len"], spec[f"{s}_tokens"], vocab_size)
                   for s in ("train", "validation", "test")}
        splits = spec.get("splits")
        sources = list(splits.items()) if splits else [(None, spec.get("source_split", "train"))]
        skipped, scanned = 0, 0
        for fixed_split, source_split in sources:
            for record in iter_records(spec, source_split, revision):
                scanned += 1
                text = record.get(spec.get("text_field", "text"))
                if not isinstance(text, str) or len(text.strip()) < spec.get("min_chars", 64):
                    continue
                split = fixed_split or route_record(record, spec)
                if writers[split].full:
                    if all(w.full for w in writers.values()) or fixed_split:
                        break
                    continue
                key = hashlib.sha256(text.strip().encode()).digest()
                if key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                writers[split].add(ids + [tokenizer.eos_token_id])
                if scanned % 5000 == 0:
                    print(domain, scanned, {s: w.blocks for s, w in writers.items()}, flush=True)
                if spec.get("max_scan_documents") and scanned >= spec["max_scan_documents"]:
                    break
        stats = {s: w.close() for s, w in writers.items()}
        manifest["corpora"][domain] = {"source": source, "spec": spec, "splits": stats,
                                         "exact_duplicates_skipped": skipped, "documents_scanned": scanned}
        print(f"Prepared {domain}: {stats}", flush=True)
    write_json(tmp / "manifest.json", manifest)
    os.replace(tmp, target)
    print(f"Prepared data saved to {target}", flush=True)


class TokenBlocks:
    def __init__(self, path, seq_len):
        self.path, self.seq_len = Path(path), seq_len
        nbytes = self.path.stat().st_size
        if nbytes % (4 * (seq_len + 1)):
            raise ValueError(f"Corrupt packed file: {path}")
        self.data = np.memmap(path, dtype="<u4", mode="r").reshape(-1, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def batch(self, indices, device):
        import torch
        return torch.from_numpy(np.asarray(self.data[indices], dtype=np.int64).copy()).to(device)


class BlockOrder:
    """Counter-based epoch permutations make checkpoint replay independent of dataloader state."""
    def __init__(self, size, seed, stream):
        self.size, self.seed, self.stream, self.cache = size, seed, stream, {}

    def indices(self, start, count):
        out = []
        for pos in range(start, start + count):
            epoch, within = divmod(pos, self.size)
            if epoch not in self.cache:
                self.cache = {epoch: np.random.default_rng(np.random.SeedSequence([self.seed, self.stream, epoch])).permutation(self.size)}
            out.append(int(self.cache[epoch][within]))
        return out

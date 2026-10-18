"""Prepare immutable, document-disjoint packed token streams once for all arms."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .common import write_json, digest, environment


def document_split(text, seed, validation_permyriad):
    # Content rather than row ID: exact duplicates always choose the same split.
    key = hashlib.sha256((str(seed) + "\0" + text.strip()).encode()).digest()
    return "validation" if int.from_bytes(key[:8], "little") % 10000 < validation_permyriad else "train"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--local-jsonl", help="Optional local documents, one JSON object with a text field per line")
    p.add_argument("--tokenizer", default="openai-community/gpt2")
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument("--train-tokens", type=int, default=2_001_000_000)
    p.add_argument("--validation-tokens", type=int, default=2_000_000)
    p.add_argument("--validation-permyriad", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--tokenize-batch", type=int, default=128)
    a = p.parse_args()
    if min(a.train_tokens, a.validation_tokens, a.tokenize_batch) < 2 or not 0 < a.validation_permyriad < 10000:
        p.error("positive budgets and a nonempty split are required")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error("Output must be empty. Existing data is immutable; use a new output directory.")
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    tokenizer_revision = None if Path(a.tokenizer).exists() else HfApi().model_info(a.tokenizer, revision=a.tokenizer_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, revision=tokenizer_revision, use_fast=True)
    if tokenizer.eos_token_id is None: raise ValueError("Tokenizer must define EOS")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**12
    tokenizer.save_pretrained(out / "tokenizer")
    dataset_revision = None
    if a.local_jsonl:
        def local_rows():
            with open(a.local_jsonl) as f:
                for line in f:
                    if line.strip(): yield json.loads(line)
        rows = local_rows()
        source = dict(local_jsonl=str(Path(a.local_jsonl).resolve()), sha256=digest(a.local_jsonl))
    else:
        from datasets import load_dataset
        dataset_revision = HfApi().dataset_info(a.dataset, revision=a.dataset_revision).sha
        rows = load_dataset(a.dataset, name=a.dataset_config, split=a.split, streaming=True,
                            revision=dataset_revision).shuffle(seed=a.seed, buffer_size=a.shuffle_buffer)
        source = dict(dataset=a.dataset, config=a.dataset_config, split=a.split, revision=dataset_revision,
                      shuffle_buffer=a.shuffle_buffer)
    dtype = "<u2" if len(tokenizer) <= 65536 else "<u4"
    targets = dict(train=a.train_tokens, validation=a.validation_tokens)
    counts = {s: 0 for s in targets}
    documents = {s: 0 for s in targets}
    handles = {s: open(out / f"{s}.bin", "wb") for s in targets}
    max_id, scanned = 0, 0

    def consume(batch):
        nonlocal max_id
        texts = [item[1] for item in batch]
        tokenized = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for (split, _), ids in zip(batch, tokenized):
            ids = ids + [tokenizer.eos_token_id]
            ids = ids[:targets[split] - counts[split]]
            if not ids: continue
            max_id = max(max_id, max(ids))
            np.asarray(ids, dtype=dtype).tofile(handles[split])
            counts[split] += len(ids)
            documents[split] += 1

    batch = []
    try:
        for row in rows:
            text = row.get(a.text_column)
            scanned += 1
            if not isinstance(text, str) or not text.strip(): continue
            split = document_split(text, a.seed, a.validation_permyriad)
            if counts[split] < targets[split]: batch.append((split, text))
            if len(batch) >= a.tokenize_batch:
                consume(batch)
                batch = []
                if scanned % 10000 < a.tokenize_batch:
                    print(json.dumps(dict(documents_scanned=scanned, tokens=counts)), flush=True)
            if all(counts[s] >= targets[s] for s in targets): break
        if batch: consume(batch)
    finally:
        for f in handles.values(): f.close()
    if counts != targets:
        raise RuntimeError(f"Source exhausted: {counts}, requested {targets}. Incomplete directory has no manifest; retry in a fresh directory.")
    manifest = dict(schema=1, dtype=dtype, source=source, tokenizer=a.tokenizer,
                    tokenizer_revision=tokenizer_revision, tokenizer_vocab_size=len(tokenizer),
                    eos_token_id=tokenizer.eos_token_id, max_token_id=max_id,
                    seed=a.seed, validation_permyriad=a.validation_permyriad,
                    split_policy="SHA256(seed + normalized document text); no exact-document cross-split overlap",
                    packing="EOS between documents; attention may cross EOS; exactly T shifted labels per block",
                    documents_scanned=scanned, environment=environment(),
                    splits={s: dict(tokens=counts[s], documents=documents[s], sha256=digest(out / f"{s}.bin")) for s in counts},
                    tokenizer_files={f.name: digest(f) for f in sorted((out / "tokenizer").iterdir()) if f.is_file()})
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {counts}; tokenizer size={len(tokenizer)}; output={out}")


if __name__ == "__main__": main()

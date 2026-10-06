"""Bounded streaming Hugging Face preparation; uint16/uint32 token memmaps."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validation_document(text, seed, fraction=0.01):
    # Split DOCUMENTS before token packing. Exact duplicates route to the same
    # split; this is not a near-duplicate decontamination algorithm.
    digest = hashlib.blake2b(f"{seed}:".encode()+text.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64 < fraction


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    out = Path(args.data)
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}; choose a new data directory.")
    tmp = out.with_name(out.name + ".preparing")
    if tmp.exists():
        raise FileExistsError(f"Incomplete preparation at {tmp}; inspect/remove it before retrying.")
    if min(args.train_tokens, args.val_tokens) < 2:
        raise ValueError("Token budgets must be >=2.")
    tmp.mkdir(parents=True)
    api = HfApi()
    ds_revision = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
    tok_revision = api.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=tok_revision)
    tok.model_max_length = 10**12
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token for document packing.")
    dtype = "<u2" if len(tok) <= 65536 else "<u4"
    required = (args.train_tokens + args.val_tokens)*np.dtype(dtype).itemsize + 2*1024**3
    if shutil.disk_usage(tmp).free < required:
        raise OSError(f"Need at least {required/1024**3:.2f} GiB free for tokens and safety margin.")
    tok.save_pretrained(tmp / "tokenizer")
    limits = {"train": args.train_tokens, "val": args.val_tokens}
    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    files = {s: open(tmp/f"{s}.bin", "wb") for s in counts}
    dataset_kwargs = dict(path=args.dataset, revision=ds_revision, streaming=True)
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    def consume(stream, designated_split=None):
        stream = stream.shuffle(seed=args.data_seed, buffer_size=args.shuffle_buffer)
        pending = []
        def flush():
            if not pending:
                return
            ids = tok([t for _, t in pending], add_special_tokens=False,
                      truncation=False, return_attention_mask=False)["input_ids"]
            for (split, _), seq in zip(pending, ids):
                room = limits[split] - counts[split]
                if room <= 0:
                    continue
                seq = (seq + [tok.eos_token_id])[:room]
                np.asarray(seq, dtype=dtype).tofile(files[split])
                counts[split] += len(seq)
                docs[split] += 1
            pending.clear()
        for row in stream:
            text = row[args.text_column]
            if not isinstance(text, str) or not text.strip():
                continue
            split = designated_split or ("val" if validation_document(text, args.data_seed) else "train")
            if counts[split] < limits[split]:
                pending.append((split, text))
            if len(pending) >= 32:
                flush()
                if sum(docs.values()) % 1024 < 32:
                    print(f"prepared train={counts['train']:,} val={counts['val']:,}", flush=True)
            done = (counts[designated_split] >= limits[designated_split] if designated_split
                    else all(counts[s] >= limits[s] for s in counts))
            if done:
                break
        flush()
    try:
        if args.validation_split:
            consume(load_dataset(**dataset_kwargs, split=args.train_split), "train")
            consume(load_dataset(**dataset_kwargs, split=args.validation_split), "val")
        else:
            consume(load_dataset(**dataset_kwargs, split=args.train_split))
    finally:
        for f in files.values():
            f.close()
    if min(counts.values()) < 2:
        raise ValueError("Empty/too-small train or validation set. Check split and token budgets.")
    meta = dict(dataset=args.dataset, dataset_config=args.dataset_config,
                dataset_revision=ds_revision, tokenizer=args.tokenizer,
                tokenizer_revision=tok_revision, vocab_size=len(tok), eos_id=tok.eos_token_id,
                dtype=dtype, counts=counts, documents=docs, data_seed=args.data_seed,
                train_split=args.train_split, validation_split=args.validation_split,
                split_method="native" if args.validation_split else "document_hash_1_percent",
                shuffle_buffer=args.shuffle_buffer, synthetic=False,
                sha256={s: sha256(tmp/f"{s}.bin") for s in counts})
    (tmp/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    tmp.rename(out)
    print(json.dumps(meta, indent=2))


def synthetic_data(out, seed=17):
    """Offline plumbing test only. Not OpenWebText, not evidence about nGPT quality."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    counts = {"train": 65536, "val": 8192}
    for split, n in counts.items():
        arr = np.empty(n, dtype="<u2")
        arr[0] = rng.integers(0, 32)
        for i in range(1, n):
            arr[i] = (int(arr[i-1])+1) % 32 if rng.random() > .05 else rng.integers(0, 32)
        arr.tofile(out/f"{split}.bin")
    meta = dict(dataset="SYNTHETIC noisy cyclic transition test", synthetic=True,
                vocab_size=32, dtype="<u2", counts=counts,
                sha256={s: sha256(out/f"{s}.bin") for s in counts})
    (out/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    return meta


def load_tokens(path, context):
    path = Path(path)
    meta = json.loads((path/"metadata.json").read_text())
    arrays = {}
    for split in ("train", "val"):
        file = path/f"{split}.bin"
        if sha256(file) != meta["sha256"][split]:
            raise ValueError(f"Token checksum mismatch: {file}")
        arr = np.memmap(file, mode="r", dtype=meta["dtype"])
        if len(arr) <= context:
            raise ValueError(f"{split} has {len(arr)} tokens; need at least context+1.")
        arrays[split] = arr
    return meta, arrays


def starts_for_step(n_tokens, context, batch_size, seed, step, micro=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro]))
    return rng.integers(0, n_tokens-context, size=batch_size)


def token_batch(array, starts, context, device):
    import torch
    batch = np.stack([array[int(i):int(i)+context+1] for i in starts]).astype(np.int64)
    return torch.from_numpy(batch).to(device)

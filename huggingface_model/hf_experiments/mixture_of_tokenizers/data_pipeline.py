"""Disk-cached, document-bounded sampling for the five tokenizer routes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LANGUAGES = ("zh", "ko", "ja", "en")
ROUTES = ("byte",) + LANGUAGES


def route_index(route: str | int) -> int:
    if isinstance(route, str):
        return ROUTES.index(route)
    route = int(route)
    if not 0 <= route < len(ROUTES):
        raise ValueError(f"Invalid route {route}")
    return route


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def iter_documents(path: Path):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{index + 1}: missing string text")
            yield row


class LanguageWindowDataset:
    """One route group at a time, with deterministic resumable window shuffles.

    ``get_batch`` returns ordinary model tensors plus ``batch_stats`` (remove
    that entry before model.forward). Labels already ignore position zero and
    padding; the causal model still performs the normal one-position shift.
    Every document is conceptualized as BOS + text + EOS. Adjacent windows
    overlap by exactly one input token so every target is predicted once per
    epoch; EOS occurs only at a real document end. No window crosses documents.
    """

    def __init__(self, data_dir, tokenizer, context_length, split="train", seed=1337, cache_dir=None):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        if self.context_length < 3:
            raise ValueError("context_length must be at least 3")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.split, self.seed = split, int(seed)
        self.fingerprint = tokenizer.fingerprint
        self.cache_dir = Path(cache_dir or self.data_dir.parent / "token_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pairs = {}
        self._byte_cursor = 0
        self._payload_lengths = np.array([len(tokenizer.token_byte_payload(i))
                                          for i in range(tokenizer.vocab_size)], dtype=np.int32)
        # Count UTF-8 leading bytes. A fallback byte window can cut a code point,
        # but each original character is counted once when its lead byte is a
        # predicted target, and totals remain exact across the full document.
        self._payload_characters = np.array([
            sum((byte & 0xC0) != 0x80 for byte in tokenizer.token_byte_payload(i))
            for i in range(tokenizer.vocab_size)], dtype=np.int32)
        self._source_digests = {}
        for language in LANGUAGES:
            path = self.data_dir / language / f"{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            # Hash once at startup: cache reuse is never based on filename alone.
            self._source_digests[language] = file_sha256(path)

    def iter_texts(self, language, limit=None):
        if language not in LANGUAGES:
            raise ValueError(f"Unknown language {language}")
        for index, row in enumerate(iter_documents(self.data_dir / language / f"{self.split}.jsonl")):
            if limit is not None and index >= limit:
                break
            yield row["text"]

    def _cache(self, route: int, language: str):
        pair_key = f"{route}:{language}"
        if pair_key in self._pairs:
            return self._pairs[pair_key]
        cache_identity = {
            "format": 2, "source_sha256": self._source_digests[language],
            "tokenizer": "raw-utf8-v1" if route == 0 else self.fingerprint,
            "dtype": "uint8" if route == 0 else "<u4",
            "route": route, "language": language,
        }
        digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        folder = self.cache_dir / digest
        folder.mkdir(exist_ok=True)
        marker = folder / "complete.json"
        if not marker.exists():
            offsets = [0]
            token_path = folder / "tokens.bin.tmp"
            source = self.data_dir / language / f"{self.split}.jsonl"
            with token_path.open("wb") as handle:
                for record in iter_documents(source):
                    if route == 0:
                        # Raw content byte IDs do not depend on K or on route
                        # controls. Store once, one byte per byte, for all sweeps.
                        values = np.frombuffer(record["text"].encode("utf-8"), dtype=np.uint8)
                    else:
                        ids = self.tokenizer.encode_route(record["text"], route)
                        values = np.asarray(ids, dtype=np.int64)
                        if len(values) and (values.min() < 0 or values.max() >= self.tokenizer.vocab_size):
                            raise ValueError("Tokenizer emitted out-of-range IDs")
                        values = values.astype("<u4")
                    if not len(values):
                        continue
                    values.tofile(handle)
                    offsets.append(offsets[-1] + len(values))
            if len(offsets) == 1:
                token_path.unlink(missing_ok=True)
                raise ValueError(f"No nonempty documents in {source}")
            token_path.replace(folder / "tokens.bin")
            np.save(folder / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
            temporary_marker = folder / "complete.json.tmp"
            temporary_marker.write_text(json.dumps(cache_identity, indent=2) + "\n")
            temporary_marker.replace(marker)
        elif json.loads(marker.read_text()) != cache_identity:
            raise ValueError(f"Corrupt token cache metadata in {folder}")
        offsets = np.load(folder / "offsets.npy", mmap_mode="r")
        tokens = np.memmap(folder / "tokens.bin", mode="r", dtype=cache_identity["dtype"])
        if int(offsets[-1]) != len(tokens):
            raise ValueError(f"Truncated token cache in {folder}")
        lengths = np.diff(offsets).astype(np.int64)
        # Predict n text tokens plus EOS from a sequence of n+2 input entries.
        windows_per_doc = (lengths + 1 + self.context_length - 2) // (self.context_length - 1)
        cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(windows_per_doc)))
        pair = {"tokens": tokens, "offsets": offsets, "cumulative": cumulative,
                "total_windows": int(cumulative[-1]), "epoch": 0, "cursor": 0,
                "order": None, "digest": digest}
        self._pairs[pair_key] = pair
        self._set_order(pair_key, pair)
        return pair

    def _set_order(self, key, pair):
        material = f"{self.seed}:{key}:{pair['epoch']}".encode()
        rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        pair["order"] = np.random.default_rng(rng_seed).permutation(pair["total_windows"])

    def _next_window(self, route: int, language: str):
        pair = self._cache(route, language)
        key = f"{route}:{language}"
        if pair["cursor"] >= pair["total_windows"]:
            pair["cursor"] = 0
            pair["epoch"] += 1
            self._set_order(key, pair)
        index = int(pair["order"][pair["cursor"]])
        pair["cursor"] += 1
        document = int(np.searchsorted(pair["cumulative"], index, side="right") - 1)
        window = index - int(pair["cumulative"][document])
        start, end = map(int, pair["offsets"][document:document + 2])
        content_length = end - start
        position = window * (self.context_length - 1)
        stop = min(position + self.context_length, content_length + 2)
        values = []
        if position == 0:
            values.append(self.tokenizer.bos_id(route))
        content_start = max(0, position - 1)
        content_end = min(content_length, stop - 1)
        values.extend(map(int, pair["tokens"][start + content_start:start + content_end]))
        if stop == content_length + 2:
            values.append(self.tokenizer.eos_id(route))
        if not 2 <= len(values) <= self.context_length:
            raise AssertionError("Window construction did not produce any prediction targets")
        return values, document, pair["epoch"]

    def get_batch(self, route, batch_size, device="cpu"):
        route = route_index(route)
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        inputs, masks, labels = [], [], []
        stats = {"route": ROUTES[route], "sequences": batch_size, "predicted_tokens": 0,
                 "text_tokens": 0, "text_bytes": 0, "fallback_tokens": 0,
                 "source_bytes": 0, "source_characters": 0,
                 "input_text_bytes": 0, "eos_targets": 0, "languages": {}}
        for _ in range(batch_size):
            if route == 0:
                language = LANGUAGES[self._byte_cursor % 4]
                self._byte_cursor += 1
            else:
                language = ROUTES[route]
            values, document, epoch = self._next_window(route, language)
            valid = len(values)
            predicted = np.asarray(values[1:], dtype=np.int64)
            text_mask = predicted < 256 + 4 * self.tokenizer.lexical_size
            text_ids = predicted[text_mask]
            text_bytes = int(self._payload_lengths[text_ids].sum())
            eos_targets = int(np.count_nonzero(predicted == self.tokenizer.eos_id(route)))
            stats["predicted_tokens"] += valid - 1
            stats["text_tokens"] += len(text_ids)
            stats["text_bytes"] += text_bytes
            stats["source_bytes"] += text_bytes
            stats["source_characters"] += int(self._payload_characters[text_ids].sum())
            stats["input_text_bytes"] += int(self._payload_lengths[values].sum())
            stats["fallback_tokens"] += int(np.count_nonzero(text_ids < 256))
            stats["eos_targets"] += eos_targets
            lang_stats = stats["languages"].setdefault(language, {"sequences": 0, "predicted_tokens": 0, "text_bytes": 0})
            lang_stats["sequences"] += 1
            lang_stats["predicted_tokens"] += valid - 1
            lang_stats["text_bytes"] += text_bytes
            padding = self.context_length - valid
            inputs.append(values + [self.tokenizer.pad_id(route)] * padding)
            masks.append([1] * valid + [0] * padding)
            labels.append([-100] + values[1:] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "tokenizer_ids": torch.full((batch_size,), route, dtype=torch.long, device=device),
            "batch_stats": stats,
        }

    def state_dict(self):
        return {"format": 1, "seed": self.seed, "split": self.split,
                "context_length": self.context_length, "fingerprint": self.fingerprint,
                "source_digests": self._source_digests, "byte_cursor": self._byte_cursor,
                "pairs": {key: {"epoch": p["epoch"], "cursor": p["cursor"], "digest": p["digest"]}
                          for key, p in self._pairs.items()}}

    def load_state_dict(self, state):
        for key in ("seed", "split", "context_length", "fingerprint", "source_digests"):
            expected = self.state_dict()[key]
            if state.get(key) != expected:
                raise ValueError(f"Sampler resume mismatch for {key}")
        self._byte_cursor = int(state["byte_cursor"])
        # Restoring a state captured before first use must also rewind caches
        # materialized later (for example fixed validation batches).
        for key, pair in self._pairs.items():
            pair.update(epoch=0, cursor=0)
            self._set_order(key, pair)
        for key, progress in state["pairs"].items():
            route, language = key.split(":")
            pair = self._cache(int(route), language)
            if progress["digest"] != pair["digest"]:
                raise ValueError(f"Sampler cache changed for {key}")
            if not 0 <= progress["cursor"] <= pair["total_windows"] or progress["epoch"] < 0:
                raise ValueError(f"Invalid sampler progress for {key}")
            pair.update(epoch=int(progress["epoch"]), cursor=int(progress["cursor"]))
            self._set_order(key, pair)

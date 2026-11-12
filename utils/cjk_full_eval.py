"""Deterministic, bounded-memory CJK evaluation without a Trainer or optimizer.

Only trusted local checkpoints/metadata should be loaded: both use pickle.
"""

from contextlib import nullcontext
from dataclasses import dataclass, fields
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1
LANGUAGES = ("ja", "ko", "zh_cn")
RUN_NAMES = tuple(
    f"{lang}_{representation}_{tokenizer}"
    for lang in LANGUAGES for representation in ("original", "ipa")
    for tokenizer in ("char", "char_bpe")
)
METRICS = ("loss", "top1_prob", "top1_correct", "target_rank", "target_prob",
           "target_left_prob", "ln_f_cosine")


def load_meta(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_identity(identity):
    path = Path(identity["path"])
    if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
        raise ValueError(f"Artifact changed since preparation: {path}")
    return path


def write_json(path, value):
    """Atomic, standards-compliant JSON; unavailable measurements are null."""
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if math.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def token_bytes(meta):
    """Raw bytes per ID, including individual fallback bytes (never U+FFFD)."""
    result = []
    for index in range(int(meta["vocab_size"])):
        value = meta["itos"][index]
        result.append(value if isinstance(value, bytes) else value.encode("utf-8"))
    if any(not value for value in result):
        raise ValueError("Empty vocabulary entry")
    return result


class FrozenTokenizer:
    """The experiment's Char / longest-match CharBPE rules, with fixed IDs."""

    def __init__(self, meta):
        self.meta = meta
        self.stoi = meta["stoi"]
        self.pieces = token_bytes(meta)
        self.is_bpe = meta.get("tokenizer") == "char_bpe"
        if not self.is_bpe and "chars" not in meta:
            raise ValueError("Only experiment Char and CharBPE metadata are supported")
        self.trie = {}
        if self.is_bpe:
            # Preserve the existing longest-string semantics, but avoid scanning
            # all 8192 vocabulary entries at each text position.
            for value, index in self.stoi.items():
                if not isinstance(value, str):
                    continue
                node = self.trie
                for char in value:
                    node = node.setdefault(char, {})
                node[None] = int(index)

    def unknown_characters(self, text):
        if self.is_bpe:
            return {}
        from collections import Counter
        return dict(Counter(char for char in text if char not in self.stoi))

    def encode(self, text):
        if not self.is_bpe:
            return [self.stoi[char] for char in text]
        result, position = [], 0
        while position < len(text):
            node, cursor, best = self.trie, position, None
            while cursor < len(text) and text[cursor] in node:
                node = node[text[cursor]]
                cursor += 1
                if None in node:
                    best = cursor, node[None]
            if best is not None:
                position, index = best
                result.append(index)
            else:
                result.extend(self.stoi[bytes([byte])] for byte in text[position].encode("utf-8"))
                position += 1
        return result

    def decode_bytes(self, ids):
        return b"".join(self.pieces[int(index)] for index in ids)


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    score_from: int


def iter_windows(token_count, block_size=1024, stride=512):
    """Input [start:end], labels [start+1:end+1]; score each label once.

    All positions before score_from are context-only. No padding or BOS is
    invented. The first source token has no preceding context and is excluded.
    """
    if not 1 <= stride <= block_size:
        raise ValueError("Require 1 <= stride <= block_size")
    covered = 0
    while covered < token_count - 1:
        end = min(block_size if covered == 0 else covered + stride, token_count - 1)
        start = max(0, end - block_size)
        yield Window(start, end, covered - start)
        covered = end


def histogram_quantile(histogram, probability):
    """Linear quantile, matching torch.quantile's default interpolation."""
    size = int(histogram.sum())
    if not size:
        return math.nan
    index = (size - 1) * probability
    cumulative = np.cumsum(histogram)
    lo = int(np.searchsorted(cumulative, math.floor(index), side="right"))
    hi = int(np.searchsorted(cumulative, math.ceil(index), side="right"))
    return lo + (hi - lo) * (index - math.floor(index))


class MetricAccumulator:
    """FP32 position metrics, FP64 sums and disk-backed exact quantiles.

    Definitions follow train.py and PerTokenMetrics. left_prob_95 includes the
    target's own mass; ln_f_cosine_95 is the historical *5th* percentile.
    """

    def __init__(self, vocab_size, capacity, scratch_dir):
        self.vocab_size = vocab_size
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.sums = {name: np.zeros(vocab_size, dtype=np.float64) for name in METRICS}
        self.rank_histogram = np.zeros(vocab_size + 1, dtype=np.int64)
        self.capacity, self.count = capacity, 0
        self.temporary = tempfile.TemporaryDirectory(dir=scratch_dir, prefix="quantiles-")
        self.quantiles = {
            name: np.memmap(Path(self.temporary.name) / f"{name}.f32", mode="w+",
                            dtype=np.float32, shape=(max(1, capacity),))
            for name in ("left_inclusive", "cosine")
        }

    def add(self, logits, targets, hidden, output_weight, chunk_size=256):
        logits = logits.detach().reshape(-1, logits.shape[-1])
        targets = targets.detach().reshape(-1)
        hidden = hidden.detach().reshape(-1, hidden.shape[-1])
        if not (len(logits) == len(targets) == len(hidden)):
            raise ValueError("Logit/target/hidden position mismatch")
        for start in range(0, len(targets), chunk_size):
            ids = targets[start:start + chunk_size]
            valid = ids != -1
            ids = ids[valid].long()
            if not ids.numel():
                continue
            if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
                raise ValueError("Target outside checkpoint vocabulary")
            chunk = logits[start:start + chunk_size][valid].float()
            if chunk.shape[-1] != self.vocab_size or not bool(torch.isfinite(chunk).all()):
                raise ValueError("Invalid logits or vocabulary size")
            probabilities = F.softmax(chunk, dim=-1)
            target_logits = chunk.gather(1, ids[:, None])
            ahead = chunk > target_logits
            rank = ahead.sum(dim=-1) + 1
            target_prob = probabilities.gather(1, ids[:, None]).squeeze(1)
            left = probabilities.masked_fill(~ahead, 0).sum(dim=-1)
            top1_prob, top1_id = probabilities.max(dim=-1)
            cosine = F.cosine_similarity(hidden[start:start + chunk_size][valid].float(),
                                         output_weight[ids].float(), dim=-1)
            values = dict(loss=F.cross_entropy(chunk, ids, reduction="none"),
                          top1_prob=top1_prob, top1_correct=(top1_id == ids).float(),
                          target_rank=rank, target_prob=target_prob,
                          target_left_prob=left, ln_f_cosine=cosine)
            cpu_ids = ids.cpu().numpy()
            self.counts += np.bincount(cpu_ids, minlength=self.vocab_size)
            for name, value in values.items():
                self.sums[name] += np.bincount(cpu_ids, weights=value.double().cpu().numpy(),
                                              minlength=self.vocab_size)
            self.rank_histogram += np.bincount(rank.cpu().numpy(), minlength=self.vocab_size + 1)
            end = self.count + len(cpu_ids)
            if end > self.capacity:
                raise ValueError("More scored positions than the coverage plan")
            self.quantiles["left_inclusive"][self.count:end] = (left + target_prob).cpu().numpy()
            self.quantiles["cosine"][self.count:end] = cosine.cpu().numpy()
            self.count = end

    def summary(self, byte_lengths):
        if self.count != self.capacity:
            raise ValueError(f"Incomplete evaluation: {self.count} / {self.capacity} targets")
        scored_bytes = int(self.counts @ np.asarray(byte_lengths, dtype=np.int64))
        output = {name: float(values.sum() / self.count) if self.count else math.nan
                  for name, values in self.sums.items()}
        output.update(scored_tokens=self.count, scored_utf8_bytes=scored_bytes,
                      nll_sum=float(self.sums["loss"].sum()),
                      bits_per_byte=float(self.sums["loss"].sum() / math.log(2) / scored_bytes)
                      if scored_bytes else math.nan,
                      perplexity=math.exp(output["loss"]) if output["loss"] < 709 else math.inf,
                      target_rank_95=histogram_quantile(self.rank_histogram, 0.95))
        for name, probability, key in (("left_inclusive", 0.95, "left_prob_95"),
                                       ("cosine", 0.05, "ln_f_cosine_95")):
            output[key] = float(np.quantile(self.quantiles[name][:self.count], probability,
                                            overwrite_input=True)) if self.count else math.nan
        return output

    def close(self):
        self.quantiles.clear()
        self.temporary.cleanup()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path, device="cpu"):
    from gpt_conf import GPTConfig
    from model import GPT
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = checkpoint["config"]
    if config.get("multicontext") or config.get("multidataset_wte"):
        raise ValueError("This evaluator supports single-dataset CJK checkpoints only")
    arguments = checkpoint["model_args"]
    allowed = {field.name for field in fields(GPTConfig)}
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"Unsupported checkpoint model_args: {sorted(unknown)}")
    model = GPT(GPTConfig(**arguments))
    state = {name.removeprefix("_orig_mod."): value for name, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    if model.config.n_embd_wte:
        raise ValueError("Cosine definition requires an unfactored output embedding")
    model.to(device).eval().requires_grad_(False)
    info = {"iteration": int(checkpoint["iter_num"]), "config": config,
            "model_args": arguments, "checkpoint_metrics": checkpoint.get("metrics")}
    # Drop optimizer/scheduler state without constructing or restoring either.
    del checkpoint, state
    return model, info


def evaluate_stream(model, ids, meta, scratch_dir, device="cpu", dtype="float32",
                    stride=512, batch_size=4, iteration=0, progress=None):
    """Score one split as a newline-separated stream, never across split boundaries."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pieces = token_bytes(meta)
    if model.config.vocab_size != len(pieces):
        raise ValueError("Model/tokenizer vocabulary mismatch")
    if len(ids) and (int(np.min(ids)) < 0 or int(np.max(ids)) >= len(pieces)):
        raise ValueError("Input IDs outside checkpoint vocabulary")
    if str(device).startswith("cpu") and dtype != "float32":
        raise ValueError("CPU evaluation uses float32; CUDA supports mixed precision")
    accumulator = MetricAccumulator(len(pieces), max(0, len(ids) - 1), scratch_dir)
    model.eval().requires_grad_(False)
    before = state_fingerprint(model)
    hidden = []
    handle = model.transformer.ln_f.register_forward_hook(lambda _m, _i, output: hidden.append(output))
    block_size = model.config.block_size

    def run_batch(windows):
        length = max(window.end - window.start for window in windows)
        x = torch.zeros((len(windows), length), dtype=torch.long, device=device)
        y = torch.full_like(x, -1)
        for row, window in enumerate(windows):
            size = window.end - window.start
            x[row, :size] = torch.as_tensor(np.array(ids[window.start:window.end], dtype=np.int64), device=device)
            y[row, :size] = torch.as_tensor(np.array(ids[window.start+1:window.end+1], dtype=np.int64), device=device)
            y[row, :window.score_from] = -1
        hidden.clear()
        ctx = (torch.autocast("cuda", dtype=getattr(torch, dtype))
               if str(device).startswith("cuda") and dtype != "float32" else nullcontext())
        with ctx:
            logits, _ = model(x, y, iter_num=iteration)
        if len(hidden) != 1:
            raise ValueError("Expected exactly one final normalization output")
        accumulator.add(logits, y, hidden[0], model.lm_head.weight)
        hidden.clear()
        if progress:
            progress(accumulator.count, accumulator.capacity)

    try:
        with torch.inference_mode():
            batch = []
            for window in iter_windows(len(ids), block_size, stride):
                batch.append(window)
                if len(batch) == batch_size:
                    run_batch(batch)
                    batch = []
            if batch:
                run_batch(batch)
        if state_fingerprint(model) != before:
            raise RuntimeError("Model parameters/buffers changed during evaluation")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Unexpected gradients during evaluation")
        summary = accumulator.summary([len(piece) for piece in pieces])
        summary.update(source_tokens=len(ids), excluded_initial_tokens=min(1, len(ids)),
                       block_size=block_size, stride=stride, batch_size=batch_size,
                       dtype=dtype, model_state_unchanged=True)
        return summary, accumulator
    except BaseException:
        accumulator.close()
        raise
    finally:
        handle.remove()


def export_tokens(path, meta, accumulator, training_counts, geometry, iteration=0,
                  dataset="", evaluation_split="val"):
    fields = ("iteration", "dataset", "token_id", "token_text_escaped", "vector_magnitude", "min_pairwise_angle_deg",
              "train_loss", "train_eval_count", "val_loss", "val_eval_count",
              "eval_count", "loss", "avg_target_probability", "avg_target_rank",
              "avg_left_probability", "training_seen_count")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(accumulator.vocab_size):
            count = int(accumulator.counts[index])
            mean = lambda key: float(accumulator.sums[key][index] / count) if count else math.nan
            row = dict(iteration=iteration, dataset=dataset,
                       token_id=index, token_text_escaped=repr(meta["itos"][index]),
                       train_loss=math.nan, train_eval_count=0, val_loss=math.nan, val_eval_count=0,
                       vector_magnitude=math.nan, min_pairwise_angle_deg=math.nan,
                       eval_count=count, loss=mean("loss"), avg_target_rank=mean("target_rank"),
                       avg_target_probability=mean("target_prob"),
                       avg_left_probability=mean("target_left_prob"),
                       training_seen_count=training_counts[index] if training_counts is not None else "")
            row.update({f"{evaluation_split}_loss": mean("loss"),
                        f"{evaluation_split}_eval_count": count})
            if geometry is not None:
                row.update(vector_magnitude=float(geometry["norms"][index]),
                           min_pairwise_angle_deg=float(geometry["min_angles"][index]))
            writer.writerow(row)

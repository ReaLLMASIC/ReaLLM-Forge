"""Shared configuration and atomic file utilities (Apache-2.0)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODES = ("full", "freeze_head", "freeze_embed", "freeze_both", "slow_head", "replay",
         "freeze_head_norm", "tied_full", "tied_freeze_both")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_json(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()


def load_config(path):
    c = read_json(path)
    defaults = dict(
        revision="main", device="cuda", memory_limit_gib=70.0, memory_fraction=0.90,
        activation_reserve_gib=10.0, seq_len=1024, micro_batch=1, accumulation=16,
        head_chunk=128, seeds=[42, 43, 44], lrs=[1e-5, 3e-5, 1e-4],
        modes=["full", "freeze_head", "freeze_embed", "freeze_both"],
        a_steps=500, b_steps=1500, a_lr=3e-5, weight_decay=0.01,
        warmup_fraction=0.05, min_lr_fraction=0.1, clip_grad=1.0,
        eval_every=100, save_every=100, log_every=10, probe_blocks=4,
        slow_head_multiplier=0.1, replay_fraction=0.1, gradient_checkpointing=True,
        attn_implementation="sdpa", deterministic=False, cpu_threads=4,
        freeze_sample_rows=1024, disable_dropout=True,
        retention_label="A-domain retention; A is not the original model's complete pretraining corpus",
    )
    for k, v in defaults.items():
        c.setdefault(k, v)
    for key in ("model", "output", "data"):
        if key not in c:
            raise ValueError(f"Missing configuration key: {key}")
    if not set(c["modes"]).issubset(MODES):
        raise ValueError(f"Supported modes: {MODES}")
    for k in ("seq_len", "micro_batch", "accumulation", "head_chunk", "b_steps", "eval_every", "save_every", "log_every"):
        if c[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if c["a_steps"] < 0 or not c["seeds"] or not c["lrs"]:
        raise ValueError("Need nonnegative A steps, at least one seed, and at least one LR")
    if any(x <= 0 for x in c["lrs"]) or c["a_lr"] <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 <= c["replay_fraction"] < 1:
        raise ValueError("replay_fraction must lie in [0, 1)")
    if not 0 < c["memory_fraction"] < 1 or c["memory_limit_gib"] <= 0:
        raise ValueError("Invalid memory limit")
    if not 0 <= c["warmup_fraction"] < 1 or not 0 <= c["min_lr_fraction"] <= 1:
        raise ValueError("Invalid learning-rate schedule")
    c["output"] = str(Path(c["output"]).resolve())
    if Path(c["model"]).is_dir():
        c["model"] = str(Path(c["model"]).resolve())
    return c


def run_name(mode, lr):
    return f"{mode}_lr{lr:.8g}"


def anchor_dir(c, seed):
    return Path(c["output"]) / f"seed_{seed}" / "anchor"


def run_dir(c, seed, mode, lr):
    return Path(c["output"]) / f"seed_{seed}" / run_name(mode, lr)


def check_identity(path, identity):
    path = Path(path)
    if path.exists() and read_json(path) != identity:
        raise RuntimeError(f"Configuration/source changed at {path}. Use a new output directory.")
    write_json(path, identity)

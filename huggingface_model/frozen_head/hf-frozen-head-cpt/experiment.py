"""Sequential, resumable A -> B experiments. No cloud training or Hub uploads."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import time
from pathlib import Path

import torch
import transformers

from common import (anchor_dir, append_json, check_identity, digest, file_sha,
                    read_json, run_dir, write_json)
from diagnostics import (data_blocks, evaluate, head_geometry, head_swap,
                         make_reference, rare_tokens)
from model_utils import (autocast, chunked_loss, configure_mode, frozen_hashes,
                         load_model, lr_factor, make_optimizer, peak_memory,
                         place_model, restore_rng, rng_state, seed_all, setup_device)
from prepare_data import BlockOrder, resolve_model


def latest_checkpoint(root):
    path = Path(root) / "latest.json"
    return Path(root) / read_json(path)["path"] if path.exists() else None


def save_checkpoint(root, model, optimizer, step, identity, total_train_seconds, extra_state=None):
    root = Path(root)
    dest = root / f"checkpoint-{step:08d}"
    if dest.exists():
        # Recover a crash between directory commit and pointer commit.
        write_json(root / "latest.json", {"path": dest.name, "step": step})
        return dest
    tmp = root / f"checkpoint-{step:08d}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    state = {"step": step, "identity": identity, "rng": rng_state(),
             "total_train_seconds": total_train_seconds}
    if extra_state is not None:
        state["extra_state"] = extra_state
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="2GB")
    torch.save(state, tmp / "trainer.pt")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    os.replace(tmp, dest)
    write_json(root / "latest.json", {"path": dest.name, "step": step})
    # The previous committed checkpoint stays available until the new pointer commits.
    for previous in root.glob("checkpoint-*"):
        if previous.is_dir() and previous != dest:
            shutil.rmtree(previous)
    return dest


def trim_log(path, step):
    path = Path(path)
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # interrupted trailing write
        if row["step"] <= step:
            rows.append(row)
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))


def choose_stream(micro_index, replay):
    old_count = math.floor(micro_index * replay + 1e-9)
    next_count = math.floor((micro_index + 1) * replay + 1e-9)
    return ("a", old_count) if next_count > old_count else ("b", micro_index - old_count)


def step_train(model, optimizer, datasets, orders, c, device, step, phase, replay, total):
    model.train()
    factor = lr_factor(step, total, c)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for micro in range(c["accumulation"]):
        micro_index = (step - 1) * c["accumulation"] + micro
        source, source_index = ("a", micro_index) if phase == "a" else choose_stream(micro_index, replay)
        indices = orders[source].indices(source_index * c["micro_batch"], c["micro_batch"])
        batch = datasets[source].batch(indices, device)
        with autocast(device):
            loss = chunked_loss(model, batch, c["head_chunk"]) / c["accumulation"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        running += loss.detach().item()
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], c["clip_grad"], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return running, float(grad_norm), factor


def environment(c, device):
    packages = {}
    for name in ("datasets", "huggingface_hub", "tokenizers", "safetensors", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "torch": torch.__version__, "packages": packages,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "config": c}


def train(c, seed, phase, mode="full", lr=None, stop_after=None):
    """stop_after is an interruption test, not a schedule change; resume uses the same total."""
    if phase == "a":
        mode, lr = "full", c["a_lr"]
    device, cap = setup_device(c)
    seed_all(seed, c["deterministic"])
    data_root = Path(c["output"]) / "data"
    manifest = read_json(data_root / "manifest.json")
    aroot = anchor_dir(c, seed)
    root = aroot if phase == "a" else run_dir(c, seed, mode, lr)
    root.mkdir(parents=True, exist_ok=True)
    identity = {"config": digest(c), "data": digest(manifest), "seed": seed,
                "phase": phase, "mode": mode, "lr": lr}
    check_identity(root / "identity.json", identity)
    if (root / "complete.json").exists():
        print(f"Already complete: {root}", flush=True)
        return 0
    previous = latest_checkpoint(root)
    state = None
    if previous:
        state = torch.load(previous / "trainer.pt", map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise RuntimeError("Checkpoint identity mismatch")
        model = load_model(str(previous), c)
    elif phase == "a":
        source = manifest["model"]
        for name, checksum in source.get("local_hashes", {}).items():
            if file_sha(Path(source["id"]) / name) != checksum:
                raise RuntimeError("Local source model changed after data preparation")
        model = load_model(source["id"], c, source["revision"])
    else:
        if not (aroot / "complete.json").exists():
            raise RuntimeError("Train the shared A anchor first")
        model = load_model(str(latest_checkpoint(aroot)), c)
    actual_mode = "anchor" if phase == "a" else mode
    tie_status = configure_mode(model, actual_mode)
    if c["seq_len"] > getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 10**9)):
        raise ValueError("seq_len exceeds configured positional context")
    if manifest["tokenizer_vocab_size"] > model.get_output_embeddings().out_features:
        raise ValueError("Tokenizer/model vocab mismatch; do not resize the vocabulary during this experiment")
    memory = place_model(model, c, device, cap, actual_mode)
    optimizer = make_optimizer(model, c, c["a_lr"] if phase == "a" else lr, actual_mode)
    start = state["step"] if state else 0
    total_train_seconds = state["total_train_seconds"] if state else 0.0
    if previous:
        opt = torch.load(previous / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(opt)
        del opt
        restore_rng(state["rng"])
    else:
        # Same B dropout RNG and same optimizer reset for every arm at this seed.
        seed_all(seed + (100000 if phase == "b" else 0), c["deterministic"])
    for name in ("metrics.jsonl", "train.jsonl"):
        trim_log(root / name, start)
    write_json(root / "environment.json", environment(c, device))
    write_json(root / "memory_estimate.json", {**memory, **tie_status})
    datasets = {domain: data_blocks(c, domain, "train") for domain in ("a", "b")}
    val = {domain: data_blocks(c, domain, "validation") for domain in ("a", "b")}
    orders = {domain: BlockOrder(len(datasets[domain]), seed, i) for i, domain in enumerate(("a", "b"))}
    rare = rare_tokens(data_root, model.get_output_embeddings().out_features, device)
    ref = torch.load(aroot / "reference.pt", map_location="cpu", weights_only=True) if phase == "b" else None
    frozen_start = frozen_hashes(model)
    if (root / "frozen_initial.json").exists():
        if read_json(root / "frozen_initial.json") != frozen_start:
            raise RuntimeError("A frozen parameter changed across checkpoint/resume")
    else:
        write_json(root / "frozen_initial.json", frozen_start)
    total = c["a_steps"] if phase == "a" else c["b_steps"]
    replay = c["replay_fraction"] if mode == "replay" and phase == "b" else 0.0

    def validation(step):
        result = {domain: evaluate(model, val[domain], c, device, rare) for domain in ("a", "b")}
        a_micro = step * c["accumulation"] if phase == "a" else math.floor(step * c["accumulation"] * replay + 1e-9)
        per_micro = c["micro_batch"] * c["seq_len"]
        row = {"step": step, "phase": phase, "seed": seed, "mode": actual_mode, "lr": lr,
               "a": result["a"], "b": result["b"], "train_a_tokens": a_micro * per_micro,
               "train_b_tokens": (step * c["accumulation"] - a_micro) * per_micro,
               "train_seconds": total_train_seconds, **peak_memory(device)}
        if ref is not None:
            row["head_swap"] = head_swap(model, val["a"], ref, c, device)
            row["head_geometry"] = head_geometry(model, ref, c["freeze_sample_rows"], seed)
        append_json(root / "metrics.jsonl", row)
        print(json.dumps({"run": root.name, "step": step, "a_loss": row["a"]["loss"],
                          "b_loss": row["b"]["loss"], **peak_memory(device)}), flush=True)
        return row

    metric_path = root / "metrics.jsonl"
    rows = [json.loads(x) for x in metric_path.read_text().splitlines()] if metric_path.exists() else []
    if not rows:
        baseline = validation(start)
    else:
        baseline = rows[0]
    already_hit_target = (phase == "b" and c.get("stop_at_validation_gain") is not None and rows
                          and baseline["b"]["loss"] - rows[-1]["b"]["loss"] >= c["stop_at_validation_gain"])
    if start == 0 and previous is None:
        save_checkpoint(root, model, optimizer, 0, identity, total_train_seconds)
    interrupted = {"stop": False}
    old_handlers = {}
    def request_stop(signum, frame):
        interrupted["stop"] = True
        print("Stopping after this optimizer step; writing a resumable checkpoint.", flush=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, request_stop)
    completed_step = start
    try:
        for step in range(start + 1, (start if already_hit_target else total) + 1):
            t0 = time.perf_counter()
            loss, norm, factor = step_train(model, optimizer, datasets, orders, c, device, step, phase, replay, total)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - t0
            total_train_seconds += duration
            completed_step = step
            if step % c["log_every"] == 0 or step == 1:
                row = {"step": step, "loss": loss, "grad_norm": norm, "lr_factor": factor,
                       "step_seconds": duration, "train_seconds": total_train_seconds,
                       "target_tokens_per_second": c["accumulation"] * c["micro_batch"] * c["seq_len"] / duration,
                       **peak_memory(device)}
                append_json(root / "train.jsonl", row)
                print(f"{root.name}: step {step}/{total}, loss={loss:.5f}, {row['target_tokens_per_second']:.0f} tokens/s", flush=True)
            due_eval = step % c["eval_every"] == 0 or step in {1, 2, 5, 10, 20, 50} or step == total
            row = validation(step) if due_eval else None
            target_hit = (phase == "b" and c.get("stop_at_validation_gain") is not None and row is not None
                          and baseline["b"]["loss"] - row["b"]["loss"] >= c["stop_at_validation_gain"])
            stopping = interrupted["stop"] or (stop_after is not None and step >= stop_after)
            if step % c["save_every"] == 0 or due_eval or stopping or target_hit or step == total:
                save_checkpoint(root, model, optimizer, step, identity, total_train_seconds)
            if stopping and step < total and not target_hit:
                return 75
            if target_hit:
                break
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if completed_step == 0:
        save_checkpoint(root, model, optimizer, 0, identity, total_train_seconds)
    final_hash = frozen_hashes(model)
    write_json(root / "frozen_final.json", final_hash)
    if final_hash != frozen_start:
        raise AssertionError("Frozen tensors changed during training")
    # Test data is evaluated at the end only, never used for checkpoint or LR selection.
    test = {domain: evaluate(model, data_blocks(c, domain, "test"), c, device, rare) for domain in ("a", "b")}
    write_json(root / "test.json", {"step": completed_step, "a": test["a"], "b": test["b"]})
    if phase == "a":
        make_reference(model, val["a"], c, device, aroot / "reference.pt")
    # Completed runs need weights for re-evaluation; optimizer state is only needed to resume incomplete runs.
    write_json(root / "complete.json", {"step": completed_step, "planned_steps": total,
               "train_seconds": total_train_seconds, "frozen_unchanged": True, **peak_memory(device)})
    if phase == "b" and not c.get("keep_completed_optimizer", False):
        (latest_checkpoint(root) / "optimizer.pt").unlink(missing_ok=True)
    return 0


def preflight(c):
    """Two real optimizer steps on throwaway synthetic inputs, before downloading corpora."""
    device, cap = setup_device(c)
    seed_all(c["seeds"][0], c["deterministic"])
    manifest_path = Path(c["output"]) / "data" / "manifest.json"
    source = read_json(manifest_path)["model"] if manifest_path.exists() else resolve_model(c)
    model = load_model(source["id"], c, source["revision"])
    configure_mode(model, "full")
    estimate = place_model(model, c, device, cap, "full")
    original_head = model.get_output_embeddings().weight.detach().cpu().clone()
    original_bias = model.get_output_embeddings().bias
    if original_bias is not None:
        original_bias = original_bias.detach().cpu().clone()
    optimizer = make_optimizer(model, c, c["lrs"][0], "full")
    model.train()
    batch = torch.randint(0, model.get_output_embeddings().out_features,
                          (c["micro_batch"], c["seq_len"] + 1), device=device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Two microsteps exercise accumulation buffers; number of accumulation steps does not change their size.
        for _ in range(min(c["accumulation"], 2)):
            with autocast(device):
                loss = chunked_loss(model, batch, c["head_chunk"]) / min(c["accumulation"], 2)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip_grad"], error_if_nonfinite=True)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    from model_utils import hidden
    from diagnostics import score_states
    model.eval()
    with torch.no_grad(), autocast(device):
        h = hidden(model, batch)
        score_states(h, batch[:, 1:], original_head.to(device),
                     original_bias.to(device) if original_bias is not None else None, c["head_chunk"])
    result = {"source": source, "estimate": estimate, "measured": peak_memory(device),
              "config": c, "scope": "Two throwaway synthetic optimizer steps and original-head evaluation; not a research result."}
    write_json(Path(c["output"]) / "preflight.json", result)
    print(json.dumps(result["measured"], indent=2), flush=True)

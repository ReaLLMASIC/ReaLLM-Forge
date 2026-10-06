#!/usr/bin/env python3
"""Task-specific supervised fine-tuning and retention experiments on one GPU."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import append_json, check_identity, digest, load_config, read_json, run_dir, write_json
from diagnostics import data_blocks, evaluate, head_geometry, head_swap, make_reference
from experiment import choose_stream, environment, latest_checkpoint, save_checkpoint, trim_log
from model_utils import (autocast, configure_mode, frozen_hashes, load_model, lr_factor, make_optimizer,
                         peak_memory, place_model, restore_rng, seed_all, setup_device)
from prepare_data import BlockOrder
from task_data import collate_examples, prepare_task, task_examples, task_manifest
from task_metrics import evaluate_math, response_loss_sum


def baseline_dir(c):
    return Path(c["output"]) / "baseline"


def task_identity(c):
    return {"config": digest(c), "data": digest(read_json(Path(c["output"]) / "data" / "manifest.json")),
            "task_data": digest(task_manifest(c))}


def task_validation(model, tokenizer, examples, c, device, split, output):
    return {"math": evaluate_math(model, tokenizer, examples, c, device, output),
            "retention": evaluate(model, data_blocks(c, "a", split), c, device)}


def baseline(c):
    root = baseline_dir(c)
    root.mkdir(parents=True, exist_ok=True)
    check_identity(root / "identity.json", task_identity(c))
    if (root / "complete.json").exists():
        return
    device, cap = setup_device(c)
    seed_all(c["seeds"][0], c["deterministic"])
    source = read_json(Path(c["output"]) / "data" / "manifest.json")["model"]
    model = load_model(source["id"], c, source["revision"])
    tokenizer = AutoTokenizer.from_pretrained(Path(c["output"]) / "data" / "tokenizer")
    # No A consolidation: these are the untouched pretrained model's task/general capabilities.
    place_model(model, c, device, cap, "anchor")
    model.save_pretrained(root / "model", safe_serialization=True, max_shard_size="2GB")
    for split in ("validation", "test"):
        result = task_validation(model, tokenizer, task_examples(c, split), c, device, split,
                                 root / f"{split}_predictions.jsonl")
        write_json(root / f"{split}.json", result)
    make_reference(model, data_blocks(c, "a", "validation"), c, device, root / "reference.pt")
    write_json(root / "environment.json", environment(c, device))
    write_json(root / "complete.json", {"trained": False, **peak_memory(device)})


def microbatches(c, rows, order, replay_blocks, replay_order, step, mode, pad_id):
    batches, counts = [], {"task_examples": 0, "response_tokens": 0, "replay_tokens": 0, "input_tokens": 0}
    replay = c["replay_fraction"] if mode == "replay" else 0.0
    for micro in range(c["accumulation"]):
        index = (step - 1) * c["accumulation"] + micro
        source, pos = choose_stream(index, replay)
        if source == "a":
            ids = replay_blocks.batch(replay_order.indices(pos * c["micro_batch"], c["micro_batch"]), "cpu")
            batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
            counts["replay_tokens"] += ids[:, 1:].numel()
        else:
            indices = order.indices(pos * c["micro_batch"], c["micro_batch"])
            batch = collate_examples([rows[i] for i in indices], pad_id)
            counts["task_examples"] += len(indices)
            counts["response_tokens"] += int((batch["labels"][:, 1:] != -100).sum())
        counts["input_tokens"] += int(batch["attention_mask"].sum())
        batches.append(batch)
    return batches, counts


def train_task(c, seed, mode, lr, stop_after=None):
    if mode not in c["modes"]:
        raise ValueError("mode is not in this experiment's configured conditions")
    root, initial = run_dir(c, seed, mode, lr), baseline_dir(c)
    if not (initial / "complete.json").exists():
        raise RuntimeError("Run the common pretrained baseline first")
    root.mkdir(parents=True, exist_ok=True)
    identity = {**task_identity(c), "seed": seed, "mode": mode, "lr": lr}
    check_identity(root / "identity.json", identity)
    if (root / "complete.json").exists():
        return 0
    device, cap = setup_device(c)
    seed_all(seed, c["deterministic"])
    previous = latest_checkpoint(root)
    state = torch.load(previous / "trainer.pt", map_location="cpu", weights_only=False) if previous else None
    if state is not None and state["identity"] != identity:
        raise RuntimeError("Checkpoint identity mismatch")
    model = load_model(str(previous or initial / "model"), c)
    tying = configure_mode(model, mode)
    estimate = place_model(model, c, device, cap, mode)
    optimizer = make_optimizer(model, c, lr, mode)
    if previous:
        opt_state = torch.load(previous / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(opt_state)
        del opt_state
        restore_rng(state["rng"])
    else:
        seed_all(seed + 100000, c["deterministic"])
    tokenizer = AutoTokenizer.from_pretrained(Path(c["output"]) / "data" / "tokenizer")
    rows, val = task_examples(c, "train"), task_examples(c, "validation")
    order = BlockOrder(len(rows), seed, 8)
    replay_blocks = data_blocks(c, "a", "train")
    replay_order = BlockOrder(len(replay_blocks), seed, 0)
    ref = torch.load(initial / "reference.pt", map_location="cpu", weights_only=True)
    frozen = frozen_hashes(model)
    check_identity(root / "frozen_initial.json", frozen)
    write_json(root / "memory_estimate.json", {**estimate, **tying})
    write_json(root / "environment.json", environment(c, device))
    start = state["step"] if state else 0
    seconds = state["total_train_seconds"] if state else 0.0
    counts = state["extra_state"] if state else {"task_examples": 0, "response_tokens": 0, "replay_tokens": 0, "input_tokens": 0}
    for name in ("metrics.jsonl", "train.jsonl"):
        trim_log(root / name, start)
    before = read_json(initial / "validation.json")

    def validation(step):
        score = task_validation(model, tokenizer, val, c, device, "validation", root / f"validation_predictions_{step}.jsonl")
        entry = {"step": step, "seed": seed, "mode": mode, "lr": lr, **score, **counts,
                 "math_gain_pp": 100 * (score["math"]["relaxed_accuracy"] - before["math"]["relaxed_accuracy"]),
                 "retention_loss_increase": score["retention"]["loss"] - before["retention"]["loss"],
                 "head_geometry": head_geometry(model, ref, c["freeze_sample_rows"], seed),
                 "head_swap": head_swap(model, data_blocks(c, "a", "validation"), ref, c, device),
                 "train_seconds": seconds, **peak_memory(device)}
        append_json(root / "metrics.jsonl", entry)
        print(f"{root.name}: step={step}, math={100*score['math']['relaxed_accuracy']:.1f}%, "
              f"old loss increase={entry['retention_loss_increase']:.4f}", flush=True)
        return entry

    def save(step):
        save_checkpoint(root, model, optimizer, step, identity, seconds, counts)

    if previous is None:
        validation(0)
        save(0)
    metric_rows = [json.loads(x) for x in (root / "metrics.jsonl").read_text().splitlines()]
    target = c["sft"].get("stop_at_validation_accuracy")
    if target is not None and target <= before["math"]["relaxed_accuracy"]:
        raise ValueError("The requested task-accuracy target is already achieved by the pretrained baseline. Choose a higher target.")
    already_reached = target is not None and metric_rows[-1]["math"]["relaxed_accuracy"] >= target
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    final_step = start
    try:
        for step in range(start + 1, (start if already_reached else c["b_steps"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            factor = lr_factor(step, c["b_steps"], c)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            batches, increment = microbatches(c, rows, order, replay_blocks, replay_order, step, mode, tokenizer.eos_token_id)
            # Token-weighted average over ALL microbatches; avoids bias from variable answer lengths.
            denominator = increment["response_tokens"] + increment["replay_tokens"]
            t0, train_loss = time.perf_counter(), 0.0
            for cpu_batch in batches:
                batch = {k: v.to(device) for k, v in cpu_batch.items()}
                with autocast(device):
                    summed, n = response_loss_sum(model, batch, c["head_chunk"])
                    loss = summed / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite SFT loss")
                train_loss += loss.item()
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], c["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - t0
            for k, value in increment.items():
                counts[k] += value
            final_step = step
            if step % c["log_every"] == 0 or step == 1:
                append_json(root / "train.jsonl", {"step": step, "loss": train_loss, "grad_norm": float(norm),
                                                   "train_seconds": seconds, **counts, **peak_memory(device)})
            due_eval = step % c["eval_every"] == 0 or step == c["b_steps"]
            score = validation(step) if due_eval else None
            hit = target is not None and score is not None and score["math"]["relaxed_accuracy"] >= target
            interrupted = stop["requested"] or (stop_after is not None and step >= stop_after)
            if due_eval or step % c["save_every"] == 0 or interrupted or hit:
                save(step)
            if interrupted and step < c["b_steps"] and not hit:
                return 75
            if hit:
                break
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    save(final_step)
    final_hash = frozen_hashes(model)
    if frozen != final_hash:
        raise AssertionError("Frozen tensors changed")
    write_json(root / "frozen_final.json", final_hash)
    test = task_validation(model, tokenizer, task_examples(c, "test"), c, device, "test", root / "test_predictions.jsonl")
    write_json(root / "test.json", {"step": final_step, **test})
    write_json(root / "complete.json", {"step": final_step, "train_seconds": seconds,
               "frozen_unchanged": True, **counts, **peak_memory(device)})
    if not c.get("keep_completed_optimizer", False):
        (latest_checkpoint(root) / "optimizer.pt").unlink(missing_ok=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "baseline", "train", "run", "report"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    c = load_config(args.config)
    if "sft" not in c:
        raise ValueError("Use a task_sft config containing an sft section")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.command == "prepare":
        prepare_task(c)
    elif args.command == "baseline":
        baseline(c)
    elif args.command == "train":
        return train_task(c, args.seed if args.seed is not None else c["seeds"][0], args.mode,
                          args.lr if args.lr is not None else c["lrs"][0], args.stop_after)
    elif args.command == "run":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Use one Python process, not torchrun")
        prepare_task(c)
        base = [sys.executable, str(Path(__file__).resolve())]
        commands = [["baseline", "--config", args.config]]
        commands += [["train", "--config", args.config, "--seed", str(seed), "--mode", mode, "--lr", str(lr)]
                     for seed in c["seeds"] for lr in c["lrs"] for mode in c["modes"]]
        for command in commands:
            result = subprocess.run(base + command)
            if result.returncode:
                return result.returncode
        from task_report import report_task
        report_task(c)
    else:
        from task_report import report_task
        report_task(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())

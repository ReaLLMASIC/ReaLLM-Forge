#!/usr/bin/env python3
"""Run standard EleutherAI harness tasks on the pretrained baseline and SFT checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path

from common import check_identity, digest, load_config, read_json, run_dir, write_json
from experiment import latest_checkpoint
from model_utils import setup_device
from task_sft import baseline_dir

DEFAULT_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq"]
METRICS = {"hellaswag": "acc_norm,none", "arc_easy": "acc_norm,none",
           "arc_challenge": "acc_norm,none", "piqa": "acc_norm,none", "boolq": "acc,none"}


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(x) for x in value]
    if hasattr(value, "tolist"):
        return sanitize(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def benchmark_root(c, limit):
    return Path(c["output"]) / "benchmarks" / (f"limit_{limit}" if limit else "full")


def task_code_hash(package_root):
    h = hashlib.sha256()
    for path in sorted((Path(package_root) / "tasks").rglob("*")):
        if path.is_file() and path.suffix in {".yaml", ".py"}:
            h.update(str(path.relative_to(package_root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def sample_fingerprints(samples):
    # Compare the actual evaluated records, not just task names or assumed dataset revisions.
    return {task: digest([{ "id": x.get("doc_id"), "doc": x.get("doc"),
                            "prompt_hash": x.get("prompt_hash")} for x in records])
            for task, records in samples.items()}


def evaluate_checkpoint(c, checkpoint_path, name, limit):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError("Install requirements-benchmarks.txt to run the standard benchmark suite") from exc
    root = benchmark_root(c, limit)
    root.mkdir(parents=True, exist_ok=True)
    tasks = c.get("benchmarks", {}).get("tasks", DEFAULT_TASKS)
    device, _ = setup_device(c)
    config = c.get("benchmarks", {})
    protocol = {"tasks": tasks, "limit": limit or None, "num_fewshot": 0, "batch_size": 1,
                "dtype": "bfloat16" if device.type == "cuda" else "float32",
                "max_length": config.get("max_length", 2048),
                "lm_eval_version": importlib.metadata.version("lm_eval"),
                "task_code_sha256": task_code_hash(Path(lm_eval.__file__).parent),
                "data_identity": digest(read_json(Path(c["output"]) / "data" / "manifest.json")),
                "seed": 1234, "apply_chat_template": False}
    check_identity(root / "protocol.json", protocol)
    destination = root / f"{name}.json"
    checkpoint = Path(checkpoint_path).resolve()
    model_files = sorted(checkpoint.glob("*.safetensors"))
    checkpoint_id = {"path": str(checkpoint), "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in model_files]}
    if destination.exists():
        old = read_json(destination)
        if old["protocol"] != protocol or old["checkpoint"] != checkpoint_id:
            raise RuntimeError("Existing benchmark belongs to a different checkpoint/protocol")
        return
    lm = HFLM(pretrained=str(checkpoint), tokenizer=str(Path(c["output"]) / "data" / "tokenizer"),
              dtype=protocol["dtype"], batch_size=1, device=str(device), max_length=protocol["max_length"],
              trust_remote_code=False, attn_implementation="sdpa")
    # Explicitly preserve the untied config saved by SFT; do not evaluate a re-tied variant.
    if not lm.model.config.tie_word_embeddings:
        assert lm.model.get_input_embeddings().weight.data_ptr() != lm.model.get_output_embeddings().weight.data_ptr()
    result = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
             limit=limit or None, log_samples=True, apply_chat_template=False,
             random_seed=1234, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234)
    result = sanitize(result)
    if not result.get("samples"):
        raise RuntimeError("Harness returned no sample logs; cannot audit identical benchmark records")
    fingerprints = sample_fingerprints(result["samples"])
    basepath = root / "baseline.json"
    if name != "baseline":
        if not basepath.exists():
            raise RuntimeError("Evaluate baseline first")
        if fingerprints != read_json(basepath)["sample_fingerprints"]:
            raise RuntimeError("Benchmark records/prompts changed between baseline and checkpoint")
    write_json(destination, {"name": name, "checkpoint": checkpoint_id, "protocol": protocol,
                            "sample_fingerprints": fingerprints, "harness": result})


def summarize_benchmarks(c, limit):
    from analyze import csv_write
    root = benchmark_root(c, limit)
    if not (root / "baseline.json").exists():
        return
    baseline = read_json(root / "baseline.json")
    rows = []
    for path in sorted(root.glob("seed_*.json")):
        current = read_json(path)
        if current["protocol"] != baseline["protocol"] or current["sample_fingerprints"] != baseline["sample_fingerprints"]:
            raise RuntimeError("Refusing to compare mismatched benchmark protocols")
        for task in current["protocol"]["tasks"]:
            metric = METRICS.get(task, "acc,none")
            old = baseline["harness"]["results"][task]
            new = current["harness"]["results"][task]
            if metric not in old or metric not in new:
                raise RuntimeError(f"Missing configured metric {metric} for {task}; inspect raw harness output")
            row = {"run": current["name"], "task": task, "metric": metric, "baseline": old[metric],
                   "after": new[metric], "forgetting_pp": 100 * (old[metric] - new[metric]), "limit": limit or "full"}
            sample_key = metric.split(",")[0]
            a = baseline["harness"]["samples"][task]
            b = current["harness"]["samples"][task]
            if all(sample_key in x for x in a + b):
                row["old_correct_now_wrong"] = sum(bool(x[sample_key]) and not bool(y[sample_key]) for x, y in zip(a, b))
                row["old_wrong_now_correct"] = sum(not bool(x[sample_key]) and bool(y[sample_key]) for x, y in zip(a, b))
                row["paired_examples"] = len(a)
            rows.append(row)
    csv_write(root / "retention.csv", rows)
    print(f"Benchmark comparison: {root / 'retention.csv'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["suite", "one", "report"])
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, help="0 for full benchmarks; default comes from config")
    p.add_argument("--checkpoint")
    p.add_argument("--name", default="baseline")
    args = p.parse_args()
    c = load_config(args.config)
    limit = args.limit if args.limit is not None else c.get("benchmarks", {}).get("limit", 0)
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.command == "one":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required")
        evaluate_checkpoint(c, args.checkpoint, args.name, limit)
    elif args.command == "suite":
        models = [(str(baseline_dir(c) / "model"), "baseline")]
        for seed in c["seeds"]:
            for lr in c["lrs"]:
                for mode in c["modes"]:
                    root = run_dir(c, seed, mode, lr)
                    if not (root / "complete.json").exists():
                        raise RuntimeError(f"Run is not complete: {root}")
                    models.append((str(latest_checkpoint(root)), f"seed_{seed}_{root.name}"))
        for checkpoint, name in models:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", "--config", args.config,
                                     "--limit", str(limit), "--checkpoint", checkpoint, "--name", name])
            if result.returncode:
                return result.returncode
        summarize_benchmarks(c, limit)
    else:
        summarize_benchmarks(c, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

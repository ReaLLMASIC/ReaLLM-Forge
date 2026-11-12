#!/usr/bin/env python3
"""Prepare an artifact manifest, evaluate CJK checkpoints, or build a report."""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.cjk_full_eval import (
    METRICS, RUN_NAMES, SCHEMA_VERSION, MetricAccumulator, evaluate_stream,
    export_tokens, file_identity, load_checkpoint, load_meta, sha256_file, token_bytes,
    verify_identity, write_json,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_training_counts(path, iteration, vocab_size):
    if not Path(path).is_file():
        return None
    result = [None] * vocab_size
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["iteration"]) != iteration:
                continue
            index = int(row["token_id"])
            if not 0 <= index < vocab_size or result[index] is not None:
                raise ValueError("Invalid or duplicate training-count row")
            result[index] = int(row["training_seen_count"])
    if any(count is None or count < 0 for count in result):
        raise ValueError(f"Incomplete training counts: {path}")
    return result


def prepare(args):
    if args.output.exists():
        raise ValueError("Manifest already exists; choose a fresh output path")
    records = []
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    flores = load_json(args.flores_manifest) if args.flores_manifest else None
    for name in RUN_NAMES:
        directory = args.run_root / "final_resume" / name
        data = args.data_root / name
        checkpoint_path, meta_path = directory / "ckpt.pt", data / "meta.pkl"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        meta = load_meta(meta_path)
        if checkpoint["model_args"]["vocab_size"] != meta["vocab_size"]:
            raise ValueError(f"Vocabulary mismatch: {name}")
        if checkpoint["config"]["dataset"].split("/")[-1] != name:
            raise ValueError(f"Checkpoint belongs to another run: {name}")
        result_path = directory / "final_result.json"
        final_result = load_json(result_path)
        if int(checkpoint["iter_num"]) != int(final_result["target_iteration"]):
            raise ValueError(f"Checkpoint is not the recorded endpoint: {name}")
        counts_path = directory / "per_token_metrics/per_token_metrics.csv"
        counts = read_training_counts(counts_path, checkpoint["iter_num"], meta["vocab_size"])
        if counts is not None and sum(counts) != final_result["final_training_tokens"]:
            raise ValueError(f"Training-count total mismatch: {name}")
        pieces = token_bytes(meta)
        jobs = []
        for split in ("train", "val"):
            path = data / f"{split}.bin"
            dtype = "uint16" if meta["vocab_size"] <= 65536 else "uint32"
            if path.stat().st_size % np.dtype(dtype).itemsize:
                raise ValueError(f"Misaligned token file: {path}")
            ids = np.memmap(path, mode="r", dtype=dtype)
            expected = meta["byte_metrics"][f"{split}_token_count"]
            if len(ids) != expected or int(ids.max()) >= len(pieces):
                raise ValueError(f"Invalid token counts or IDs: {path}")
            histogram = np.zeros(len(pieces), dtype=np.int64)
            for offset in range(0, len(ids), 1_000_000):
                histogram += np.bincount(ids[offset:offset+1_000_000], minlength=len(pieces))
            byte_count = int(histogram @ np.array([len(piece) for piece in pieces]))
            if byte_count != meta["byte_metrics"][f"{split}_utf8_bytes"]:
                raise ValueError(f"Decoded byte count mismatch: {name}/{split}")
            jobs.append(dict(id=f"ted_{split}", corpus="ted", split=split, scope="full",
                             status="ready", tokens=file_identity(path), dtype=dtype,
                             source_tokens=len(ids), source_utf8_bytes=byte_count))
        if flores:
            if flores["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported FLORES manifest schema")
            for job in flores["runs"][name]:
                if job["meta_sha256"] != sha256_file(meta_path):
                    raise ValueError(f"FLORES encoded with different tokenizer: {name}")
                jobs.append(job)
        records.append(dict(run=name, checkpoint=file_identity(checkpoint_path),
                            meta=file_identity(meta_path), iteration=int(checkpoint["iter_num"]),
                            training_counts=file_identity(counts_path) if counts is not None else None,
                            final_result=file_identity(result_path), jobs=jobs))
        del checkpoint
        print(f"Verified {name}: " + ", ".join(f"{job['id']}={job['status']}" for job in jobs), flush=True)
    write_json(args.output, dict(schema_version=SCHEMA_VERSION, code_commit=source_revision,
                                source_pr=926, ted_revision=args.ted_revision,
                                flores_manifest=file_identity(args.flores_manifest) if flores else None,
                                runs=records))


def persist_distribution(directory, accumulator):
    np.savez(directory / "totals.npz", counts=accumulator.counts,
             rank_histogram=accumulator.rank_histogram, **accumulator.sums)
    for name, array in accumulator.quantiles.items():
        with (directory / f"{name}.f32").open("wb") as handle:
            array[:accumulator.count].tofile(handle)


def mark_complete(directory):
    outputs = {}
    for name in ("result.json", "per_token_metrics.csv", "totals.npz", "left_inclusive.f32", "cosine.f32"):
        identity = file_identity(directory / name)
        outputs[name] = {key: identity[key] for key in ("bytes", "sha256")}
    write_json(directory / ".complete", dict(schema_version=SCHEMA_VERSION, outputs=outputs))


def verify_completed(directory):
    marker = load_json(directory / ".complete")
    if marker.get("schema_version") != SCHEMA_VERSION or not marker.get("outputs"):
        raise ValueError(f"Invalid completion marker: {directory}")
    for name, identity in marker["outputs"].items():
        verify_identity(dict(path=str(directory / name), **identity))


def combine_ted(directory, meta):
    """Combine saved train/val distributions without mixing their contexts."""
    directories = [directory / f"ted_{split}" for split in ("train", "val")]
    if not all((path / ".complete").exists() for path in directories):
        return
    for path in directories:
        verify_completed(path)
    summaries = [load_json(path / "result.json") for path in directories]
    if any(item["partial"] for item in summaries):
        return
    capacity = sum(item["scored_tokens"] for item in summaries)
    accumulator = MetricAccumulator(meta["vocab_size"], capacity, directory)
    try:
        for path, summary in zip(directories, summaries):
            with np.load(path / "totals.npz") as totals:
                accumulator.counts += totals["counts"]
                accumulator.rank_histogram += totals["rank_histogram"]
                for name in METRICS:
                    accumulator.sums[name] += totals[name]
            end = accumulator.count + summary["scored_tokens"]
            for name in accumulator.quantiles:
                if end > accumulator.count:
                    source = np.memmap(path / f"{name}.f32", mode="r", dtype=np.float32)
                    accumulator.quantiles[name][accumulator.count:end] = source
            accumulator.count = end
        combined = accumulator.summary([len(piece) for piece in token_bytes(meta)])
        combined.update(run=summaries[0]["run"], corpus="ted", split="train+val", scope="full",
                        status="complete", partial=False, contexts_reset_between_splits=True,
                        excluded_initial_tokens=sum(item["excluded_initial_tokens"] for item in summaries),
                        source_tokens=sum(item["source_tokens"] for item in summaries),
                        source_utf8_bytes=sum(item["source_utf8_bytes"] for item in summaries),
                        block_size=summaries[0]["block_size"], stride=summaries[0]["stride"],
                        dtype=summaries[0]["dtype"], model_state_unchanged=True)
        write_json(directory / "ted_combined.json", combined)
    finally:
        accumulator.close()


def evaluate(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. No evaluation started; restore GPU access or explicitly select CPU.")
    if args.device.startswith("cpu") and args.dtype != "float32":
        raise ValueError("Use --device cpu --dtype float32 together")
    if args.stride < 1 or args.batch_size < 1:
        raise ValueError("Stride and batch size must be positive")
    manifest = load_json(args.manifest)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    args.output.mkdir(parents=True, exist_ok=True)
    invocation = dict(manifest_sha256=sha256_file(args.manifest), device=args.device,
                      dtype=args.dtype, stride=args.stride, batch_size=args.batch_size,
                      max_tokens=args.max_tokens, geometry=args.geometry, schema_version=SCHEMA_VERSION)
    invocation["implementation_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (ROOT / "utils/cjk_full_eval.py", Path(__file__), ROOT / "model.py", ROOT / "gpt_conf.py")
    }
    invocation["torch_version"] = torch.__version__
    invocation_path = args.output / "evaluation_config.json"
    if invocation_path.exists() and load_json(invocation_path) != invocation:
        raise ValueError("Output belongs to a different evaluation; choose a fresh directory")
    write_json(invocation_path, invocation)
    if not (args.output / "source_manifest.json").exists():
        shutil.copyfile(args.manifest, args.output / "source_manifest.json")
    elif sha256_file(args.output / "source_manifest.json") != invocation["manifest_sha256"]:
        raise ValueError("Saved source manifest changed; inspect the output directory")
    selected = [record for record in manifest["runs"] if not args.runs or record["run"] in args.runs]
    if not selected or (args.runs and set(args.runs) - {record["run"] for record in selected}):
        raise ValueError("Unknown or empty run selection")
    for record in selected:
        directory = args.output / record["run"]
        directory.mkdir(exist_ok=True)
        checkpoint_path = verify_identity(record["checkpoint"])
        meta = load_meta(verify_identity(record["meta"]))
        counts = (read_training_counts(verify_identity(record["training_counts"]), record["iteration"],
                                      meta["vocab_size"]) if record["training_counts"] else None)
        jobs = [job for job in record["jobs"] if args.corpus == "all" or job["corpus"] == args.corpus]
        pending = []
        for job in jobs:
            output = directory / job["id"]
            if job["status"] != "ready":
                output.mkdir(exist_ok=True)
                write_json(output / "result.json", dict(run=record["run"], **job))
                continue
            verify_identity(job["tokens"])
            if job.get("selection"):
                verify_identity(job["selection"])
            if (output / ".complete").exists():
                verify_completed(output)
                print(f"Already complete: {record['run']}/{job['id']}", flush=True)
                continue
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"Incomplete output {output}; inspect it, then use a fresh output directory")
            pending.append(job)
        if pending:
            model, info = load_checkpoint(checkpoint_path, args.device)
            if info["iteration"] != record["iteration"]:
                raise ValueError("Checkpoint iteration changed")
            geometry = None
            if args.geometry:
                from utils.min_angle_graph_export import compute_min_angle_graph
                geometry = compute_min_angle_graph(model.lm_head.weight, compute_device=args.device)
            for job in pending:
                output = directory / job["id"]
                output.mkdir(exist_ok=True)
                ids = np.memmap(job["tokens"]["path"], mode="r", dtype=job["dtype"])
                partial = args.max_tokens is not None and args.max_tokens < len(ids)
                if args.max_tokens:
                    ids = ids[:args.max_tokens]
                started, last_print = time.monotonic(), [0.0]

                def progress(done, total):
                    now = time.monotonic()
                    if now - last_print[0] > 30 or done == total:
                        print(f"{record['run']}/{job['id']}: {done:,}/{total:,} targets; "
                              f"{done / max(now-started, 0.001):.0f} targets/s", flush=True)
                        last_print[0] = now

                summary, accumulator = evaluate_stream(
                    model, ids, meta, output, args.device, args.dtype, args.stride,
                    args.batch_size, info["iteration"], progress)
                try:
                    summary.update(run=record["run"], corpus=job["corpus"], split=job["split"],
                                   scope=job["scope"], status="complete", partial=partial,
                                   evaluated_source_tokens=len(ids), source_tokens=job["source_tokens"],
                                   source_utf8_bytes=job["source_utf8_bytes"],
                                   checkpoint_iteration=record["iteration"],
                                   elapsed_seconds=time.monotonic()-started,
                                   tokens_sha256=job["tokens"]["sha256"],
                                   source_selection=job.get("selection"))
                    export_tokens(output / "per_token_metrics.csv", meta, accumulator, counts, geometry,
                                  record["iteration"], record["run"],
                                  "train" if job["corpus"] == "ted" and job["split"] == "train" else "val")
                    persist_distribution(output, accumulator)
                    write_json(output / "result.json", summary)
                finally:
                    accumulator.close()
                mark_complete(output)
            verify_identity(record["checkpoint"])
            del model
        combine_ted(directory, meta)
    report(args.output)


def report(directory):
    results = [load_json(path) for path in sorted(directory.glob("*/*/result.json"))]
    results.extend(load_json(path) for path in sorted(directory.glob("*/ted_combined.json")))
    fields = ("run", "corpus", "split", "scope", "status", "partial", "source_tokens", "scored_tokens",
              "source_utf8_bytes", "scored_utf8_bytes", *METRICS, "bits_per_byte", "perplexity",
              "target_rank_95", "left_prob_95", "ln_f_cosine_95", "elapsed_seconds")
    with (directory / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["# CJK checkpoint evaluation", "", "Teacher-forced evaluation; no gradient updates.", "",
             "Loss is in nats/token. BPB uses scored UTF-8 bytes of the evaluated representation.",
             "TED train+val is a corpus aggregate, not held-out validation.",
             "Rank/left probability use strictly larger logits; ties are excluded.",
             "left_prob_95 includes target mass; ln_f_cosine_95 is the historical 5th percentile.",
             "Full and jointly encodable FLORES subsets must be compared separately.", "",
             "| Run | Corpus/split | Scope | Status | Targets | Loss | BPB | Target rank |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for result in results:
        number = lambda key: f"{result[key]:.6f}" if isinstance(result.get(key), (int, float)) else "—"
        status = "partial smoke" if result.get("partial") else result["status"]
        lines.append(f"| {result['run']} | {result['corpus']}/{result['split']} | {result['scope']} | "
                     f"{status} | {result.get('scored_tokens', '—')} | {number('loss')} | "
                     f"{number('bits_per_byte')} | {number('target_rank')} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed = [result for result in results if result.get("status") == "complete"
                 and not result.get("partial") and result["split"] != "train+val"]
    if completed:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric in ("bits_per_byte", "target_rank", "target_prob", "target_left_prob", "loss"):
            figure, axis = plt.subplots(figsize=(max(9, len(completed) * .45), 5))
            labels = [f"{row['run']}\n{row['corpus']}/{row['split']}/{row['scope']}" for row in completed]
            axis.bar(range(len(completed)), [row[metric] if row.get(metric) is not None else math.nan
                                            for row in completed])
            axis.set_xticks(range(len(completed)), labels, rotation=90)
            axis.set_ylabel(metric)
            figure.tight_layout()
            figure.savefig(directory / f"{metric}.png")
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("prepare", help="Verify 12 final checkpoints and TED token files")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--ted-revision", required=True, help="Hub revision used to prepare the original TED data")
    plan.add_argument("--flores-manifest", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(action=prepare)
    run = subparsers.add_parser("run", help="Evaluate; refuses implicit CPU fallback")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    run.add_argument("--stride", type=int, default=512)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--runs", nargs="+", choices=RUN_NAMES)
    run.add_argument("--corpus", choices=("ted", "flores", "all"), default="all")
    run.add_argument("--max-tokens", type=int, help="Smoke test only; output explicitly marked partial")
    run.add_argument("--geometry", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(action=evaluate)
    summary = subparsers.add_parser("report")
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(action=lambda args: report(args.output))
    args = parser.parse_args()
    if getattr(args, "max_tokens", None) is not None and args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")
    try:
        args.action(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

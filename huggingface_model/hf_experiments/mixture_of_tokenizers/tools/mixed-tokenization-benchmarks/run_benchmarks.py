#!/usr/bin/env python3
"""Evaluate the four completed checkpoints sequentially; never train a model."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import uuid
import zipfile

LANGUAGES = ("zh", "ko", "ja", "en")
RUNS = {1337: "runs/100m", 1338: "runs/100m-seed1338"}
SIZES = (10000, 20000)
HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_jobs(project, output, args):
    tasks = output / "tasks"
    jobs = []
    for seed, root in RUNS.items():
        for size in SIZES:
            run = project / root / f"k{size}"
            model = run / "checkpoint-best"
            dest = output / "results" / f"seed{seed}" / f"k{size}"
            shared = ["--package-dir", str(project), "--model", str(model),
                      "--device", args.device, "--threads", str(args.threads),
                      "--training-seed", str(seed), "--skip-existing"]
            lm_output = dest / "lm-bytes.json"
            task_output = dest / "tasks.json"
            lm_cmd = [sys.executable, str(HERE / "score_lm.py"), *shared,
                      "--data", str(project / "data/prepared"),
                      "--split", "test" if args.split_set == "final" else "val",
                      "--context-mode", "bytes", "--max-documents", str(args.max_documents),
                      "--overlap-cache", str(output / "overlap-cache"),
                      "--output", str(lm_output)]
            task_cmd = [sys.executable, str(HERE / "score_tasks.py"), *shared,
                        "--tasks", str(tasks / "tasks.jsonl"),
                        "--manifest", str(tasks / "manifest.json"),
                        "--output", str(task_output)]
            jobs.append(dict(seed=seed, size=size, run=run, model=model,
                             lm_output=lm_output, task_output=task_output,
                             commands=[lm_cmd, task_cmd]))
    return jobs


def preflight(project, jobs, args):
    errors = []
    required = [project / name for name in
                ("evaluate.py", "modeling_mixture.py", "configuration_mixture.py", "tokenization_mixture.py")]
    required += [HERE / name for name in ("prepare_tasks.py", "score_lm.py", "score_tasks.py", "report_tasks.py")]
    for lang in LANGUAGES:
        required += [project / "data/prepared" / lang / f"{split}.jsonl"
                     for split in ("train", "val", "test")]
    errors += [f"Missing required file: {p}" for p in required if not p.is_file()]
    fingerprints = {}
    for job in jobs:
        try:
            run = read_json(job["run"] / "run_config.json")
            summary = read_json(job["run"] / "summary.json")
            model = read_json(job["model"] / "config.json")
            effective = run["effective_training"]
            settings = summary["settings"]
            if effective["seed"] != job["seed"] or settings["seed"] != job["seed"]:
                raise ValueError("training seed does not match run identity")
            if effective["target_tokens"] != 100_000_000 or settings["target_tokens"] != 100_000_000:
                raise ValueError("expected the completed 100M training runs")
            if summary.get("budget_reached") is not True or summary.get("interrupted") is not False:
                raise ValueError("training summary does not confirm budget completion")
            if summary.get("predicted_tokens", 0) < 100_000_000:
                raise ValueError("training budget is incomplete")
            if model.get("lexical_size") != job["size"] or model.get("max_position_embeddings") != 512:
                raise ValueError("checkpoint vocabulary/context does not match experiment")
            fingerprint = run.get("tokenizer_fingerprint")
            if not fingerprint or summary.get("tokenizer_fingerprint") != fingerprint or model.get("tokenizer_fingerprint") != fingerprint:
                raise ValueError("checkpoint/run/summary tokenizer fingerprints differ")
            if job["size"] in fingerprints and fingerprints[job["size"]] != fingerprint:
                raise ValueError("tokenizers differ across seeds")
            fingerprints[job["size"]] = fingerprint
            if not (job["model"] / "tokenizer").is_dir():
                raise ValueError("saved checkpoint tokenizer directory is missing")
            if not (list(job["model"].glob("*.safetensors")) or list(job["model"].glob("pytorch_model*.bin"))):
                raise ValueError("checkpoint weights are missing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid run {job['run']}: {exc}")
    split = "test" if args.split_set == "final" else "val"
    for lang in LANGUAGES:
        path = project / "data/prepared" / lang / f"{split}.jsonl"
        if not path.is_file():
            continue
        try:
            count = 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    if not isinstance(json.loads(line).get("text"), str):
                        raise ValueError("document has no text string")
                    count += 1
                    if count >= args.max_documents:
                        break
            if count < args.max_documents:
                raise ValueError(f"has {count} documents; requested {args.max_documents}; use a smaller --max-documents for all runs")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Invalid LM input {path}: {exc}")
    return errors


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".benchmark-run.lock"
    token = uuid.uuid4().hex
    record = dict(pid=os.getpid(), hostname=socket.gethostname(), token=token,
                  created_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {lock}. Check its PID/hostname; remove it only after confirming that process has stopped.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream)
        yield
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            lock.unlink()


def collect_archive(project, output, jobs, commands, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    destination = project / f"language-benchmark-results-{stamp}.zip"
    staging = destination.with_suffix(".zip.partial")
    members = {}
    for job in jobs:
        prefix = f"results/seed{job['seed']}/k{job['size']}"
        members[prefix + "/lm-bytes.json"] = job["lm_output"]
        members[prefix + "/tasks.json"] = job["task_output"]
        for name in ("run_config.json", "summary.json"):
            members[prefix + "/" + name] = job["run"] / name
    for path in (output / "report").rglob("*"):
        if path.is_file():
            members["report/" + path.relative_to(output / "report").as_posix()] = path
    members["task-manifest.json"] = output / "tasks/manifest.json"
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".md", ".json"):
            members["source/mixed-tokenization-benchmarks/" + path.name] = path
    manifest = dict(format_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    purpose="Language benchmark results; excludes model weights, corpus and benchmark text",
                    project_root=str(project), split_set=args.split_set,
                    max_documents=args.max_documents, commands=commands, files=[])
    created = False
    try:
        with staging.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, path in sorted(members.items()):
                    content = path.read_bytes()
                    archive.writestr(name, content)
                    manifest["files"].append(dict(path=name, size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
                archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        with zipfile.ZipFile(staging) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive CRC validation failed")
            for entry in manifest["files"]:
                if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                    raise ValueError("archive checksum validation failed")
        # Exclusive destination creation also protects against an unlikely name collision.
        os.link(staging, destination)
        staging.unlink()
    except BaseException:
        if created:
            staging.unlink(missing_ok=True)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("runs/language-benchmarks-v1"))
    parser.add_argument("--split-set", choices=("dev", "final"), default="final")
    parser.add_argument("--max-documents", type=positive, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--bootstrap-repetitions", type=positive, default=10000)
    parser.add_argument("--dry-run", action="store_true", help="validate local inputs and print commands without downloading/scoring/writing")
    args = parser.parse_args(argv)
    project = args.project_root.expanduser().resolve()
    output = args.output.expanduser()
    output = (output if output.is_absolute() else project / output).resolve()
    jobs = make_jobs(project, output, args)
    errors = preflight(project, jobs, args)
    if errors:
        parser.exit(2, "Preflight failed:\n" + "\n".join(errors) + "\n")
    preparation = [sys.executable, str(HERE / "prepare_tasks.py"), "--output", str(output / "tasks"), "--split-set", args.split_set]
    report = [sys.executable, str(HERE / "report_tasks.py"), "--results",
              *[str(job["task_output"]) for job in jobs], "--lm-results",
              *[str(job["lm_output"]) for job in jobs], "--output-dir", str(output / "report"),
              "--bootstrap-repetitions", str(args.bootstrap_repetitions)]
    commands = [preparation] + [command for job in jobs for command in job["commands"]] + [report]
    if args.dry_run:
        print("Preflight passed. Commands (sequential; no training):")
        for command in commands:
            print(shlex.join(command))
        return 0
    with output_lock(output):
        for command in commands:
            print("\nRunning: " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=project, check=True)
        archive = collect_archive(project, output, jobs, commands, args)
    print(f"\nReport: {output / 'report/report.md'}\nUpload this result archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

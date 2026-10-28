#!/usr/bin/env python3
"""Export paired document losses, bootstrap them, and collect the audit evidence.

This launcher uses only the Python standard library. It never trains a model.
Run from the existing experiment environment; model dependencies are loaded only
by the evaluator subprocess. Existing exports are checked by the evaluator
before being reused, rather than silently accepted based on their filenames.
"""

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
RUN_ROOTS = {1337: Path("runs/100m"), 1338: Path("runs/100m-seed1338")}
LEXICAL_SIZES = (10000, 20000)
CONTEXT_MODES = ("bytes", "token")
TARGET_TOKENS = 100_000_000


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def unique_stamp() -> str:
    # UTC and a random suffix prevent collisions between repeated invocations.
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def make_jobs(project: Path, output: Path, args: argparse.Namespace) -> list[dict]:
    evaluator = Path(__file__).resolve().with_name("evaluate_documents.py")
    jobs = []
    for seed, run_root in RUN_ROOTS.items():
        for size in LEXICAL_SIZES:
            run = run_root / f"k{size}"
            model = run / "checkpoint-best"
            for mode in CONTEXT_MODES:
                reference = run / f"evaluation-val-{mode}-1000.json"
                destination = output / f"seed{seed}" / f"k{size}" / f"{mode}.json"
                command = [
                    sys.executable, str(evaluator),
                    "--project-root", str(project),
                    "--model", model.as_posix(),
                    "--data", "data/prepared", "--split", "val",
                    "--context-mode", mode,
                    "--max-documents", str(args.max_documents),
                    "--device", args.device, "--threads", str(args.threads),
                    "--training-seed", str(seed),
                    "--reference", str(project / reference),
                    "--output", str(destination), "--skip-existing",
                ]
                jobs.append({"seed": seed, "size": size, "mode": mode,
                             "run": run, "model": model, "reference": reference,
                             "output": destination, "command": command})
    return jobs


def preflight(project: Path, jobs: list[dict]) -> list[str]:
    """Check all inputs before scheduling any expensive model work."""
    errors: list[str] = []
    package = Path(__file__).resolve().parent
    required = [package / "evaluate_documents.py", package / "paired_bootstrap.py",
                project / "evaluate.py", project / "modeling_mixture.py",
                project / "tokenization_mixture.py", project / "configuration_mixture.py"]
    required.extend(project / "data/prepared" / language / "val.jsonl"
                    for language in LANGUAGES)
    for path in required:
        if not path.is_file():
            errors.append(f"Missing required file: {path}")

    shared_text = None
    shared_source_digests = None
    fingerprints: dict[int, str] = {}
    checked_runs = set()
    for job in jobs:
        run = project / job["run"]
        model = project / job["model"]
        if run not in checked_runs:
            checked_runs.add(run)
            if not (model / "tokenizer").is_dir():
                errors.append(f"Missing saved checkpoint tokenizer directory: {model / 'tokenizer'}")
            try:
                config = read_json(run / "run_config.json")
                summary = read_json(run / "summary.json")
                model_config = read_json(model / "config.json")
                effective = config["effective_training"]
                settings = summary["settings"]
                if effective["seed"] != job["seed"] or settings["seed"] != job["seed"]:
                    raise ValueError("effective training seed does not match run identity")
                if effective["target_tokens"] != TARGET_TOKENS or settings["target_tokens"] != TARGET_TOKENS:
                    raise ValueError("expected a fresh 100M-target training configuration")
                if summary.get("budget_reached") is not True or summary.get("interrupted") is not False:
                    raise ValueError("summary does not confirm uninterrupted budget completion")
                if not isinstance(summary.get("predicted_tokens"), int) or summary["predicted_tokens"] < TARGET_TOKENS:
                    raise ValueError("summary predicted-token total is below the completed budget")
                if model_config.get("lexical_size") != job["size"]:
                    raise ValueError("checkpoint lexical_size does not match run identity")
                if model_config.get("max_position_embeddings") != 512:
                    raise ValueError("expected the accepted 512-token checkpoint context")
                fingerprint = config.get("tokenizer_fingerprint")
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError("run configuration has no tokenizer fingerprint")
                if summary.get("tokenizer_fingerprint") != fingerprint or model_config.get("tokenizer_fingerprint") != fingerprint:
                    raise ValueError("tokenizer fingerprints disagree across checkpoint/configuration/summary")
                if job["size"] in fingerprints and fingerprints[job["size"]] != fingerprint:
                    raise ValueError("tokenizer fingerprint differs between training seeds")
                fingerprints[job["size"]] = fingerprint
                identity = summary["validation_identity"]
                source_digests = identity["source_digests"]
                if set(source_digests) != set(LANGUAGES):
                    raise ValueError("summary validation digests do not cover all four languages")
                if shared_source_digests is None:
                    shared_source_digests = source_digests
                elif source_digests != shared_source_digests:
                    raise ValueError("validation source digests differ between training runs")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"Invalid run {run}: {exc}")
        try:
            reference = read_json(project / job["reference"])
            if reference.get("lexical_size") != job["size"]:
                raise ValueError("reference lexical size is wrong")
            if reference.get("context_mode") != job["mode"] or reference.get("context_tokens") != 512:
                raise ValueError("reference context mode/length is wrong")
            if reference.get("split") != "val":
                raise ValueError("reference is not validation data")
            if reference.get("metric") != "canonical_tokenization_bits_per_utf8_byte":
                raise ValueError("reference uses an unexpected metric")
            model_parts = Path(str(reference.get("model", ""))).parts
            expected_parts = job["model"].parts
            if model_parts[-len(expected_parts):] != expected_parts:
                raise ValueError("reference does not identify the expected best checkpoint")
            language_rows = reference["languages"]
            if set(language_rows) != set(LANGUAGES):
                raise ValueError("reference language set is wrong")
            counts = {}
            for language in LANGUAGES:
                routes = language_rows[language]
                if set(routes) != {language, "byte"}:
                    raise ValueError(f"wrong route set for {language}")
                for route in (language, "byte"):
                    row = routes[route]
                    if row["documents"] != 1000 or row["eos_count"] != 1000:
                        raise ValueError(f"expected exactly 1000 documents and EOS events for {language}/{route}")
                native = routes[language]
                byte = routes["byte"]
                keys = ("documents", "source_bytes", "source_characters", "eos_count")
                if any(native[key] != byte[key] for key in keys):
                    raise ValueError(f"native and byte source counts differ for {language}")
                counts[language] = {key: native[key] for key in keys}
            hashes = reference["evaluated_text_sha256"]
            if set(hashes) != set(LANGUAGES):
                raise ValueError("reference text hashes do not cover all languages")
            text_identity = {"hashes": hashes, "counts": counts}
            if shared_text is None:
                shared_text = text_identity
            elif shared_text != text_identity:
                raise ValueError("reference text hashes/source counts differ across conditions")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"Invalid reference {project / job['reference']}: {exc}")
    return errors


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run_next.lock"
    token = uuid.uuid4().hex
    record = {"pid": os.getpid(), "hostname": socket.gethostname(), "token": token,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(
            f"Output is locked: {lock}\n"
            "Another launcher may be running. Inspect the lock's PID/hostname. "
            "Remove that lock manually only after confirming its process stopped; "
            "this launcher never removes a pre-existing lock."
        ) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        yield
    finally:
        # Do not remove a replacement lock if somebody manually removed ours.
        try:
            if read_json(lock).get("token") == token:
                lock.unlink()
        except (OSError, ValueError):
            pass


def create_archive(project: Path, jobs: list[dict], comparison: Path,
                   commands: list[list[str]], stamp: str, args: argparse.Namespace) -> Path:
    archive = project / f"document-evaluation-results-{stamp}.zip"
    members: dict[str, Path] = {}
    for job in jobs:
        prefix = f"results/seed{job['seed']}/k{job['size']}"
        members[f"{prefix}/{job['mode']}.json"] = job["output"]
        members[f"{prefix}/reference-{job['mode']}.json"] = project / job["reference"]
        for name in ("run_config.json", "summary.json"):
            members[f"{prefix}/{name}"] = project / job["run"] / name
    members["comparison.json"] = comparison
    package = Path(__file__).resolve().parent
    for source in sorted(package.glob("*.py")):
        members[f"source/document_eval/{source.name}"] = source
    readme = package / "README.md"
    if readme.is_file():
        members["source/document_eval/README.md"] = readme
    manifest = {
        "format": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "paired document evaluation; no training, weights, or corpus text included",
        "project_root": str(project), "device": args.device,
        "max_documents": args.max_documents, "bootstrap_repetitions": args.bootstrap_repetitions,
        "commands": commands, "files": [],
    }
    # Exclusive creation preserves previous evidence even on name collisions.
    created = False
    try:
        with archive.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, path in sorted(members.items()):
                    contents = path.read_bytes()
                    bundle.writestr(name, contents)
                    manifest["files"].append({"path": name, "size_bytes": len(contents),
                                               "sha256": hashlib.sha256(contents).hexdigest()})
                bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("runs/document-evaluation"),
                        help="output root; relative paths are relative to --project-root")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-documents", type=positive_int, default=1000,
                        help="currently must be 1000 to reconcile the accepted reference evaluations")
    parser.add_argument("--bootstrap-repetitions", type=positive_int, default=10000)
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="print all commands and validate inputs without running or writing anything")
    args = parser.parse_args(argv)
    if args.max_documents != 1000:
        parser.error("only --max-documents 1000 is supported by this accepted-reference workflow")
    project = args.project_root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = project / output
    output = output.resolve()
    stamp = unique_stamp()
    comparison = output / f"comparison-{stamp}.json"
    jobs = make_jobs(project, output, args)
    bootstrap = [sys.executable, str(Path(__file__).resolve().with_name("paired_bootstrap.py")),
                 "--input-root", str(output), "--output", str(comparison),
                 "--repetitions", str(args.bootstrap_repetitions)]
    commands = [job["command"] for job in jobs] + [bootstrap]
    errors = preflight(project, jobs)
    if args.dry_run:
        print(f"Project: {project}\nOutput: {output}\nEvaluation only; eight exports then paired bootstrap.")
        for index, command in enumerate(commands, 1):
            print(f"[{index}/{len(commands)}] {shlex.join(command)}")
        print(f"Archive destination: {project / ('document-evaluation-results-' + stamp + '.zip')}")
        if errors:
            print("Preflight failed; nothing was run or written:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 2
        print("Preflight passed. Nothing was run or written.")
        return 0
    if errors:
        print("Preflight failed before starting evaluation:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    try:
        with output_lock(output):
            for index, job in enumerate(jobs, 1):
                print(f"[{index}/{len(jobs)}] seed {job['seed']}, K={job['size']}, {job['mode']}; "
                      "existing exports will be strictly checked before reuse.", flush=True)
                subprocess.run(job["command"], cwd=project, check=True)
                if not job["output"].is_file():
                    raise RuntimeError(f"Evaluator returned success but output is absent: {job['output']}")
            print("Computing paired document-bootstrap intervals...", flush=True)
            subprocess.run(bootstrap, cwd=project, check=True)
            if not comparison.is_file():
                raise RuntimeError(f"Bootstrap returned success but comparison is absent: {comparison}")
            archive = create_archive(project, jobs, comparison, commands, stamp, args)
            print(f"\nComplete. Upload this archive:\n{archive}", flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"Stopped after a subprocess failed with exit code {exc.returncode}. "
              "Completed exports were preserved. Fix the reported issue and rerun this command; "
              "existing exports will be revalidated before reuse.", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed exports were preserved; rerun to validate and reuse them.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

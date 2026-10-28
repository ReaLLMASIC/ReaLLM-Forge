"""Evaluate a frozen checkpoint on an untouched prepared corpus split.

The installed experiment supplies evaluate_corpus and its original precision,
context-window, tokenization, and EOS policies. This wrapper adds exact sample
counts, document-overlap checks, provenance, and validated output reuse. It does
not train a model, filter overlapping examples, or download a dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

LANGUAGES = ("zh", "ko", "ja", "en")
METRIC = "canonical_tokenization_bits_per_utf8_byte"
EOS_POLICY = "one genuine document EOS, reported separately from BPB"
POLICIES = {
    "bytes": "same Unicode chunks <= T-2 UTF-8 bytes for every route; BOS resets each chunk",
    "token": "T-1 stride; one token overlap; reset positions each window",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def iter_texts(path, limit=None):
    """Match the original evaluate.py JSONL reader without importing torch."""
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{number}: each record must have a string 'text'")
            yield row["text"]
            count += 1
            if limit is not None and count >= limit:
                return


def select_documents(data, split, maximum):
    """Audit the exact first N documents; never silently reduce N or filter."""
    if maximum < 1:
        raise ValueError("max-documents must be positive")
    selected, sequence_hashes, file_hashes = {}, {}, {}
    for language in LANGUAGES:
        path = Path(data) / language / f"{split}.jsonl"
        rows, digest = [], hashlib.sha256()
        for index, text in enumerate(iter_texts(path, maximum)):
            encoded = text.encode("utf-8")
            if not encoded:
                raise ValueError(f"Empty selected document in {path}, document index {index}; "
                                 "refusing to filter or change the requested sample")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            rows.append({"document_index": index,
                         "text_sha256": hashlib.sha256(encoded).hexdigest(),
                         "source_bytes": len(encoded), "source_characters": len(text)})
        if len(rows) != maximum:
            raise ValueError(f"Requested exactly {maximum} documents for {language}/{split}; "
                             f"only {len(rows)} are available")
        selected[language] = rows
        sequence_hashes[language] = digest.hexdigest()
        file_hashes[f"{language}/{split}.jsonl"] = sha256_file(path)
    return selected, sequence_hashes, file_hashes


def publish_json(path, value):
    """Publish a complete JSON atomically without replacing existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def scan_overlap(path, selected):
    candidate_hashes = {row["text_sha256"] for row in selected}
    matched_hashes, corpus_documents, matching_corpus_documents = set(), 0, 0
    for text in iter_texts(path):
        corpus_documents += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in candidate_hashes:
            matching_corpus_documents += 1
            matched_hashes.add(digest)
    matches = [row["document_index"] for row in selected if row["text_sha256"] in matched_hashes]
    return {"compared_corpus_documents": corpus_documents,
            "matching_corpus_documents": matching_corpus_documents,
            "matching_selected_documents": len(matches),
            "matched_selected_document_indices": matches,
            "matched_text_sha256": sorted(matched_hashes)}


def overlap_for_file(path, selected, language, compared_split, cache_dir):
    file_hash = sha256_file(path)
    signature = {"schema_version": 1, "language": language, "compared_split": compared_split,
                 "compared_file_sha256": file_hash,
                 "selected_documents_sha256": json_digest(selected)}
    key = json_digest(signature)
    cache_path = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        cached = read_json(cache_path)
        if cached.get("signature") != signature or cached.get("result_sha256") != json_digest(cached.get("result")):
            raise ValueError(f"Overlap cache failed its identity/integrity check: {cache_path}")
        result = cached["result"]
        indices = result.get("matched_selected_document_indices")
        hashes = result.get("matched_text_sha256")
        if (not isinstance(indices, list) or not isinstance(hashes, list)
                or indices != sorted(set(indices))
                or any(not isinstance(i, int) or i < 0 or i >= len(selected) for i in indices)
                or result.get("matching_selected_documents") != len(indices)
                or hashes != sorted({selected[i]["text_sha256"] for i in indices})
                or any(not isinstance(result.get(k), int) or result[k] < 0 for k in
                       ("compared_corpus_documents", "matching_corpus_documents"))):
            raise ValueError(f"Invalid overlap cache counters: {cache_path}")
    else:
        result = scan_overlap(path, selected)
        if sha256_file(path) != file_hash:
            raise ValueError(f"Corpus file changed during overlap audit: {path}")
        if cache_path is not None:
            cached = {"signature": signature, "result": result, "result_sha256": json_digest(result)}
            try:
                publish_json(cache_path, cached)
            except FileExistsError:
                # A concurrent checkpoint evaluation may finish the same scan.
                if read_json(cache_path) != cached:
                    raise ValueError(f"Concurrent overlap cache contents disagree: {cache_path}") from None
    return {**signature, **result}


def audit_overlap(data, split, selected, cache_dir=None, enabled=True):
    if not enabled:
        return {"status": "not_checked", "enabled": False,
                "limitation": "Document overlap with training/validation was explicitly not checked."}
    comparisons = {}
    compared_splits = ("train", "val") if split == "test" else ("train",)
    for language in LANGUAGES:
        comparisons[language] = {}
        for compared in compared_splits:
            result = overlap_for_file(Path(data) / language / f"{compared}.jsonl", selected[language],
                                      language, compared, cache_dir)
            comparisons[language][compared] = result
            if result["matching_selected_documents"]:
                raise ValueError(f"Held-out overlap: {result['matching_selected_documents']} selected "
                                 f"{language}/{split} documents exactly match {language}/{compared}. "
                                 "Refusing to filter examples or publish an evaluation.")
    return {"status": "passed", "enabled": True, "comparisons": comparisons,
            "policy": "Exact UTF-8 document equality against same-language prepared splits; no filtering.",
            "limitation": "Does not detect near-duplicates, excerpts, cross-language overlap, or benchmark contamination."}


def validate_report(report, selected, text_hashes, *, split, mode, context, size, seed):
    required = {"metric": METRIC, "eos_policy": EOS_POLICY, "split": split,
                "context_mode": mode, "context_tokens": context, "context_policy": POLICIES[mode],
                "lexical_size": size, "training_seed": seed, "evaluated_text_sha256": text_hashes}
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"LM report identity/policy mismatch: {key}")
    if set(report.get("languages", {})) != set(LANGUAGES):
        raise ValueError("LM report has an unexpected language set")
    native, byte = [], []
    integer_keys = ("documents", "text_tokens", "source_bytes", "source_characters",
                    "fallback_tokens", "fallback_bytes", "eos_count", "windows")
    for language in LANGUAGES:
        rows = selected[language]
        routes = report["languages"][language]
        if set(routes) != {language, "byte"}:
            raise ValueError(f"Unexpected routes for {language}")
        for route in (language, "byte"):
            values = routes[route]
            for key in integer_keys:
                if not isinstance(values.get(key), int) or isinstance(values[key], bool) or values[key] < 0:
                    raise ValueError(f"Invalid {language}/{route}/{key}")
            for key in ("text_nll", "eos_nll", "bpb"):
                if not isinstance(values.get(key), (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
                    raise ValueError(f"Invalid {language}/{route}/{key}")
            if (values["documents"] != len(rows) or values["eos_count"] != len(rows)
                    or values["source_bytes"] != sum(x["source_bytes"] for x in rows)
                    or values["source_characters"] != sum(x["source_characters"] for x in rows)):
                raise ValueError(f"Selected source totals disagree for {language}/{route}")
            expected_bpb = values["text_nll"] / values["source_bytes"] / math.log(2)
            if not math.isclose(values["bpb"], expected_bpb, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"BPB arithmetic mismatch for {language}/{route}")
            if route == "byte" and not (values["text_tokens"] == values["source_bytes"]
                                         == values["fallback_tokens"] == values["fallback_bytes"]):
                raise ValueError(f"Byte-route token accounting mismatch for {language}")
            (native if route == language else byte).append(values["bpb"])
    for name, values in (("native_bpb", native), ("byte_bpb", byte)):
        recorded = report.get("macro_average", {}).get(name)
        if not isinstance(recorded, (int, float)) or not math.isfinite(recorded) or not math.isclose(
                recorded, sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Macro BPB arithmetic mismatch: {name}")


def configure_runtime(device, threads):
    import torch
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")


def run(args, shared=None, runtime_setup=None):
    if args.max_documents < 1 or args.threads < 1:
        raise ValueError("max-documents and threads must be positive")
    if shared is None:
        import score_tasks as shared
    package = Path(args.package_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.skip_existing:
        raise FileExistsError(f"Output already exists: {output}; use --skip-existing for strict verification")
    print(f"Auditing LM checkpoint: {model_path}", flush=True)
    identity = shared.read_training_identity(model_path, expected_seed=args.training_seed)
    seed, size = identity["training_seed"], identity["lexical_size"]
    context = int(identity["model_config"]["max_position_embeddings"])
    script_path = Path(__file__).resolve()
    proof = shared.build_provenance(package, model_path, args.device, args.threads, script_path)
    print(f"Checking exactly {args.max_documents} {args.split} documents per language...", flush=True)
    selected, text_hashes, data_hashes = select_documents(data, args.split, args.max_documents)
    cache_dir = Path(args.overlap_cache).expanduser().resolve() if args.overlap_cache else output.parent / ".overlap-cache"
    print("Checking document overlap against prepared splits (reusing verified cache entries)..."
          if args.check_overlap else "Document overlap checking explicitly disabled.", flush=True)
    overlap = audit_overlap(data, args.split, selected, cache_dir, args.check_overlap)
    request = {"model": str(model_path), "data": str(data), "split": args.split,
               "max_documents": args.max_documents, "context_mode": args.context_mode,
               "context_tokens": context, "training_seed": seed, "check_overlap": args.check_overlap}
    provenance = {**proof, "training_identity": identity,
                  "data_files_sha256": data_hashes, "request": request,
                  "selected_documents_sha256": json_digest(selected),
                  "evaluated_text_sha256": text_hashes, "overlap_audit": overlap}
    helper_path = script_path.with_name("score_tasks.py")
    if helper_path.is_file():
        provenance["benchmark_helper_sha256"] = sha256_file(helper_path)
    validation_args = dict(split=args.split, mode=args.context_mode, context=context, size=size, seed=seed)
    if output.exists():
        saved = read_json(output)
        if saved.get("provenance") != provenance or saved.get("selected_documents") != selected:
            raise ValueError(f"Existing LM evaluation provenance changed: {output}; use a new output path")
        if Path(saved.get("model", "")).resolve() != model_path:
            raise ValueError("Existing LM evaluation identifies another checkpoint")
        validate_report(saved, selected, text_hashes, **validation_args)
        print(f"Verified existing LM evaluation; skipped scoring: {output}", flush=True)
        return saved
    print(f"Loading LM checkpoint on {args.device}: {model_path}", flush=True)
    (runtime_setup or configure_runtime)(args.device, args.threads)
    base, model, tokenizer = shared.load_checkpoint(package, model_path, args.device)
    if tokenizer.lexical_size != size or int(model.config.max_position_embeddings) != context:
        raise ValueError("Loaded model/tokenizer identity differs from the audited checkpoint configuration")
    print(f"Scoring all four languages with native and byte routes ({args.context_mode} context); "
          "the original evaluator reports results after this scoring pass...", flush=True)
    started = time.monotonic()
    result = base.evaluate_corpus(model, tokenizer, data, split=args.split,
                                  max_documents=args.max_documents, context_mode=args.context_mode,
                                  device=args.device)
    elapsed = time.monotonic() - started
    result.update({"schema_version": 1, "evaluation_kind": "heldout_corpus_lm",
                   "model": str(model_path), "lexical_size": size, "training_seed": seed,
                   "selected_documents": selected, "overlap_audit": overlap,
                   "provenance": provenance, "elapsed_scoring_seconds": elapsed})
    validate_report(result, selected, text_hashes, **validation_args)
    if shared.build_provenance(package, model_path, args.device, args.threads, script_path) != proof:
        raise ValueError("Checkpoint or experiment provenance changed during evaluation; no output published")
    if shared.read_training_identity(model_path, expected_seed=args.training_seed) != identity:
        raise ValueError("Training identity changed during evaluation; no output published")
    if any(sha256_file(data / relative) != digest for relative, digest in data_hashes.items()):
        raise ValueError("Evaluation corpus file changed during scoring; no output published")
    if args.check_overlap:
        for language, comparisons in overlap["comparisons"].items():
            for split, comparison in comparisons.items():
                if sha256_file(data / language / f"{split}.jsonl") != comparison["compared_file_sha256"]:
                    raise ValueError("Overlap-audit corpus file changed during scoring; no output published")
    if helper_path.is_file() and sha256_file(helper_path) != provenance.get("benchmark_helper_sha256"):
        raise ValueError("Benchmark loading helper changed during scoring; no output published")
    publish_json(output, result)
    print(json.dumps(result["macro_average"], indent=2), flush=True)
    print(f"Saved LM evaluation: {output}", flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default=".", help="existing experiment directory containing evaluate.py")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--context-mode", choices=("bytes", "token"), default="bytes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overlap-cache", help="shared audit cache directory; defaults beside output")
    parser.add_argument("--check-overlap", action=argparse.BooleanOptionalAction, default=True,
                        help="check exact same-language document overlap with train (and val for test)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_documents < 1 or args.threads < 1:
        parser.error("max-documents and threads must be positive")
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"LM evaluation stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and package the four expanded validation JSONs without changing them.

Run from the actual project root:
    python collect_expanded_evaluations.py
    python collect_expanded_evaluations.py --root runs/100m --output verified.zip

Only k{10000,20000}/evaluation-val-{bytes,token}-1000.json are read.
Actual document counts are checked and recorded, not inferred from filenames.
No training/evaluation is run. No dependencies outside the Python standard library.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

LANGUAGES = ("zh", "ko", "ja", "en")
SIZES = (10000, 20000)
POLICIES = {
    "bytes": "same Unicode chunks <= T-2 UTF-8 bytes for every route; BOS resets each chunk",
    "token": "T-1 stride; one token overlap; reset positions each window",
}
METRIC = "canonical_tokenization_bits_per_utf8_byte"
EOS_POLICY = "one genuine document EOS, reported separately from BPB"


class ValidationError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValidationError(f"Non-finite JSON number: {value}")


def count(value, label, positive=True):
    require(type(value) is int and value >= (1 if positive else 0),
            f"{label} must be a {'positive' if positive else 'non-negative'} integer")
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{label} must be finite and non-negative")
    return value


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
            f"{label}: stored {actual!r}, recomputed {expected!r}")


def validate_file(data, size, mode):
    require(isinstance(data, dict), "Top-level JSON must be an object")
    require(type(data.get("lexical_size")) is int and data["lexical_size"] == size,
            f"lexical_size must be {size}; got {data.get('lexical_size')!r}. "
            "Check for files copied from the other vocabulary directory.")
    model = data.get("model")
    require(isinstance(model, str) and
            model.replace("\\", "/").rstrip("/").split("/")[-2:] == [f"k{size}", "checkpoint-best"],
            f"model must end in k{size}/checkpoint-best; got {model!r}")
    for field, expected in (("metric", METRIC), ("context_mode", mode),
                            ("split", "val"), ("context_policy", POLICIES[mode]),
                            ("eos_policy", EOS_POLICY)):
        require(data.get(field) == expected,
                f"{field} must be {expected!r}; got {data.get(field)!r}")
    require(type(data.get("context_tokens")) is int and data["context_tokens"] == 512,
            "context_tokens must be 512")
    hashes = data.get("evaluated_text_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(LANGUAGES),
            "evaluated_text_sha256 must contain exactly zh, ko, ja, en")
    for language, value in hashes.items():
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
                f"Invalid SHA-256 for {language}")
    languages = data.get("languages")
    require(isinstance(languages, dict) and set(languages) == set(LANGUAGES),
            "languages must contain exactly zh, ko, ja, en")
    identities = {}
    for language in LANGUAGES:
        routes = languages[language]
        require(isinstance(routes, dict) and {language, "byte"}.issubset(routes),
                f"{language}: native and byte routes are required")
        require(set(routes).issubset({"byte", *LANGUAGES}), f"{language}: unknown route")
        for route, result in routes.items():
            label = f"{language}/{route}"
            require(isinstance(result, dict), f"{label} must be an object")
            for field in ("documents", "source_bytes", "source_characters", "text_tokens", "windows", "eos_count"):
                count(result.get(field), f"{label}.{field}")
            require(result["eos_count"] == result["documents"], f"{label}: EOS count differs from documents")
            bpb = number(result.get("bpb"), f"{label}.bpb")
            nll = number(result.get("text_nll"), f"{label}.text_nll")
            close(bpb, nll / (math.log(2) * result["source_bytes"]), f"{label}.bpb")
            identity = {"sha256": hashes[language].lower(), **{
                field: result[field] for field in ("documents", "source_bytes", "source_characters")}}
            if language in identities:
                require(identity == identities[language], f"{language}: source identity/counts differ between routes")
            else:
                identities[language] = identity
        if mode == "bytes":
            require(len({r["windows"] for r in routes.values()}) == 1,
                    f"{language}: matched-text window counts differ between routes")
    macros = data.get("macro_average")
    require(isinstance(macros, dict), "macro_average must be an object")
    for name, route in (("native_bpb", None), ("byte_bpb", "byte")):
        expected = sum(languages[l][route or l]["bpb"] for l in LANGUAGES) / len(LANGUAGES)
        close(number(macros.get(name), f"macro_average.{name}"), expected, f"macro_average.{name}")
    return identities


def collect(root, output):
    # Read each file once. The validated byte strings are exactly what goes in the ZIP.
    payloads = {}
    records = []
    reference = None
    reference_path = None
    for size in SIZES:
        for mode in POLICIES:
            archive_path = f"k{size}/evaluation-val-{mode}-1000.json"
            path = root / archive_path
            try:
                raw = path.read_bytes()
                data = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=reject_constant)
                identity = validate_file(data, size, mode)
                if reference is None:
                    reference, reference_path = identity, archive_path
                else:
                    require(identity == reference,
                            f"Document hashes/counts or source-byte/character counts differ from {reference_path}")
            except (OSError, ValueError, ValidationError) as exc:
                raise ValidationError(f"{path}: {exc}") from exc
            payloads[archive_path] = raw
            records.append({"archive_path": archive_path, "source_path": str(path.resolve()),
                            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "lexical_size": size, "context_mode": mode, "model": data["model"],
                            "macro_average": data["macro_average"]})
    require(len({r["sha256"] for r in records}) == 4, "Expected four distinct evaluation files")
    manifest = {
        "schema_version": 1,
        "collector": "collect_expanded_evaluations.py",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root.resolve()),
        "files": records,
        "shared_evaluated_text": reference,
        "checks": {
            "all_four_expected_files_present": True,
            "vocabulary_size_matches_directory": True,
            "model_suffix_matches_vocabulary_and_checkpoint_best": True,
            "validation_split_and_512_context_confirmed": True,
            "metric_context_and_eos_policies_confirmed": True,
            "document_hashes_and_source_counts_match_all_routes_and_files": True,
            "required_counts_positive": True,
            "finite_nonnegative_bpb_recomputed_from_text_nll": True,
            "native_and_byte_macro_bpb_recomputed": True,
            "four_distinct_file_hashes": True,
            "original_json_bytes_preserved": True,
        },
        "limitations": [
            "Checks validate supplied evaluation metadata and arithmetic; they do not prove checkpoint or corpus provenance.",
            "Document counts above are actual reported counts; the -1000 filename alone is not proof of 1000 documents.",
        ],
    }
    payloads["collection-manifest.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Exclusive creation protects existing output, including symlinks. Do not create
    # output directories, and remove our own incomplete archive on a write failure.
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, raw in payloads.items():
                    archive.writestr(name, raw)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/100m"), help="Run directory (default: runs/100m)")
    parser.add_argument("--output", type=Path, help="New ZIP path; existing files are never overwritten")
    args = parser.parse_args()
    output = args.output or Path("expanded-validation-verified-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + ".zip")
    try:
        manifest = collect(args.root, output)
    except (ValidationError, OSError, ValueError, OverflowError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Verified four evaluation files; wrote {output.resolve()}")
    print("Documents per language: " + ", ".join(f"{l}={r['documents']}" for l, r in manifest["shared_evaluated_text"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

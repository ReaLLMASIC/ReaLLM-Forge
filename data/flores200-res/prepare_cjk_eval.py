#!/usr/bin/env python3
"""Prepare FLORES CJK with the TED checkpoints' frozen tokenizers and IPA rules."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.cjk_full_eval import (
    FrozenTokenizer, LANGUAGES, RUN_NAMES, SCHEMA_VERSION, file_identity,
    load_meta, sha256_file, write_json,
)

HF_REPO = "muhammadravi251001/restructured-flores200"
COLUMNS = dict(ja="text_jpn_Jpan", ko="text_kor_Hang", zh_cn="text_zho_Hans")


def validate_records(records):
    seen = set()
    if not records:
        raise ValueError("No FLORES records")
    for record in records:
        split, identifier = record["split"], str(record["id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
            raise ValueError("Split names must be safe directory labels")
        if (split, identifier) in seen:
            raise ValueError(f"Duplicate sentence ID: {split}/{identifier}")
        seen.add((split, identifier))
        for language in LANGUAGES:
            text = record["texts"][language]
            if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
                raise ValueError(f"Empty or multiline source sentence: {split}/{identifier}/{language}")
    return records


def download_records(repo_id, revision, output):
    """Pin the Hub revision and preserve the source dataset's split labels.

    A restructured dataset's 'train' is not relabeled as canonical dev/devtest.
    Shard/index IDs are explicit when a source does not expose sentence IDs.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision, timeout=20)
    pinned = info.sha
    files = sorted(path for path in api.list_repo_files(repo_id, repo_type="dataset", revision=pinned)
                   if path.endswith(".parquet"))
    if not files:
        raise ValueError("No Parquet files in the pinned FLORES source")
    records, artifacts = [], []
    for path in files:
        local = hf_hub_download(repo_id, filename=path, repo_type="dataset", revision=pinned,
                                cache_dir=output / "hub_cache", etag_timeout=20)
        table = pq.read_table(local)
        missing = set(COLUMNS.values()) - set(table.column_names)
        if missing:
            raise ValueError(f"Missing CJK fields in {path}: {sorted(missing)}")
        # HF's standard sharded layout: data/<split>-00000-of-00001.parquet.
        match = re.fullmatch(r"(.+)-\d+-of-\d+\.parquet", Path(path).name)
        file_split = match.group(1) if match else Path(path).stem
        artifacts.append(dict(source_path=path, **file_identity(local)))
        for index, row in enumerate(table.to_pylist()):
            split = str(row.get("split") or file_split)
            identifier = row.get("id", row.get("sentence_id"))
            records.append(dict(id=str(identifier) if identifier is not None else f"{path}:{index}",
                                id_kind="source" if identifier is not None else "shard_row",
                                split=split, source_shard=path, source_row=index,
                                texts={lang: row[column] for lang, column in COLUMNS.items()}))
    metadata = dict(repo_id=repo_id, requested_revision=revision, revision=pinned,
                    source_files=artifacts, split_policy="preserve source labels; no inferred canonical split",
                    card_data=info.card_data.to_dict() if info.card_data else None)
    return validate_records(records), metadata


def ipa_command(language, source, destination, stats, workers):
    utils = ROOT / "data/template/utils"
    if language == "ja":
        return [sys.executable, str(utils / "ja2ipa.py"), str(source), str(destination),
                "--text_output", "--text_no_sentence", "--text_field", "unspaced_ipa",
                "--stats_json", str(stats)]
    if language == "ko":
        return [sys.executable, str(utils / "espeak2ipa.py"), str(source), "--lang", "ko",
                "--mode", "text", "--output_file", str(destination), "--no-wrapper",
                "--multithread", "--workers", str(workers), "--stats_json", str(stats)]
    if language == "zh_cn":
        return [sys.executable, str(utils / "zh_to_ipa.py"), str(source), str(destination),
                "--input_type", "text", "--no-wrapper", "--stats_json", str(stats)]
    raise ValueError(f"Unknown language: {language}")


def convert_ipa(language, texts, output, workers):
    source, destination = output / f"{language}.original.txt", output / f"{language}.ipa.txt"
    stats = output / f"{language}.ipa.stats.json"
    source.write_text("".join(text + "\n" for text in texts), encoding="utf-8")
    command = ipa_command(language, source, destination, stats, workers)
    with (output / f"{language}.ipa.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    converted = destination.read_text(encoding="utf-8").splitlines()
    if len(converted) != len(texts) or any(not text.strip() for text in converted):
        raise ValueError(f"IPA conversion changed sentence alignment or emitted empty text: {language}")
    provenance = dict(command=command, converter=file_identity(command[1]),
                      input=file_identity(source), output=file_identity(destination),
                      stats=file_identity(stats), diagnostics=json.loads(stats.read_text()))
    packages = {"ja": ("pykakasi",), "ko": (), "zh_cn": ("dragonmapper", "jieba")}[language]
    provenance["package_versions"] = {package: version(package) for package in packages}
    if language == "ko":
        provenance["espeak_version"] = subprocess.check_output(["espeak-ng", "--version"], text=True).strip()
    return converted, provenance


def encode_job(name, tokenizer, meta_sha, texts, identifiers, output, split, scope):
    job_id = f"flores_{split}_{scope}"
    text = "".join(sentence + "\n" for sentence in texts)
    selection_path = output / f"{job_id}.ids.json"
    write_json(selection_path, identifiers)
    job = dict(id=job_id, corpus="flores", split=split, scope=scope,
               meta_sha256=meta_sha, source_utf8_bytes=len(text.encode("utf-8")),
               sentences=len(texts), selection=file_identity(selection_path))
    unknown = tokenizer.unknown_characters(text)
    if unknown:
        job.update(status="unencodable_oov", unknown_characters=unknown,
                   unknown_occurrences=sum(unknown.values()), source_tokens=None)
        return job
    ids = tokenizer.encode(text)
    if tokenizer.decode_bytes(ids) != text.encode("utf-8"):
        raise ValueError(f"Frozen tokenizer failed byte-exact roundtrip: {name}/{job_id}")
    dtype = "uint16" if len(tokenizer.pieces) <= 65536 else "uint32"
    destination = output / f"{job_id}.bin"
    np.asarray(ids, dtype=dtype).tofile(destination)
    job.update(status="ready" if len(ids) > 1 else "no_scorable_tokens",
               tokens=file_identity(destination), dtype=dtype, source_tokens=len(ids))
    return job


def prepare_records(records, data_root, output, workers=4, converter=convert_ipa):
    """All rows remain in the full result; paired subsets are explicit additions."""
    validate_records(records)
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    runs = {name: [] for name in RUN_NAMES}
    provenance, coverage = {}, {}
    for split, rows in sorted(by_split.items()):
        split_output = output / split
        split_output.mkdir(parents=True, exist_ok=True)
        provenance[split], coverage[split] = {}, {}
        for language in LANGUAGES:
            original = [row["texts"][language] for row in rows]
            converted, details = converter(language, original, split_output, workers)
            if len(converted) != len(rows) or any(not text.strip() for text in converted):
                raise ValueError("IPA converter did not preserve nonempty sentence alignment")
            provenance[split][language] = details
            for representation, sentences in (("original", original), ("ipa", converted)):
                pair = f"{language}_{representation}"
                char_name, bpe_name = f"{pair}_char", f"{pair}_char_bpe"
                char_meta_path, bpe_meta_path = [data_root / name / "meta.pkl" for name in (char_name, bpe_name)]
                char_meta, bpe_meta = [load_meta(path) for path in (char_meta_path, bpe_meta_path)]
                char, bpe = FrozenTokenizer(char_meta), FrozenTokenizer(bpe_meta)
                supported = [not char.unknown_characters(text + "\n") for text in sentences]
                unknown = Counter()
                for sentence in sentences:
                    unknown.update(char.unknown_characters(sentence + "\n"))
                coverage[split][pair] = dict(total_sentences=len(rows),
                    encodable_sentences=sum(supported), unknown_characters=dict(unknown),
                    unknown_occurrences=sum(unknown.values()),
                    supported_sentence_fraction=sum(supported) / len(rows),
                    supported_text_utf8_bytes=sum(len((text + "\n").encode("utf-8"))
                                                 for text, keep in zip(sentences, supported) if keep),
                    total_text_utf8_bytes=sum(len((text + "\n").encode("utf-8")) for text in sentences))
                identifiers = [str(row["id"]) for row in rows]
                for name, tokenizer, meta_path in ((char_name, char, char_meta_path), (bpe_name, bpe, bpe_meta_path)):
                    destination = split_output / name
                    destination.mkdir(exist_ok=True)
                    runs[name].append(encode_job(name, tokenizer, sha256_file(meta_path), sentences,
                                                 identifiers, destination, split, "full"))
                    if not all(supported):
                        subset = [text for text, keep in zip(sentences, supported) if keep]
                        subset_ids = [identifier for identifier, keep in zip(identifiers, supported) if keep]
                        runs[name].append(encode_job(name, tokenizer, sha256_file(meta_path), subset,
                                                     subset_ids, destination, split, "jointly_encodable"))
    return dict(schema_version=SCHEMA_VERSION, runs=runs, coverage=coverage, ipa=provenance,
                text_policy="source order, newline after each sentence; stream within split; no cross-split context",
                oov_policy="full Char unavailable on OOV; additional paired subset; no substitution or vocabulary growth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Original TED directory containing all 12 meta.pkl files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory outside the Git checkout")
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--revision", default="main", help="Resolved to an immutable Hub revision before download")
    parser.add_argument("--records-jsonl", type=Path, help="Offline input: {id, split, texts: {ja, ko, zh_cn}} per line")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a fresh output directory; existing source data will not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.records_jsonl:
            records = [json.loads(line) for line in args.records_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            source = dict(offline_records=file_identity(args.records_jsonl),
                          revision=None, split_policy="preserve supplied labels")
        else:
            records, source = download_records(args.repo_id, args.revision, args.output_dir)
        validate_records(records)
        source_path = args.output_dir / "source_records.jsonl"
        source_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        write_json(args.output_dir / "source.json", source)
        manifest = prepare_records(records, args.data_root, args.output_dir, args.workers)
        manifest["source"] = source
        manifest["source_records"] = file_identity(source_path)
        write_json(args.output_dir / "manifest.json", manifest)
        print(args.output_dir / "manifest.json")
    except Exception as error:
        write_json(args.output_dir / "failure.json", dict(error_type=type(error).__name__, error=str(error)))
        parser.exit(1, f"FLORES preparation failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()

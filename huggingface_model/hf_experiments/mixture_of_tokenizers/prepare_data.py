"""Prepare bounded, document-disjoint corpus samples before tokenizer training.

Chinese t2s conversion is an explicit corpus preparation policy. The tokenizer
itself never normalizes text. JSONL strings preserve all whitespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

LANGUAGES = ("zh", "ko", "ja", "en")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_digest(digest: str, seed: int, val_fraction: float,
                     test_fraction: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:16], 16) / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def iter_local(path: str | Path) -> Iterable[dict]:
    """JSONL documents, or one text-file line (including its newline) per doc."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        for index, line in enumerate(handle):
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"{path}:{index + 1}: expected a JSON object with string text")
                yield row
            else:
                yield {"text": line, "id": f"{path.name}:{index + 1}"}


def demo_documents(language: str, count: int = 512) -> Iterable[dict]:
    """Synthetic mechanics-only corpus. Never a language quality benchmark."""
    phrases = {
        "zh": ["今天我们学习语言模型。", "小河流过绿色森林。", "科学研究需要清晰记录。", "机器可以预测下一个符号。", "春天的花园有许多颜色。", "共享字节保留原始文字。"],
        "ko": ["오늘 우리는 언어 모델을 공부합니다.", "작은 강물이 푸른 숲을 지나갑니다.", "과학 연구에는 정확한 기록이 필요합니다.", "기계는 다음 문자를 예측합니다.", "봄의 정원에는 여러 색깔이 있습니다.", "공유 바이트는 원래 글을 보존합니다."],
        "ja": ["今日は言語モデルを学びます。", "小さな川が緑の森を流れます。", "科学の研究には正確な記録が必要です。", "機械は次の文字を予測します。", "春の庭には様々な色があります。", "共有バイトで元の文字を保存します。"],
        "en": ["Today we study language models.", "A small river flows through a green forest.", "Scientific research needs clear records.", "A machine predicts the next symbol.", "The spring garden has many colors.", "Shared bytes preserve the original text."],
    }[language]
    for index in range(count):
        # Unequal repeated contexts supply actual BPE merges, while the index
        # makes each source document unique without special tokenizer tokens.
        text = "  ".join(phrases[(index // (6**k)) % 6] for k in range(4))
        text += f"\n{index:06d}\t0123456789\r\n"
        if index % 11 == 0:
            text += "Emoji🙂 and literal <0x41> are retained.\n"
        yield {"text": text, "id": f"synthetic-{language}-{index}"}


def _source_rows(language: str, config: dict, demo: bool) -> tuple[Iterable[dict], dict]:
    if demo:
        count = int(config.get("max_documents_per_language", 512))
        return demo_documents(language, count), {"kind": "synthetic_mechanics_only", "language": language}
    if config.get("source", "hf") == "local":
        path = Path(config["local_files"][language]).resolve()
        return iter_local(path), {"kind": "local", "path": str(path), "sha256": file_sha256(path)}
    from datasets import load_dataset
    from huggingface_hub import HfApi

    source = dict(config["sources"][language])
    revision = source.get("revision", "main")
    # Resolve moving refs once and record the immutable revision actually read.
    resolved = HfApi().dataset_info(source["path"], revision=revision).sha
    if not resolved:
        raise RuntimeError(f"Could not resolve dataset revision for {source['path']}")
    rows = load_dataset(source["path"], source.get("name"),
                        revision=resolved, split=source.get("split", "train"), streaming=True)
    buffer_size = int(config.get("shuffle_buffer", 1000))
    if buffer_size > 1:
        rows = rows.shuffle(seed=int(config.get("seed", 1337)), buffer_size=buffer_size)
    column = source.get("text_column", "text")
    source.update(kind="huggingface", requested_revision=revision, resolved_revision=resolved)

    def documents():
        for row in rows:
            text = row.get(column)
            if not isinstance(text, str):
                raise ValueError(f"Dataset {source['path']} has no string column {column!r}")
            yield {"text": text, "id": row.get("id"), "url": row.get("url")}

    return documents(), source


def prepare(config: dict, output: str | Path | None = None, demo: bool = False) -> dict:
    config = dict(config.get("data", config))
    output = Path(output or config.get("output", "data/prepared"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; use a new directory to preserve frozen splits")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 1337))
    val_fraction = float(config.get("val_fraction", 0.01))
    test_fraction = float(config.get("test_fraction", 0.01))
    if demo:
        # Documented larger smoke splits avoid fragile empty validation sets.
        val_fraction = test_fraction = 0.1
        config.setdefault("max_documents_per_language", 512)
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than one")
    max_docs = int(config.get("max_documents_per_language", 200000))
    max_bytes = int(config.get("max_bytes_per_language", 500000000))
    max_scanned = int(config.get("max_scanned_documents_per_language", max_docs * 20))
    if min(max_docs, max_bytes, max_scanned) <= 0:
        raise ValueError("Document and byte limits must be positive")
    conversion = config.get("zh_conversion", "t2s")
    if conversion not in {"none", "t2s"}:
        raise ValueError("zh_conversion must be 't2s' or 'none'")
    converter = None
    if conversion == "t2s":
        from opencc import OpenCC
        converter = OpenCC("t2s")
    exclusion_hashes: set[str] = set()
    for path in config.get("exclusion_hash_files", []):
        exclusion_hashes.update(x.strip() for x in Path(path).read_text().splitlines() if x.strip())
    seen: set[str] = set()
    manifest = {
        "format_version": 1, "synthetic": demo, "seed": seed,
        "split_policy": "seeded_sha256_of_prepared_document", "val_fraction": val_fraction,
        "test_fraction": test_fraction, "zh_conversion": conversion,
        "normalization": "Chinese OpenCC t2s only; otherwise preserve exact text",
        "deduplication": "exact prepared-text SHA256 across all four languages",
        "benchmark_decontamination": "hash exclusions only" if exclusion_hashes else "not performed",
        "exclusion_hash_count": len(exclusion_hashes), "languages": {},
    }
    for language in LANGUAGES:
        rows, source = _source_rows(language, config, demo)
        directory = output / language
        directory.mkdir()
        handles = {s: (directory / f"{s}.jsonl").open("w", encoding="utf-8", newline="\n") for s in ("train", "val", "test")}
        stats = {s: {"documents": 0, "utf8_bytes": 0} for s in handles}
        scanned = accepted = prepared_bytes = duplicates = excluded = converted = 0
        try:
            for row in rows:
                if scanned >= max_scanned or accepted >= max_docs:
                    break
                scanned += 1
                original = row["text"]
                text = converter.convert(original) if language == "zh" and converter else original
                encoded = text.encode("utf-8")
                if not encoded:
                    continue
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                if digest in exclusion_hashes:
                    excluded += 1
                    continue
                if prepared_bytes + len(encoded) > max_bytes:
                    break
                seen.add(digest)
                split = split_for_digest(digest, seed, val_fraction, test_fraction)
                record = {"id": str(row.get("id") or digest), "text": text,
                          "sha256": digest, "raw_sha256": sha256_text(original)}
                if row.get("url"):
                    record["url"] = row["url"]
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                prepared_bytes += len(encoded)
                converted += text != original
                stats[split]["documents"] += 1
                stats[split]["utf8_bytes"] += len(encoded)
        finally:
            for handle in handles.values():
                handle.close()
        for split in stats:
            stats[split]["sha256"] = file_sha256(directory / f"{split}.jsonl")
        manifest["languages"][language] = {
            "source": source, "scanned": scanned, "accepted": accepted,
            "duplicate_documents_skipped": duplicates, "excluded_documents_skipped": excluded,
            "chinese_documents_converted": converted, "splits": stats,
        }
        print(f"{language}: {accepted} documents, {prepared_bytes:,} UTF-8 bytes; "
              f"train/val/test = {stats['train']['documents']}/{stats['val']['documents']}/{stats['test']['documents']}")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    empty = [(lang, split) for lang, info in manifest["languages"].items()
             for split, stats in info["splits"].items() if stats["documents"] == 0]
    if empty:
        raise ValueError(f"Prepared empty splits {empty}; increase corpus limits or validation fractions. "
                         f"Partial output remains in {output}; inspect manifest before rerunning in a new directory.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/4090.yaml")
    parser.add_argument("--output")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--demo", action="store_true", help="Synthetic CPU smoke data, never a quality benchmark")
    args = parser.parse_args()
    import yaml
    config_path = Path(args.config)
    if not config_path.exists() and not args.demo:
        parser.error(f"Configuration file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {"data": {}}
    if args.max_documents is not None:
        config.setdefault("data", {})["max_documents_per_language"] = args.max_documents
    elif args.demo:
        config.setdefault("data", {})["max_documents_per_language"] = 512
    prepare(config, output=args.output, demo=args.demo)


if __name__ == "__main__":
    main()

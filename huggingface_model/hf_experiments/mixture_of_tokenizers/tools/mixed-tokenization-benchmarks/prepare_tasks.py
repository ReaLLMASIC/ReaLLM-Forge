#!/usr/bin/env python3
"""Prepare immutable, short multilingual commonsense evaluation inputs.

Only Python's standard library is required. These fixed zero-shot continuation
templates define this experiment's protocol, not an official leaderboard score.
PIQA has public validation labels only; both split sets deliberately use them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


SCHEMA_VERSION = 1
TEMPLATE_VERSION = "continuation-v1"
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
XCOPA_REVISION = "e2e9d7f105a758ee869cd13e0fe251ef93a29840"
KOBEST_REVISION = "a5ea15e3ac77ed694b79f6204eb31889a2ba989f"
JGLUE_REVISION = "6f071c09316baae89c3d083a90985b4b1cb9968c"
TASK_ORDER = ("xcopa_zh", "kobest_copa_ko", "jcommonsenseqa_ja", "piqa_en")
LANGUAGES = dict(zip(TASK_ORDER, ("zh", "ko", "ja", "en")))
EXPECTED_COUNTS = {
    "xcopa_zh": {"validation": 100, "test": 500},
    "kobest_copa_ko": {"validation": 500, "test": 1000},
    "jcommonsenseqa_ja": {"validation": 1119, "test": 1118},
    "piqa_en": {"validation": 1838},
}


class PreparationError(ValueError):
    """Input data or immutable output does not match this protocol."""


@dataclass(frozen=True)
class Source:
    task: str
    split: str
    filename: str
    url: str
    sha256: str
    revision: str | None
    license: str
    license_url: str
    role: str = "examples"

    @property
    def cache_path(self) -> str:
        return f"sources/{self.task}/{self.filename}"


def sources_for(split_set: str) -> list[Source]:
    if split_set not in {"dev", "final"}:
        raise PreparationError(f"Unknown split set: {split_set}")
    final = split_set == "final"
    split = "test" if final else "validation"
    zh_file = "test.zh.jsonl" if final else "val.zh.jsonl"
    ja_file = "test-v1.3.json" if final else "valid-v1.3.json"
    return [
        Source(
            "xcopa_zh", split, zh_file,
            f"https://raw.githubusercontent.com/cambridgeltl/xcopa/{XCOPA_REVISION}/data/zh/{zh_file}",
            "c9b42590399214b9f066adacac2465d7286f5e590181fe6bfd7f6139531b83d4" if final else
            "8f638466c196342104bdbe9276e7d91fe0abcc044e9a9ad715f30c8ad618bcdd",
            XCOPA_REVISION, "CC-BY-4.0",
            f"https://github.com/cambridgeltl/xcopa/blob/{XCOPA_REVISION}/LICENSE.md",
        ),
        Source(
            "kobest_copa_ko", split, f"{split}.jsonl",
            f"https://huggingface.co/datasets/skt/kobest_v1/resolve/{KOBEST_REVISION}/copa/{split}.jsonl",
            "2912f4055e3fb170b94d0c25fef09a2015b5c61d99a1b6355d276d1099bd9f79" if final else
            "e7ed6cc07bfab320bf5643791d69cc4fc6f23a24ad38132f98047737f2687cc0",
            KOBEST_REVISION, "CC-BY-SA-4.0",
            f"https://huggingface.co/datasets/skt/kobest_v1/blob/{KOBEST_REVISION}/README.md",
        ),
        Source(
            "jcommonsenseqa_ja", split, ja_file,
            f"https://raw.githubusercontent.com/yahoojapan/JGLUE/{JGLUE_REVISION}/datasets/jcommonsenseqa-v1.3/{ja_file}",
            "ba91ef3907c91436a5e44fa9fccbbfa3df507577e2c681f4d7e39bd987f0186d" if final else
            "0d8d76f3bfa0d174866939882faccdd01fbc2bcd5a76c43748ba0c40a7b3b8d4",
            JGLUE_REVISION, "CC-BY-SA-4.0",
            f"https://github.com/yahoojapan/JGLUE/blob/{JGLUE_REVISION}/LICENSE",
        ),
        Source(
            "piqa_en", "validation", "valid.jsonl",
            "https://yonatanbisk.com/piqa/data/valid.jsonl",
            "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d",
            None, "AFL-3.0", "https://yonatanbisk.com/piqa/data/",
        ),
        Source(
            "piqa_en", "validation", "valid-labels.lst",
            "https://yonatanbisk.com/piqa/data/valid-labels.lst",
            "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb",
            None, "AFL-3.0", "https://yonatanbisk.com/piqa/data/", "labels",
        ),
    ]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(source: Source, output: Path, offline: bool) -> bytes:
    """Read or download one exact file; never replace an existing bad cache."""
    path = output / source.cache_path
    if path.is_symlink():
        raise PreparationError(f"Source cache must not be a symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise PreparationError(f"Source cache is not a file: {path}")
        data = path.read_bytes()
    elif offline:
        raise PreparationError(f"Offline source cache is missing: {path}")
    else:
        request = urllib.request.Request(source.url, headers={"User-Agent": "mixed-tokenization-benchmarks/1"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
        except Exception as exc:
            raise PreparationError(f"Could not download {source.url}: {exc}") from exc
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise PreparationError(f"Source exceeds download size limit: {source.url}")
    actual = sha256(data)
    if actual != source.sha256:
        raise PreparationError(
            f"SHA-256 mismatch for {path}: expected {source.sha256}, got {actual}. "
            "Existing files are never overwritten; inspect the source and use a fresh output directory."
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
    return data


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PreparationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_jsonl(payload: bytes) -> list[dict]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise PreparationError("Source is not valid UTF-8") from exc
    if not lines:
        raise PreparationError("Empty JSONL source")
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise PreparationError(f"Blank JSONL row at line {number}")
        try:
            item = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        except (ValueError, TypeError) as exc:
            raise PreparationError(f"Invalid JSONL at line {number}: {exc}") from exc
        if not isinstance(item, dict):
            raise PreparationError(f"JSONL row {number} must be an object")
        result.append(item)
    return result


def _text(item: dict, key: str) -> str:
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{key} must be a nonempty string")
    return value


def _integer(value: object, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise PreparationError(f"Invalid {name}: {value!r}")
    return value


def normalize_task(
    task: str,
    split: str,
    payload: bytes,
    labels_payload: bytes | None = None,
    expected_count: int | None = None,
) -> list[dict]:
    if task not in TASK_ORDER or split not in EXPECTED_COUNTS[task]:
        raise PreparationError(f"Unknown task/split: {task}/{split}")
    items = parse_jsonl(payload)
    if expected_count is not None and len(items) != expected_count:
        raise PreparationError(f"{task}/{split}: expected {expected_count} rows, got {len(items)}")
    labels = None
    if task == "piqa_en":
        if labels_payload is None:
            raise PreparationError("PIQA requires its separate aligned labels file")
        try:
            raw_labels = labels_payload.decode("utf-8").splitlines()
        except UnicodeError as exc:
            raise PreparationError("PIQA labels are not UTF-8") from exc
        if len(raw_labels) != len(items) or any(x not in {"0", "1"} for x in raw_labels):
            raise PreparationError("PIQA labels must be one 0/1 per example in original order")
        labels = [int(x) for x in raw_labels]
    elif labels_payload is not None:
        raise PreparationError(f"Unexpected separate labels for {task}")
    expected_keys = {
        "xcopa_zh": {"premise", "choice1", "choice2", "question", "label", "idx", "changed"},
        "kobest_copa_ko": {"premise", "question", "alternative_1", "alternative_2", "label"},
        "jcommonsenseqa_ja": {"q_id", "question", "choice0", "choice1", "choice2", "choice3", "choice4", "label"},
        "piqa_en": {"goal", "sol1", "sol2"},
    }[task]
    output = []
    seen = set()
    for row_index, item in enumerate(items):
        if set(item) != expected_keys:
            raise PreparationError(f"{task} row {row_index}: unexpected schema {sorted(item)}")
        source_id = row_index
        if task == "xcopa_zh":
            source_id = _integer(item["idx"], "idx", 0)
            if type(item["changed"]) is not bool:
                raise PreparationError("XCOPA changed must be boolean")
            connector = {"cause": "因为", "effect": "所以"}.get(item["question"])
            if connector is None:
                raise PreparationError(f"Unknown XCOPA question: {item['question']!r}")
            premise = _text(item, "premise").rstrip("。.!！?？")
            if not premise.strip():
                raise PreparationError("XCOPA premise contains only punctuation")
            prompt = premise + "，" + connector
            choices = [_text(item, "choice1"), _text(item, "choice2")]
        elif task == "kobest_copa_ko":
            connector = {"원인": "왜냐하면", "결과": "그래서"}.get(item["question"])
            if connector is None:
                raise PreparationError(f"Unknown KoBEST question: {item['question']!r}")
            prompt = _text(item, "premise") + " " + connector
            choices = [" " + _text(item, "alternative_1"), " " + _text(item, "alternative_2")]
        elif task == "jcommonsenseqa_ja":
            source_id = _integer(item["q_id"], "q_id", 0)
            prompt = "質問：" + _text(item, "question") + "\n答え："
            choices = [_text(item, f"choice{i}") for i in range(5)]
        else:
            prompt = "Question: " + _text(item, "goal") + "\nAnswer:"
            choices = [" " + _text(item, "sol1"), " " + _text(item, "sol2")]
        answer = labels[row_index] if labels is not None else item["label"]
        answer = _integer(answer, "answer index", 0, len(choices) - 1)
        identifier = f"{task}:{split}:{source_id}"
        if identifier in seen:
            raise PreparationError(f"Duplicate source ID: {identifier}")
        seen.add(identifier)
        output.append({
            "id": identifier, "task": task, "language": LANGUAGES[task], "split": split,
            "prompt": prompt, "choices": choices, "answer_index": answer,
        })
    return output


def write_or_verify_artifact(path: Path, data: bytes, offline: bool) -> None:
    if path.is_symlink():
        raise PreparationError(f"Output must not be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise PreparationError(
                f"Existing artifact differs from verified inputs/protocol: {path}. "
                "It was not overwritten; use a fresh output directory."
            )
    elif offline:
        raise PreparationError(f"Offline artifact is missing: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)


def prepare(output: Path, split_set: str = "final", offline: bool = False) -> dict:
    sources = sources_for(split_set)
    source_payloads = {(s.task, s.role): read_source(s, output, offline) for s in sources}
    rows = []
    task_metadata = []
    for task in TASK_ORDER:
        source = next(s for s in sources if s.task == task and s.role == "examples")
        task_rows = normalize_task(
            task, source.split, source_payloads[(task, "examples")],
            source_payloads.get((task, "labels")), EXPECTED_COUNTS[task][source.split],
        )
        rows.extend(task_rows)
        task_metadata.append({
            "task": task, "language": LANGUAGES[task], "split": source.split,
            "count": len(task_rows), "num_choices": len(task_rows[0]["choices"]),
            "chance_accuracy": 1 / len(task_rows[0]["choices"]),
        })
    task_bytes = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "template_version": TEMPLATE_VERSION,
        "split_set": split_set, "record_count": len(rows),
        "tasks_path": "tasks.jsonl", "tasks_sha256": sha256(task_bytes),
        "protocol": "Custom zero-shot continuation likelihood adapters; not official leaderboard scores.",
        "piqa_split_note": "PIQA uses public labeled validation in both dev and final; it is not an independent final test.",
        "pinning_note": "Git/HF sources use commit revisions plus SHA-256. PIQA author-hosted assets have no commit revision and are pinned by exact SHA-256.",
        "templates": {
            "xcopa_zh": "premise.rstrip('。.!！?？') + '，' + {'cause':'因为','effect':'所以'}; choices unchanged",
            "kobest_copa_ko": "premise + ' ' + {'원인':'왜냐하면','결과':'그래서'}; each choice prefixed by one ASCII space",
            "jcommonsenseqa_ja": "'質問：' + question + '\\n答え：'; choice0..choice4 unchanged",
            "piqa_en": "'Question: ' + goal + '\\nAnswer:'; each solution prefixed by one ASCII space",
        },
        "tasks": task_metadata,
        "sources": [{
            "task": s.task, "split": s.split, "role": s.role, "url": s.url,
            "revision": s.revision, "cache_path": s.cache_path,
            "sha256": s.sha256, "bytes": len(source_payloads[(s.task, s.role)]),
            "license": s.license, "license_url": s.license_url,
        } for s in sources],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Verify both existing artifacts before writing either missing one.
    for path, data in ((output / "tasks.jsonl", task_bytes), (output / "manifest.json", manifest_bytes)):
        if path.exists() or path.is_symlink() or offline:
            write_or_verify_artifact(path, data, offline)
    write_or_verify_artifact(output / "tasks.jsonl", task_bytes, offline)
    write_or_verify_artifact(output / "manifest.json", manifest_bytes, offline)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/commonsense-v1"))
    parser.add_argument("--split-set", choices=("dev", "final"), default="final")
    parser.add_argument("--offline", action="store_true", help="Require and verify all existing source and output files; never download or write.")
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.output, args.split_set, args.offline)
    except (PreparationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    action = "Verified" if args.offline else "Prepared"
    print(f"{action} {manifest['record_count']} questions: {args.output / 'tasks.jsonl'}")
    for task in manifest["tasks"]:
        print(f"  {task['task']}: {task['count']} {task['split']} questions")
    print(f"Manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

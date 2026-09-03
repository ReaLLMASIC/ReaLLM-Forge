import csv
import json
from pathlib import Path

import torch

import cjk_translation_pipeline as cjk


def write_tri(path: Path, source: str, rows):
    path.write_text(json.dumps({
        "source": source,
        "languages": ["kor_Hang", "zho_Hans", "jpn_Jpan"],
        "records": [{"translations": {"kor_Hang": ko, "zho_Hans": zh, "jpn_Jpan": ja}} for zh, ja, ko in rows],
    }, ensure_ascii=False), encoding="utf-8")


def test_load_tri_schema_skip_and_dedup_id(tmp_path):
    path = tmp_path / "pairs.json"
    write_tri(path, "toy", [("中", "日", "한"), ("", "x", "y")])
    rows, stats = cjk.load_tri_records(path)
    assert len(rows) == 1
    assert rows[0]["record_id"].startswith("triad-")
    assert rows[0]["translations_raw"] == {"zh": "中", "ja": "日", "ko": "한"}
    assert stats["skipped"]["empty_zh"] == 1


def test_split_is_record_level_deterministic():
    rid = "triad-abc"
    assert cjk.split_for_record(rid) == cjk.split_for_record(rid)


def test_expansion_has_six_minimal_examples():
    rec = {"record_id": "triad-x", "translations_raw": {"zh": "中", "ja": "日", "ko": "한"}}
    rows = cjk.iter_directed([rec], "train", "tiktoken")
    assert len(rows) == 6
    assert set(rows[0]) == {"id", "direction", "prompt", "completion"}
    assert rows[0]["direction"] in cjk.DIRECTIONS
    assert not rows[0]["prompt"].endswith(rows[0]["completion"])


def test_benchmark_inspect_and_adapt_read_only(tmp_path):
    bench = tmp_path / "benchmarks_easy.csv"
    with bench.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "Focus / Type", "English", "Chinese (Simplified)", "Japanese (Natural Polite)", "Korean (Natural Polite)"])
        w.writerow(["1", "Greeting", "hello", "你好", "こんにちは", "안녕하세요"])
    before = bench.read_bytes()
    schema = cjk.inspect_benchmark(bench)
    assert schema["compatible_with_cjk_translation_task"]
    out = tmp_path / "adapted.jsonl"
    cjk.cmd_adapt_benchmark(type("A", (), {"benchmark": str(bench), "out": str(out)})())
    assert bench.read_bytes() == before
    rows = cjk.read_jsonl(out)
    assert len(rows) == 6
    assert rows[0]["benchmark_row_id"] == "1"


def test_ipa_benchmark_render_uses_ipa_prompt_and_reference(monkeypatch):
    class FakeIpa:
        available = True
        error = None

        def convert(self, lang, text):
            return f"{lang}:{text}:ipa"

    monkeypatch.setattr(cjk, "IpaConverter", FakeIpa)
    rows = cjk.render_benchmark_examples_for_family([{
        "id": "bench:zh_to_ja",
        "direction": "zh_to_ja",
        "src_lang": "zh",
        "tgt_lang": "ja",
        "prompt": cjk.make_prompt("zh", "ja", "你好"),
        "completion": "こんにちは",
    }], "ipa")
    assert rows[0]["prompt"] == cjk.make_prompt("zh", "ja", "zh:你好:ipa")
    assert rows[0]["completion"] == "ja:こんにちは:ipa"
    assert rows[0]["orthographic_completion"] == "こんにちは"


def test_generate_until_stop_decodes_accumulated_tokens():
    class Config:
        block_size = 16

    class FakeModel:
        config = Config()

        def __call__(self, idx):
            logits = torch.full((1, idx.size(1), 4), -100.0)
            next_id = 1 if idx.size(1) == 1 else 2
            logits[:, -1, next_id] = 100.0
            return logits, None

    decode_calls = []

    def decode(ids):
        decode_calls.append(list(ids))
        return "\n" if ids == [1, 2] else ""

    _, text = cjk.generate_until_stop(
        FakeModel(),
        torch.tensor([[0]], dtype=torch.long),
        max_new_tokens=4,
        decode=decode,
        temperature=1.0,
        top_k=1,
    )
    assert text == "\n"
    assert decode_calls == [[1], [1, 2]]


def test_metrics_and_score_shape():
    assert cjk.char_f1("你好", "你好") == 1.0
    assert cjk.char_f1("", "你好") == 0.0
    result = cjk.score_predictions([
        {"direction": "zh_to_ja", "prediction": "あ", "reference": "あ"},
        {"direction": "ja_to_zh", "prediction": "中", "reference": "中"},
    ], "dev", "dev", "ckpt.pt")
    assert result["macro_avg"]["f1"] == 1.0
    assert result["macro_avg"]["exact_match"] == 1.0


def test_required_score_columns_present():
    assert "model_variant" in cjk.SCORE_COLUMNS
    assert "chrf" in cjk.SCORE_COLUMNS
    assert "is_best_within_family" in cjk.SCORE_COLUMNS

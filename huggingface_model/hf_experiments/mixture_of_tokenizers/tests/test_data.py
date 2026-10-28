import hashlib
import json

import torch

from data_pipeline import LanguageWindowDataset
from prepare_data import LANGUAGES, prepare


class ByteFixtureTokenizer:
    """A byte-only fixture suffices to verify sampling independently of BPE."""
    lexical_size = 16
    vocab_size = 256 + 4 * lexical_size + 15
    fingerprint = "fixture-byte-tokenizer-v1"

    def encode_route(self, text, route):
        return list(text.encode("utf-8"))

    def bos_id(self, route):
        return 256 + 4 * self.lexical_size + 3 * int(route)

    def eos_id(self, route):
        return self.bos_id(route) + 1

    def pad_id(self, route):
        return self.bos_id(route) + 2

    def token_byte_payload(self, index):
        return bytes([index]) if index < 256 else b""


def write_corpus(path, texts):
    for lang in LANGUAGES:
        (path / lang).mkdir(parents=True)
        for split in ("train", "val", "test"):
            with (path / lang / f"{split}.jsonl").open("w") as handle:
                for text in texts:
                    handle.write(json.dumps({"text": text}) + "\n")


def test_document_windows_predict_each_target_and_only_true_eos(tmp_path):
    texts = ["abcdefghi", "x", "two docs must not touch"]
    write_corpus(tmp_path / "data", texts)
    tok = ByteFixtureTokenizer()
    ds = LanguageWindowDataset(tmp_path / "data", tok, 5, cache_dir=tmp_path / "cache")
    pair = ds._cache(1, "zh")
    batches = [ds.get_batch("zh", 1) for _ in range(pair["total_windows"])]
    stats = [b["batch_stats"] for b in batches]
    assert sum(x["text_bytes"] for x in stats) == sum(len(t.encode()) for t in texts)
    assert sum(x["eos_targets"] for x in stats) == len(texts)
    assert sum(x["predicted_tokens"] for x in stats) == sum(len(t.encode()) + 1 for t in texts)
    for batch in batches:
        assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
        assert batch["labels"][0, 0] == -100
        ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()
        assert tok.eos_id(1) not in ids[:-1]


def test_byte_balance_resume_and_cache_dtype(tmp_path):
    write_corpus(tmp_path / "data", ["a long document " * 10, "different example"])
    tok = ByteFixtureTokenizer()
    ds = LanguageWindowDataset(tmp_path / "data", tok, 8, seed=77, cache_dir=tmp_path / "cache")
    first = ds.get_batch("byte", 4)
    assert {k: v["sequences"] for k, v in first["batch_stats"]["languages"].items()} == dict.fromkeys(LANGUAGES, 1)
    state = ds.state_dict()
    expected = ds.get_batch("byte", 7)
    resumed = LanguageWindowDataset(tmp_path / "data", tok, 8, seed=77, cache_dir=tmp_path / "cache")
    resumed.load_state_dict(state)
    observed = resumed.get_batch("byte", 7)
    for key in ("input_ids", "labels", "attention_mask", "tokenizer_ids"):
        assert torch.equal(expected[key], observed[key])
    assert observed["batch_stats"] == expected["batch_stats"]
    assert all(p["tokens"].dtype.itemsize == 1 for p in resumed._pairs.values())


def test_byte_cache_is_shared_between_vocabulary_sizes(tmp_path):
    write_corpus(tmp_path / "data", ["中文\nshared raw bytes"])
    small = ByteFixtureTokenizer()
    large = ByteFixtureTokenizer()
    large.lexical_size = 32
    large.vocab_size = 256 + 4 * large.lexical_size + 15
    large.fingerprint = "different-k-tokenizer"
    left = LanguageWindowDataset(tmp_path / "data", small, 8, cache_dir=tmp_path / "cache")
    right = LanguageWindowDataset(tmp_path / "data", large, 8, cache_dir=tmp_path / "cache")
    a, b = left._cache(0, "zh"), right._cache(0, "zh")
    assert a["digest"] == b["digest"]
    assert a["tokens"].filename == b["tokens"].filename
    assert a["tokens"].dtype.itemsize == 1
    assert bytes(a["tokens"]) == "中文\nshared raw bytes".encode()
    native_a, native_b = left._cache(1, "zh"), right._cache(1, "zh")
    assert native_a["digest"] != native_b["digest"]
    assert native_a["tokens"].dtype.itemsize == 4


def test_initial_state_rewinds_lazy_caches_and_unicode_exposure(tmp_path):
    text = "中文🙂é"
    write_corpus(tmp_path / "data", [text])
    tok = ByteFixtureTokenizer()
    ds = LanguageWindowDataset(tmp_path / "data", tok, 4, split="val", cache_dir=tmp_path / "cache")
    initial = ds.state_dict()
    expected = ds.get_batch("byte", 4)
    ds.get_batch("byte", 4)
    ds.load_state_dict(initial)
    observed = ds.get_batch("byte", 4)
    assert torch.equal(expected["input_ids"], observed["input_ids"])
    ds.load_state_dict(initial)
    pair = ds._cache(1, "zh")
    batches = [ds.get_batch("zh", 1) for _ in range(pair["total_windows"])]
    assert sum(b["batch_stats"]["source_bytes"] for b in batches) == len(text.encode())
    assert sum(b["batch_stats"]["source_characters"] for b in batches) == len(text)


def test_preparation_simplifies_before_split_preserves_whitespace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    files = {}
    for lang in LANGUAGES:
        path = source / f"{lang}.jsonl"
        files[lang] = str(path)
        with path.open("w", encoding="utf-8") as handle:
            for i in range(100):
                text = f"  {lang} 傳統漢字\t{i}\r\n" if lang == "zh" else f"  {lang} original\t{i}\r\n"
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            # Exact duplicates are excluded before split assignment.
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    config = {"source": "local", "local_files": files, "zh_conversion": "t2s",
              "val_fraction": .2, "test_fraction": .2, "max_documents_per_language": 101}
    manifest = prepare(config, tmp_path / "prepared")
    all_hashes = set()
    for lang in LANGUAGES:
        assert manifest["languages"][lang]["duplicate_documents_skipped"] == 1
        for split in ("train", "val", "test"):
            for line in (tmp_path / "prepared" / lang / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                assert row["text"].startswith("  ") and row["text"].endswith("\r\n")
                if lang == "zh":
                    assert "传统汉字" in row["text"] and row["raw_sha256"] != row["sha256"]
                assert row["sha256"] not in all_hashes
                all_hashes.add(row["sha256"])
                assert row["sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()

import json
from pathlib import Path

import pytest

from tokenization_mixture import MixtureTokenizer, ROUTES
from train_tokenizers import allowed_token, prune_candidate, train_tokenizer_bundle


@pytest.fixture(scope="module")
def tokenizer_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("tokenizer")
    alphabets = {
        "zh": "中国人民学习科学数据模型语言世界生活北京上海",
        "ko": "한국사람학교공부과학자료모델언어세계생활서울도시",
        "ja": "日本人学科学習言語世界生活東京大阪あいうえおかきくけこ",
        "en": "abcdefghijklmnopqrstuvwx",
    }
    for route, alphabet in alphabets.items():
        path = root / "data" / route / "train.jsonl"
        path.parent.mkdir(parents=True)
        chars = list(dict.fromkeys(alphabet))
        with path.open("w", encoding="utf-8") as handle:
            for i, a in enumerate(chars):
                for j, b in enumerate(chars):
                    text = (a + b + chars[(i + j) % len(chars)] + " ") * 5 + "0123456789\n"
                    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return train_tokenizer_bundle(
        root / "data", root / "tokenizers", sizes=[64, 128],
        alphabet_limit=48, candidate_size=1024, min_frequency=1,
    )


def test_roundtrip_markers_whitespace_and_unicode(tokenizer_paths):
    tokenizer = MixtureTokenizer.from_pretrained(tokenizer_paths[0])
    texts = [
        "中文 한국어 日本語 English café 2026 😀 𠀀",
        " \t\n\r\r\n\f\v\u00a0\u3000  ",
        "か\u3099 e\u0301 가\u1100\u1161",
        "<0x41> <byte:41> <control:zh:bos> zh::中",
        "\x00\x01\x7f", "",
    ]
    for route in ROUTES:
        for text in texts:
            ids = tokenizer.encode_route(text, route, add_bos=True, add_eos=True)
            assert tokenizer.decode_bytes(ids) == text.encode("utf-8")
            assert tokenizer.decode(ids, skip_special_tokens=True, errors="strict") == text
            assert ids[0] == tokenizer.bos_id(route)
            assert ids[-1] == tokenizer.eos_id(route)


def test_private_digits_and_only_bytes_shared(tokenizer_paths):
    tokenizer = MixtureTokenizer.from_pretrained(tokenizer_paths[0])
    allowed = [set(tokenizer.allowed_ids(route)) for route in ROUTES]
    for r in range(5):
        for s in range(r):
            assert allowed[r] & allowed[s] == set(range(256))
    digit_sets = []
    for route in ROUTES[1:]:
        ids = tokenizer.encode_route("0123456789", route)
        assert len(ids) == len(set(ids)) == 10
        assert all(index >= 256 for index in ids)
        assert tokenizer.encode_route("2026", route) == [ids[2], ids[0], ids[2], ids[6]]
        digit_sets.append(set(ids))
    for r in range(4):
        for s in range(r):
            assert not digit_sets[r] & digit_sets[s]
    assert tokenizer.encode_route("2026", "byte") == list(b"2026")


def test_sizes_nested_policy_fallback_and_hf_roundtrip(tokenizer_paths, tmp_path):
    small, big = [MixtureTokenizer.from_pretrained(path) for path in tokenizer_paths]
    assert small.vocab_size == 256 + 4 * 64 + 15
    assert big.vocab_size == 256 + 4 * 128 + 15
    for route in ROUTES[1:]:
        assert big.lexical_tokens[route][:64] == small.lexical_tokens[route]
        assert all(allowed_token(token, route) for token in big.lexical_tokens[route])
        assert small.encode_route("😀", route) == list("😀".encode("utf-8"))
    target = tmp_path / "saved"
    small.save_pretrained(target)
    restored = MixtureTokenizer.from_pretrained(target, default_route="zh")
    text = "测试2026 <control:zh:bos>\n"
    encoded = restored(text, padding=True, return_tensors="pt")
    assert encoded.input_ids[0, 0].item() == restored.bos_id("zh")
    assert restored.decode(encoded.input_ids[0], skip_special_tokens=True) == text
    assert restored.encode(text, add_special_tokens=False) == restored.encode_route(text, "zh")
    assert restored.fingerprint == small.fingerprint
    assert len(restored) == restored.vocab_size


def test_pruning_removes_dependencies_and_rejects_insufficient_vocabulary():
    vocab = {token: i for i, token in enumerate(list("0123456789中a") + ["中a", "中a中"])}
    candidate = {"model": {"vocab": vocab, "merges": [["中", "a"], ["中a", "中"]]}}
    with pytest.raises(ValueError, match="reachable valid lexical tokens"):
        prune_candidate(candidate, "zh", [12])
    assert allowed_token("かなー゙", "ja")
    assert not allowed_token("abc", "zh")
    assert not allowed_token("한中", "ko")
    assert not allowed_token("123", "en")
    assert not allowed_token("a1", "en")


def test_checksums_detect_incompatible_backend(tokenizer_paths, tmp_path):
    tokenizer = MixtureTokenizer.from_pretrained(tokenizer_paths[0])
    tokenizer.save_pretrained(tmp_path)
    path = tmp_path / "zh.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        MixtureTokenizer.from_pretrained(tmp_path)

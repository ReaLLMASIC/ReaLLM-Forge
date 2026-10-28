"""Train, prune, and export exact-size nested Unicode BPE specialists."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import regex
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

from tokenization_mixture import MixtureTokenizer, ROUTES

DIGITS = "0123456789"
WHITESPACE = set(" \t\r\n\f\v\u00a0\u3000")
COMMON_PUNCT = set(".,;:!?()[]{}'\"-_/\\@#%&")
CJK_PUNCT = set("，。！？；：“”‘’（）【】《》〈〉、·…—～「」『』〔〕［］｛｝﹏")
LATIN_PUNCT = set("‘’“”…–—«»‹›")
SCRIPT = {
    "zh": regex.compile(r"\A\p{Script=Han}\Z"),
    "ko": regex.compile(r"\A\p{Script=Hangul}\Z"),
    "ja": regex.compile(r"\A(?:\p{Script=Han}|\p{Script_Extensions=Hiragana}|\p{Script_Extensions=Katakana})\Z"),
    "en": regex.compile(r"\A(?:\p{Script=Latin}|(?=\p{M})\p{Script=Inherited})\Z"),
}


def allowed_token(token: str, route: str) -> bool:
    """Private token policy; bytes and route controls are handled separately."""
    if not token:
        return False
    if any(ch in DIGITS for ch in token):
        return len(token) == 1 and token in DIGITS
    punctuation = COMMON_PUNCT | (LATIN_PUNCT if route == "en" else CJK_PUNCT)
    return all(ch in WHITESPACE or ch in punctuation or SCRIPT[route].fullmatch(ch) for ch in token)


def make_pre_tokenizer():
    # Isolated matches preserve every delimiter; whitespace is not discarded.
    return pre_tokenizers.Split(Regex(r"[0-9]|\s+"), behavior="isolated")


def iter_documents(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_no}: expected a string text field")
            yield record["text"]


def train_candidate(path: Path, alphabet_limit: int, candidate_size: int, min_frequency: int):
    counts = Counter()
    document_count = 0
    digest = hashlib.sha256()
    for text in iter_documents(path):
        counts.update(text)
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        document_count += 1
    if not counts:
        raise ValueError(f"Empty training corpus: {path}")
    alphabet = list(DIGITS)
    alphabet.extend(
        ch for ch, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if ch not in DIGITS
    )
    alphabet = alphabet[:alphabet_limit]
    alphabet_set = set(alphabet)

    def chunks():
        # Characters outside the capped alphabet are boundaries during training,
        # preventing artificial merges across a deleted rare character.
        for text in iter_documents(path):
            current = []
            for ch in text:
                if ch in alphabet_set:
                    current.append(ch)
                elif current:
                    yield "".join(current)
                    current = []
            if current:
                yield "".join(current)

    candidate = Tokenizer(models.BPE())
    candidate.pre_tokenizer = make_pre_tokenizer()
    candidate.train_from_iterator(
        chunks(),
        trainer=trainers.BpeTrainer(
            vocab_size=candidate_size, min_frequency=min_frequency,
            initial_alphabet=alphabet, limit_alphabet=len(alphabet), show_progress=False,
        ),
    )
    return json.loads(candidate.to_str()), {
        "train_document_count": document_count,
        "train_text_sha256": digest.hexdigest(),
        "candidate_vocab_size": candidate.get_vocab_size(),
        "candidate_alphabet_size": len(alphabet),
    }


def prune_candidate(candidate: dict, route: str, sizes: list[int]):
    """Replay legal merges in rank order, retaining dependency-closed prefixes."""
    raw_model = candidate["model"]
    vocab = raw_model["vocab"]
    # Sort base symbols explicitly so their IDs do not depend on hash order.
    base = sorted(token for token in vocab if len(token) == 1 and allowed_token(token, route))
    for digit in DIGITS:
        if digit not in base:
            raise ValueError("Candidate trainer failed to retain mandatory digits")
    if len(base) > min(sizes):
        raise ValueError(f"{route}: {len(base)} base characters exceed smallest K={min(sizes)}; lower --alphabet-limit")
    lexical = list(base)
    retained = set(lexical)
    valid_merges = []
    snapshots = {}
    if len(lexical) in sizes:
        snapshots[len(lexical)] = (list(lexical), list(valid_merges))
    for merge in raw_model["merges"]:
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        combined = left + right
        if left not in retained or right not in retained or not allowed_token(combined, route):
            continue
        valid_merges.append((left, right))
        if combined not in retained:
            retained.add(combined)
            lexical.append(combined)
            if len(lexical) in sizes:
                snapshots[len(lexical)] = (list(lexical), list(valid_merges))
            if len(lexical) == max(sizes):
                break
    missing = sorted(set(sizes) - set(snapshots))
    if missing:
        raise ValueError(
            f"{route}: only {len(lexical)} reachable valid lexical tokens; cannot export {missing}. "
            "Use more training documents or increase --candidate-size; no dummy tokens were inserted."
        )
    return snapshots


def build_backend(tokens, merges):
    vocab = {token: i for i, token in enumerate(tokens)}
    vocab.update({f"<0x{byte:02X}>": len(tokens) + byte for byte in range(256)})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True))
    backend.pre_tokenizer = make_pre_tokenizer()
    return backend


def train_tokenizer_bundle(
    data_dir, output_dir, sizes=(10000, 20000), alphabet_limit=4000,
    candidate_size=40000, min_frequency=2,
):
    sizes = sorted(set(int(k) for k in sizes))
    if not sizes or min(sizes) < 10:
        raise ValueError("Every lexical size must be at least ten")
    if not 10 <= alphabet_limit <= min(sizes):
        raise ValueError("alphabet_limit must be between 10 and the smallest lexical size")
    if candidate_size < max(sizes):
        raise ValueError("candidate_size must be at least the largest requested lexical size")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    snapshots, audit = {}, {}
    for route in ROUTES[1:]:
        candidate, info = train_candidate(
            data_dir / route / "train.jsonl", alphabet_limit, candidate_size, min_frequency
        )
        snapshots[route] = prune_candidate(candidate, route, sizes)
        audit[route] = info
        print(f"{route}: candidate={info['candidate_vocab_size']:,}; valid exact sizes={sizes}", flush=True)
    destinations = []
    # Build every candidate before writing any bundle, so insufficient data fails
    # without producing an apparently complete half-exported experiment.
    for size in sizes:
        directory = output_dir / f"k{size}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": 1, "routes": list(ROUTES), "lexical_size": size,
            "vocab_size": 256 + 4 * size + 15,
            "normalizer": None, "digit_policy": "ASCII singleton private IDs",
            "policy_version": "unicode-script-whitelist-v1",
            "backends": {}, "training": audit,
        }
        for route in ROUTES[1:]:
            tokens, merges = snapshots[route][size]
            backend = build_backend(tokens, merges)
            path = directory / f"{route}.json"
            raw = backend.to_str(pretty=False).encode("utf-8")
            path.write_bytes(raw)
            manifest["backends"][route] = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
        path = directory / "mixture_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tokenizer = MixtureTokenizer(directory=directory)
        tokenizer.save_pretrained(directory)
        probes = ["0123456789", "中文 한국어 日本語 café 2026! 😀", " \t\n\r\r\n  ", "<0x41> <control:zh:bos>", "か\u3099 e\u0301 𠀀"]
        for route in ROUTES:
            for text in probes:
                ids = tokenizer.encode_route(text, route)
                if tokenizer.decode_bytes(ids) != text.encode("utf-8"):
                    raise AssertionError(f"Round-trip audit failed for {route}: {text!r}")
            if route != "byte":
                digit_ids = tokenizer.encode_route(DIGITS, route)
                if len(digit_ids) != 10 or any(i < 256 for i in digit_ids):
                    raise AssertionError(f"Digit audit failed for {route}")
        destinations.append(directory)
    return destinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing zh/ko/ja/en train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 20000])
    parser.add_argument("--alphabet-limit", type=int, default=4000)
    parser.add_argument("--candidate-size", type=int, default=40000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    for path in train_tokenizer_bundle(
        args.data, args.output, args.sizes, args.alphabet_limit,
        args.candidate_size, args.min_frequency,
    ):
        print(path)


if __name__ == "__main__":
    main()

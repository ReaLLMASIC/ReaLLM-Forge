#!/usr/bin/env python3
"""Compare orthographic vocabulary factorization across tokenizer families."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_HF_MODELS = ("google/gemma-3-270m", "Qwen/Qwen3-0.6B")
DEFAULT_TIKTOKEN_ENCODINGS = ("o200k_base",)
SPACE_MARKERS = ("▁", "Ġ", " ")


@dataclass(frozen=True)
class FactorizationResult:
    vocabulary_size: int
    leading_space: frozenset[str]
    initial_capital: frozenset[str]
    all_caps: frozenset[str]

    @property
    def removed(self) -> frozenset[str]:
        return self.leading_space | self.initial_capital | self.all_caps

    @property
    def factorized_vocabulary_size(self) -> int:
        return self.vocabulary_size - len(self.removed)

    @property
    def reduction_percent(self) -> float:
        return 100 * len(self.removed) / self.vocabulary_size if self.vocabulary_size else 0.0

    def exclusive_counts(self) -> tuple[int, int, int]:
        capital = self.initial_capital - self.leading_space
        all_caps = self.all_caps - self.leading_space - self.initial_capital
        return len(self.leading_space), len(capital), len(all_caps)


def _split_space_marker(token: str, marker: str) -> tuple[str, str]:
    if token.startswith(marker):
        return marker, token[len(marker) :]
    return "", token


def _has_case(character: str) -> bool:
    return character.lower() != character.upper()


def analyze_tokens(tokens: Iterable[str]) -> FactorizationResult:
    """Find rows with exact counterparts under SPACE, CAPITALIZE, or ALL_CAPS."""
    vocabulary = frozenset(tokens)
    # Tokenizer families serialize space differently. Select one convention for
    # the whole vocabulary so a literal Ġ in SentencePiece is not misread as a
    # byte-level-BPE marker (and vice versa).
    space_marker = max(SPACE_MARKERS, key=lambda marker: sum(token.startswith(marker) for token in vocabulary))
    leading_space: set[str] = set()
    initial_capital: set[str] = set()
    all_caps: set[str] = set()

    for token in vocabulary:
        prefix, text = _split_space_marker(token, space_marker)
        if prefix and text and text in vocabulary:
            leading_space.add(token)
        if text and text[0].isupper():
            counterpart = prefix + text[0].lower() + text[1:]
            if counterpart != token and counterpart in vocabulary:
                initial_capital.add(token)
        cased_characters = [character for character in text if _has_case(character)]
        if cased_characters and all(character.isupper() for character in cased_characters):
            counterpart = prefix + text.lower()
            if counterpart != token and counterpart in vocabulary:
                all_caps.add(token)

    return FactorizationResult(
        len(vocabulary), frozenset(leading_space), frozenset(initial_capital), frozenset(all_caps)
    )


def load_tokenizer_vocabulary(path: Path) -> dict[str, int]:
    """Load the model vocabulary from a Hugging Face ``tokenizer.json``."""
    with path.open(encoding="utf-8") as tokenizer_file:
        document = json.load(tokenizer_file)
    vocabulary = document.get("model", {}).get("vocab")
    if isinstance(vocabulary, list):  # Unigram tokenizer JSON uses [token, score] rows.
        if not all(isinstance(row, list) and row and isinstance(row[0], str) for row in vocabulary):
            raise ValueError(f"{path} has an invalid model.vocab")
        vocabulary = {row[0]: index for index, row in enumerate(vocabulary)}
    if not isinstance(vocabulary, dict):
        raise ValueError(f"{path} does not contain a supported model.vocab")
    if not all(isinstance(token, str) and isinstance(token_id, int) for token, token_id in vocabulary.items()):
        raise ValueError(f"{path} has an invalid model.vocab")
    return vocabulary


def load_huggingface_vocabulary(model_name: str) -> dict[str, int]:
    """Load any Hugging Face tokenizer through Transformers."""
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Hugging Face models require `pip install transformers`") from error
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    config = AutoConfig.from_pretrained(model_name, token=token)
    head_size = int(getattr(config, "vocab_size", 0) or tokenizer.vocab_size)
    vocabulary = {text: token_id for text, token_id in tokenizer.get_vocab().items() if token_id < head_size}
    used_ids = set(vocabulary.values())
    for token_id in range(head_size):
        if token_id not in used_ids:
            vocabulary[f"<reserved:{token_id}>"] = token_id
    return vocabulary


def load_tiktoken_vocabulary(encoding_name: str) -> dict[str, int]:
    """Expose a tiktoken encoding as comparable token strings.

    Valid UTF-8 byte tokens use their decoded text. Invalid byte fragments get
    a private diagnostic spelling and therefore cannot produce false matches.
    """
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("tiktoken encodings require `pip install tiktoken`") from error
    encoding = tiktoken.get_encoding(encoding_name)
    vocabulary: dict[str, int] = dict(encoding._special_tokens)
    used_ids = set(vocabulary.values())
    for raw, token_id in encoding._mergeable_ranks.items():
        used_ids.add(token_id)
        try:
            token = raw.decode("utf-8")
        except UnicodeDecodeError:
            token = f"<invalid-utf8:{raw.hex()}>"
        vocabulary[token] = token_id
    # n_vocab is the embedding/head extent. Preserve reserved holes so the
    # before/after comparison does not pretend those rows have disappeared.
    for token_id in range(encoding.n_vocab):
        if token_id not in used_ids:
            vocabulary[f"<reserved:{token_id}>"] = token_id
    return vocabulary


def _print_detail(label: str, vocabulary: dict[str, int], examples: int) -> FactorizationResult:
    result = analyze_tokens(vocabulary)
    exclusive = result.exclusive_counts()
    print(f"\n{label}")
    for name, matches, newly_removed in (
        ("leading space", result.leading_space, exclusive[0]),
        ("initial capital", result.initial_capital, exclusive[1]),
        ("all caps", result.all_caps, exclusive[2]),
    ):
        sample = sorted(matches, key=vocabulary.get)[:examples]
        print(f"  {name}: {len(matches):,} matches; {newly_removed:,} newly removed; examples={sample!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", nargs="*", type=Path, help="Local Hugging Face tokenizer.json files")
    parser.add_argument("--hf-model", action="append", default=[], help="Hugging Face model ID (repeatable)")
    parser.add_argument("--tiktoken-encoding", action="append", default=[], help="tiktoken encoding (repeatable)")
    parser.add_argument("--no-defaults", action="store_true", help="Do not include Gemma, Qwen, and o200k_base defaults")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    hf_models = list(args.hf_model)
    tiktoken_encodings = list(args.tiktoken_encoding)
    if not args.no_defaults:
        hf_models = list(DEFAULT_HF_MODELS) + hf_models
        tiktoken_encodings = list(DEFAULT_TIKTOKEN_ENCODINGS) + tiktoken_encodings

    sources = [(f"file:{path}", lambda path=path: load_tokenizer_vocabulary(path)) for path in args.tokenizer_json]
    sources += [(f"hf:{name}", lambda name=name: load_huggingface_vocabulary(name)) for name in hf_models]
    sources += [(f"tiktoken:{name}", lambda name=name: load_tiktoken_vocabulary(name)) for name in tiktoken_encodings]
    if not sources:
        parser.error("no tokenizer sources selected")

    results: list[tuple[str, FactorizationResult]] = []
    failures = 0
    for label, loader in sources:
        try:
            vocabulary = loader()
            results.append((label, _print_detail(label, vocabulary, args.examples)))
        except Exception as error:  # Continue comparing independently loadable sources.
            failures += 1
            print(f"warning: could not analyze {label}: {error}", file=sys.stderr)

    if not results:
        raise SystemExit("no tokenizer could be analyzed")
    print("\nComparison")
    print(f"{'source':<38} {'before':>12} {'removed':>12} {'after':>12} {'reduction':>10}")
    for label, result in results:
        print(
            f"{label:<38} {result.vocabulary_size:>12,} {len(result.removed):>12,} "
            f"{result.factorized_vocabulary_size:>12,} {result.reduction_percent:>9.2f}%"
        )
    if failures:
        print(f"\n{failures} source(s) failed; see warnings above.", file=sys.stderr)


if __name__ == "__main__":
    main()

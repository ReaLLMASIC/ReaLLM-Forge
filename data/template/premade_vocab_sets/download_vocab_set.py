"""Download and normalize Hugging Face tokenizer vocabularies.

This script downloads a tokenizer file from Hugging Face (``tokenizer.json`` or
``tokenizer.model``), extracts the token list, removes byte fallback tokens by
default, optionally maps common tokenizer whitespace marker characters to their
literal equivalents, and writes the normalized list to JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.error import HTTPError
from urllib.request import urlretrieve
import importlib
import importlib.util


BYTE_TOKEN_REGEX = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
BYTE_TOKENS = {chr(i) for i in range(256)}

# Common "whitespace marker" glyphs found in tokenizer vocabs.
# - GPT-2/RoBERTa byte-level BPE uses bytes->unicode where:
#   space(0x20)->'Ġ', \n(0x0A)->'Ċ', \t(0x09)->'ĉ', \r(0x0D)->'č'
# - SentencePiece uses '▁' as a visible whitespace marker.
_SPECIAL_WHITESPACE_MAP = {
    "Ġ": " ",   # U+0120
    "Ċ": "\n",  # U+010A
    "ĉ": "\t",  # U+0109
    "č": "\r",  # U+010D
    "▁": " ",   # U+2581
}


class TokenizerDownloadError(RuntimeError):
    """Raised when a tokenizer file cannot be downloaded."""


def _download_tokenizer_file(repo_id: str, filename: str, revision: str, dest_dir: Path) -> Path:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    dest_path = dest_dir / filename
    urlretrieve(url, dest_path)
    return dest_path


def _load_sentencepiece_tokens(model_path: Path) -> List[str]:
    spec = importlib.util.find_spec("sentencepiece")
    if spec is None:
        raise RuntimeError("sentencepiece is required to read tokenizer.model files.")
    sentencepiece = importlib.import_module("sentencepiece")
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    return [processor.id_to_piece(i) for i in range(processor.get_piece_size())]


def _extract_tokens_from_tokenizer_json(data: dict) -> List[str]:
    model = data.get("model", {})
    model_type = str(model.get("type", "")).lower()

    if model_type in {"bpe", "wordlevel"}:
        vocab = model.get("vocab", {})
        if not isinstance(vocab, dict):
            raise ValueError("Expected vocab dictionary in tokenizer.json model.")
        tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    elif model_type == "unigram":
        vocab = model.get("vocab", [])
        if not isinstance(vocab, list):
            raise ValueError("Expected vocab list in tokenizer.json model.")
        tokens = [token for token, _score in vocab]
    else:
        raise ValueError(f"Unsupported tokenizer model type: {model_type!r}")

    added_tokens = data.get("added_tokens", [])
    if isinstance(added_tokens, list) and added_tokens:
        added_tokens_sorted = sorted(added_tokens, key=lambda item: item.get("id", 0))
        for entry in added_tokens_sorted:
            token = entry.get("content")
            if isinstance(token, str) and token not in tokens:
                tokens.append(token)

    return tokens


def _remove_byte_tokens(tokens: Sequence[str]) -> List[str]:
    filtered: List[str] = []
    for token in tokens:
        if token in BYTE_TOKENS or BYTE_TOKEN_REGEX.match(token):
            continue
        filtered.append(token)
    return filtered


def _map_special_whitespace(tokens: Sequence[str]) -> List[str]:
    """Map common tokenizer whitespace markers to literal whitespace characters."""
    mapped: List[str] = []
    for token in tokens:
        out = token
        for src, dst in _SPECIAL_WHITESPACE_MAP.items():
            if src in out:
                out = out.replace(src, dst)
        mapped.append(out)
    return mapped


def _write_tokens(tokens: Iterable[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(list(tokens), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def _load_tokens_from_repo(repo_id: str, revision: str) -> List[str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for filename in ("tokenizer.json", "tokenizer.model"):
            try:
                tokenizer_path = _download_tokenizer_file(repo_id, filename, revision, tmp_path)
            except HTTPError:
                continue
            if tokenizer_path.suffix == ".json":
                data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
                return _extract_tokens_from_tokenizer_json(data)
            return _load_sentencepiece_tokens(tokenizer_path)

    raise TokenizerDownloadError(
        f"Unable to download tokenizer.json or tokenizer.model for {repo_id}@{revision}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo id (e.g. google/gemma-2b).")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for the normalized token list.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision (branch, tag, or commit). Defaults to 'main'.",
    )
    parser.add_argument(
        "--keep-byte-tokens",
        action="store_true",
        help="Keep byte fallback tokens instead of removing them.",
    )

    # Default ON mapping; allow disabling with a single flag.
    parser.add_argument(
        "--no-map-special-whitespace",
        dest="map_special_whitespace",
        action="store_false",
        help="Do not map common tokenizer whitespace markers (e.g. Ġ/Ċ/▁) to literal whitespace.",
    )
    parser.set_defaults(map_special_whitespace=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = _load_tokens_from_repo(args.model, args.revision)

    if not args.keep_byte_tokens:
        tokens = _remove_byte_tokens(tokens)

    if args.map_special_whitespace:
        tokens = _map_special_whitespace(tokens)

    _write_tokens(tokens, args.output)
    print(f"Wrote {len(tokens):,} tokens to {args.output}")


if __name__ == "__main__":
    main()


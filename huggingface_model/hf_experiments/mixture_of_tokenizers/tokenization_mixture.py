"""Lossless manual routing over four HF BPE backends and shared UTF-8 bytes.

Global token labels are namespaced for HF vocabulary inspection. They are never
inserted into user text, and literal control/byte-marker strings remain text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer

ROUTES = ("byte", "zh", "ko", "ja", "en")
CONTROL_NAMES = ("bos", "eos", "pad")


def route_index(route: str | int) -> int:
    if isinstance(route, str):
        if route not in ROUTES:
            raise ValueError(f"Unknown route {route!r}; choose one of {ROUTES}")
        return ROUTES.index(route)
    index = int(route)
    if not 0 <= index < len(ROUTES):
        raise ValueError(f"Route index must be in [0, 4], got {route}")
    return index


class MixtureTokenizer(PreTrainedTokenizer):
    """HF tokenizer facade with an explicit, non-learned route selection.

    ``encode_route`` is stateless and suitable for mixed-route training. Normal
    HF ``encode``/``__call__`` methods use ``default_route``. Select another
    default by loading with ``default_route='zh'``; no mutable global router is
    required. A call with ``add_special_tokens=True`` prepends route-local BOS.
    """

    vocab_files_names = {"manifest_file": "mixture_manifest.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        manifest_file: str | None = None,
        directory: str | Path | None = None,
        default_route: str = "byte",
        **kwargs,
    ):
        if manifest_file is None:
            if directory is None:
                raise ValueError("Provide directory or manifest_file for a trained bundle")
            manifest_file = str(Path(directory) / "mixture_manifest.json")
        self.manifest_file = str(manifest_file)
        self.manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported mixture tokenizer format_version")
        if tuple(self.manifest.get("routes", ())) != ROUTES:
            raise ValueError("Tokenizer route ordering does not match the model contract")
        self.lexical_size = int(self.manifest["lexical_size"])
        self.default_route = ROUTES[route_index(default_route)]
        self.backends = {}
        self.lexical_tokens = {}
        self._labels = [f"<byte:{i:02X}>" for i in range(256)]
        self._payloads = [bytes([i]) for i in range(256)]
        root = Path(manifest_file).parent
        for name in ROUTES[1:]:
            info = self.manifest["backends"][name]
            backend_path = root / info["file"]
            raw = backend_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError(f"Tokenizer backend checksum mismatch: {backend_path}")
            backend = Tokenizer.from_str(raw.decode("utf-8"))
            vocab = backend.get_vocab()
            if len(vocab) != self.lexical_size + 256:
                raise ValueError(f"Wrong vocabulary size for backend {name}")
            text_tokens = [backend.id_to_token(i) for i in range(self.lexical_size)]
            if any(token is None for token in text_tokens):
                raise ValueError(f"Noncontiguous lexical IDs in backend {name}")
            for byte in range(256):
                if vocab.get(f"<0x{byte:02X}>") != self.lexical_size + byte:
                    raise ValueError(f"Invalid byte fallback mapping in backend {name}")
            self.backends[name] = backend
            self.lexical_tokens[name] = text_tokens
            self._labels.extend(f"{name}::{token}" for token in text_tokens)
            self._payloads.extend(token.encode("utf-8") for token in text_tokens)
        self._labels.extend(
            f"<control:{name}:{kind}>" for name in ROUTES for kind in CONTROL_NAMES
        )
        self._payloads.extend([b""] * 15)
        self._vocab = {token: index for index, token in enumerate(self._labels)}
        if len(self._vocab) != self.vocab_size:
            raise ValueError("Nonunique global token labels")
        self.fingerprint = hashlib.sha256(
            json.dumps(self.manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        # Replace route-dependent HF defaults when loading under a new route.
        for key in ("bos_token", "eos_token", "pad_token", "additional_special_tokens"):
            kwargs.pop(key, None)
        kwargs.pop("clean_up_tokenization_spaces", None)
        super().__init__(
            bos_token=self._labels[self.bos_id(self.default_route)],
            eos_token=self._labels[self.eos_id(self.default_route)],
            pad_token=self._labels[self.pad_id(self.default_route)],
            additional_special_tokens=self._labels[-15:],
            clean_up_tokenization_spaces=False,
            default_route=self.default_route,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return 256 + 4 * self.lexical_size + 15

    @property
    def route_names(self):
        return ROUTES

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def bos_id(self, route: str | int) -> int:
        return 256 + 4 * self.lexical_size + 3 * route_index(route)

    def eos_id(self, route: str | int) -> int:
        return self.bos_id(route) + 1

    def pad_id(self, route: str | int) -> int:
        return self.bos_id(route) + 2

    def allowed_ids(self, route: str | int) -> list[int]:
        index = route_index(route)
        private = [] if index == 0 else list(
            range(256 + (index - 1) * self.lexical_size, 256 + index * self.lexical_size)
        )
        return list(range(256)) + private + [self.eos_id(index)]

    def encode_route(
        self, text: str, route: str | int, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        index = route_index(route)
        if not isinstance(text, str):
            raise TypeError("encode_route expects a Unicode string")
        if index == 0:
            ids = list(text.encode("utf-8"))
        else:
            local = self.backends[ROUTES[index]].encode(text, add_special_tokens=False).ids
            offset = 256 + (index - 1) * self.lexical_size
            ids = [i - self.lexical_size if i >= self.lexical_size else offset + i for i in local]
        return ([self.bos_id(index)] if add_bos else []) + ids + ([self.eos_id(index)] if add_eos else [])

    def token_byte_payload(self, token_id: int) -> bytes:
        index = int(token_id)
        if not 0 <= index < self.vocab_size:
            raise ValueError(f"Token ID out of vocabulary: {index}")
        return self._payloads[index]

    def decode_bytes(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> bytes:
        pieces = []
        control_start = 256 + 4 * self.lexical_size
        for token_id in token_ids:
            index = int(token_id)
            payload = self.token_byte_payload(index)
            if index >= control_start and not skip_special_tokens:
                payload = self._labels[index].encode("utf-8")
            pieces.append(payload)
        return b"".join(pieces)

    def _decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
        # UTF-8 must be decoded after assembling bytes, never token by token.
        return self.decode_bytes(token_ids, skip_special_tokens).decode(
            "utf-8", errors=kwargs.pop("errors", "replace")
        )

    def tokenize(self, text, **kwargs):
        # Bypass HF AddedToken matching so literal '<control:...>' remains text.
        if kwargs:
            unsupported = set(kwargs) - {"split_special_tokens"}
            if unsupported:
                raise ValueError(f"Unsupported tokenize arguments: {sorted(unsupported)}")
        return self._tokenize(text)

    def _tokenize(self, text):
        return [self._labels[i] for i in self.encode_route(text, self.default_route)]

    def _convert_token_to_id(self, token):
        if token not in self._vocab:
            raise ValueError(f"Unknown namespaced token: {token!r}")
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        if not 0 <= int(index) < self.vocab_size:
            raise ValueError(f"Token ID out of vocabulary: {index}")
        return self._labels[int(index)]

    def convert_tokens_to_string(self, tokens):
        return self.decode_bytes(self._convert_token_to_id(t) for t in tokens).decode("utf-8", errors="replace")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise ValueError("Sequence pairs are unsupported; route each causal sequence explicitly")
        return [self.bos_id(self.default_route)] + token_ids_0

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            start = 256 + 4 * self.lexical_size
            return [int(i >= start) for i in token_ids_0]
        if token_ids_1 is not None:
            raise ValueError("Sequence pairs are unsupported")
        return [1] + [0] * len(token_ids_0)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(self, save_directory, filename_prefix=None):
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        manifest = dict(self.manifest)
        manifest["backends"] = {}
        files = []
        for route, backend in self.backends.items():
            path = directory / f"{prefix}{route}.json"
            raw = backend.to_str(pretty=False).encode("utf-8")
            path.write_bytes(raw)
            manifest["backends"][route] = {
                "file": path.name, "sha256": hashlib.sha256(raw).hexdigest()
            }
            files.append(str(path))
        path = directory / f"{prefix}mixture_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return (str(path), *files)

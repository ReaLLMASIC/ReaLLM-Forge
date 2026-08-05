from __future__ import annotations

import gc
import json
import math
import os
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-3-270m")
DEFAULT_COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "auto")
DEFAULT_TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").casefold() in {"1", "true", "yes", "on"}
DEFAULT_LOAD_STRATEGY = os.getenv("MODEL_LOAD_STRATEGY", "auto").strip().casefold()
MAX_SEARCH_PATTERN_LENGTH = 256
MAX_SEARCH_RESULTS = 5000
MAX_SELECTED_TOKENS = 5000
MAX_TEXT_LENGTH = 100_000
MAX_TOKENIZE_RESULTS = 10_000

_OUTPUT_CANDIDATES: tuple[str, ...] = (
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
    "embed_out.weight",
    "gpt_neox.embed_out.weight",
    "output.weight",
    "decoder.output_projection.weight",
    "transformer.lm_head.weight",
    "cls.predictions.decoder.weight",
    "predictions.decoder.weight",
    "lm_head.decoder.weight",
    "vocab_projector.weight",
    "generator_lm_head.weight",
)
_INPUT_CANDIDATES: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "transformer.wte.weight",
    "model.transformer.wte.weight",
    "gpt_neox.embed_in.weight",
    "decoder.embed_tokens.weight",
    "model.decoder.embed_tokens.weight",
    "embed_tokens.weight",
    "shared.weight",
    "model.shared.weight",
    "embeddings.word_embeddings.weight",
    "model.embeddings.word_embeddings.weight",
)


@dataclass(frozen=True, slots=True)
class TokenInfo:
    token_id: int
    raw: str
    display: str
    search_blob: str
    special: bool
    present_in_tokenizer: bool = True


@dataclass(frozen=True, slots=True)
class LocalModelInfo:
    model_name: str
    cache_path: str | None = None
    size_bytes: int | None = None
    last_modified: float | None = None


@dataclass(slots=True)
class ModelAssets:
    model_name: str
    revision: str
    matrix_source: str
    tensor_name: str
    tokenizer: Any
    token_infos: list[TokenInfo]
    weight: torch.Tensor
    magnitudes: torch.Tensor
    compute_device: str
    load_strategy: str

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.weight.shape[1])

    @property
    def dtype(self) -> str:
        return str(self.weight.dtype).replace("torch.", "")

    @property
    def memory_bytes(self) -> int:
        return int(
            self.weight.nelement() * self.weight.element_size()
            + self.magnitudes.nelement() * self.magnitudes.element_size()
        )

    def token(self, token_id: int) -> TokenInfo:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"token_id must be in [0, {self.vocab_size - 1}], got {token_id}")
        return self.token_infos[token_id]


_ASSETS_LOCK = threading.RLock()
_ASSETS: ModelAssets | None = None


def _import_transformers():
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "The 'transformers' package is required. Install the project requirements before loading a model."
        ) from exc
    return AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


def resolve_compute_device(requested: str | None = None) -> str:
    value = (requested or DEFAULT_COMPUTE_DEVICE or "auto").strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    try:
        torch.device(value)
    except Exception as exc:
        raise ValueError(f"Invalid torch device {value!r}.") from exc
    return value


def _local_path(model_name: str) -> Path | None:
    try:
        path = Path(model_name).expanduser()
    except Exception:
        return None
    return path if path.exists() else None


def _repo_file(model_name: str, filename: str, *, revision: str, allow_download: bool) -> Path:
    local = _local_path(model_name)
    if local is not None:
        if local.is_file():
            if local.name == filename:
                return local
            raise FileNotFoundError(f"{filename!r} is not available in local file {str(local)!r}.")
        candidate = local / filename
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{filename!r} is not available in local model directory {str(local)!r}.")
    return Path(
        hf_hub_download(
            repo_id=model_name,
            filename=filename,
            revision=revision,
            local_files_only=not allow_download,
        )
    )


def _try_repo_file(model_name: str, filename: str, *, revision: str, allow_download: bool) -> Path | None:
    try:
        return _repo_file(model_name, filename, revision=revision, allow_download=allow_download)
    except Exception:
        return None


def _candidate_rank(name: str, source: str) -> tuple[int, int, str]:
    candidates = _OUTPUT_CANDIDATES if source == "output" else _INPUT_CANDIDATES
    for index, candidate in enumerate(candidates):
        if name == candidate:
            return (0, index, name)
    for index, candidate in enumerate(candidates):
        if name.endswith("." + candidate):
            return (1, index, name)
    for index, candidate in enumerate(candidates):
        suffix = "." + candidate.split(".")[-2] + ".weight" if "." in candidate else candidate
        if name.endswith(suffix):
            return (2, index, name)
    return (99, 99, name)


def _choose_tensor_name(names: Iterable[str], requested_source: str) -> tuple[str, str] | None:
    source_order = [requested_source] if requested_source in {"input", "output"} else ["output", "input"]
    name_list = [str(name) for name in names]
    for source in source_order:
        eligible = [name for name in name_list if _candidate_rank(name, source)[0] < 99]
        if eligible:
            return sorted(eligible, key=lambda item: _candidate_rank(item, source))[0], source
    return None


def _read_tensor(path: Path, tensor_name: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if tensor_name not in set(handle.keys()):
            raise KeyError(f"Tensor {tensor_name!r} was not found in {str(path)!r}.")
        tensor = handle.get_tensor(tensor_name)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2-D vocabulary matrix, but {tensor_name!r} has shape {tuple(tensor.shape)}.")
    return tensor.detach().cpu().contiguous()


def _load_from_index(
    model_name: str,
    *,
    revision: str,
    requested_source: str,
    allow_download: bool,
) -> tuple[torch.Tensor, str, str] | None:
    index_path = _try_repo_file(
        model_name,
        "model.safetensors.index.json",
        revision=revision,
        allow_download=allow_download,
    )
    if index_path is None:
        return None
    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a weight_map object.")
    choice = _choose_tensor_name(weight_map.keys(), requested_source)
    if choice is None:
        return None
    tensor_name, actual_source = choice
    shard_name = weight_map.get(tensor_name)
    if not isinstance(shard_name, str):
        raise ValueError(f"Invalid shard reference for tensor {tensor_name!r}.")
    shard_path = _repo_file(model_name, shard_name, revision=revision, allow_download=allow_download)
    return _read_tensor(shard_path, tensor_name), tensor_name, actual_source


def _list_safetensors_files(model_name: str, *, revision: str, allow_download: bool) -> list[str]:
    local = _local_path(model_name)
    if local is not None:
        if local.is_file() and local.suffix == ".safetensors":
            return [local.name]
        if local.is_dir():
            return sorted(str(path.relative_to(local)) for path in local.rglob("*.safetensors"))
        return []
    if not allow_download:
        return ["model.safetensors", "pytorch_model.safetensors"]
    try:
        return sorted(
            name
            for name in list_repo_files(repo_id=model_name, repo_type="model", revision=revision)
            if name.endswith(".safetensors")
        )
    except Exception:
        return ["model.safetensors", "pytorch_model.safetensors"]


def _load_from_single_safetensors(
    model_name: str,
    *,
    revision: str,
    requested_source: str,
    allow_download: bool,
) -> tuple[torch.Tensor, str, str] | None:
    files = _list_safetensors_files(model_name, revision=revision, allow_download=allow_download)
    preferred = ["model.safetensors", "pytorch_model.safetensors"]
    ordered = [name for name in preferred if name in files] + [name for name in files if name not in preferred]
    local = _local_path(model_name)
    if local is None and len(ordered) > 1:
        ordered = [name for name in ordered if name in preferred]
    for filename in ordered:
        path = _try_repo_file(model_name, filename, revision=revision, allow_download=allow_download)
        if path is None:
            continue
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                choice = _choose_tensor_name(handle.keys(), requested_source)
            if choice is None:
                continue
            tensor_name, actual_source = choice
            return _read_tensor(path, tensor_name), tensor_name, actual_source
        except Exception:
            if len(ordered) == 1:
                raise
    return None


def _load_weight_only(
    model_name: str,
    *,
    revision: str,
    requested_source: str,
    allow_download: bool,
) -> tuple[torch.Tensor, str, str]:
    indexed = _load_from_index(
        model_name,
        revision=revision,
        requested_source=requested_source,
        allow_download=allow_download,
    )
    if indexed is not None:
        return indexed
    single = _load_from_single_safetensors(
        model_name,
        revision=revision,
        requested_source=requested_source,
        allow_download=allow_download,
    )
    if single is not None:
        return single
    raise ValueError(
        "Could not locate a compatible input-embedding or LM-head tensor in safetensors files. "
        "The full Transformers model loader can be used as a fallback."
    )


def _load_full_model(
    model_name: str,
    *,
    revision: str,
    requested_source: str,
    allow_download: bool,
) -> tuple[torch.Tensor, str, str]:
    AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, _ = _import_transformers()
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": not allow_download,
        "trust_remote_code": DEFAULT_TRUST_REMOTE_CODE,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
    }
    if requested_source == "input":
        candidates = [AutoModel]
    elif requested_source == "output":
        candidates = [AutoModelForCausalLM, AutoModelForMaskedLM]
    else:
        candidates = [AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel]

    errors: list[str] = []
    for model_class in candidates:
        model = None
        try:
            model = model_class.from_pretrained(model_name, **kwargs)
            output = model.get_output_embeddings() if callable(getattr(model, "get_output_embeddings", None)) else None
            input_embedding = (
                model.get_input_embeddings() if callable(getattr(model, "get_input_embeddings", None)) else None
            )
            if requested_source == "output":
                chosen, source = output, "output"
            elif requested_source == "input":
                chosen, source = input_embedding, "input"
            else:
                chosen, source = (output, "output") if output is not None else (input_embedding, "input")
            if chosen is None or getattr(chosen, "weight", None) is None:
                raise ValueError(
                    f"{model_class.__name__} does not expose a usable {requested_source} vocabulary matrix."
                )
            tensor = chosen.weight.detach().cpu().contiguous()
            name = "get_output_embeddings().weight" if source == "output" else "get_input_embeddings().weight"
            return tensor, name, source
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("; ".join(errors))


def _orient_weight(weight: torch.Tensor, expected_vocab: int) -> torch.Tensor:
    if weight.shape[0] == expected_vocab:
        return weight
    if weight.shape[1] == expected_vocab:
        return weight.T.contiguous()
    # Some tokenizers expose added tokens beyond the saved matrix. Prefer the
    # dimension that looks like a vocabulary axis rather than a hidden width.
    if weight.shape[0] >= expected_vocab and weight.shape[0] > weight.shape[1]:
        return weight
    if weight.shape[1] >= expected_vocab and weight.shape[1] > weight.shape[0]:
        return weight.T.contiguous()
    return weight


def _visible_token(raw: str) -> str:
    text = raw
    text = text.replace("▁", "␠")
    text = text.replace("Ġ", "␠")
    text = text.replace("Ċ", "↵")
    text = text.replace("ĉ", "⇥")
    text = text.replace("\r", "␍")
    text = text.replace("\n", "↵")
    text = text.replace("\t", "⇥")
    if text == " ":
        return "␠"
    if text == "":
        return "∅"
    return text


def _search_variants(raw: str, display: str) -> str:
    variants = {raw, display, raw.strip(), display.strip()}
    for marker in ("▁", "Ġ"):
        if raw.startswith(marker):
            variants.add(raw[len(marker) :])
    stripped = raw.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        inner = stripped[1:-1].strip()
        variants.add(inner)
        if inner.casefold().startswith("0x"):
            variants.add(inner[2:])
    return "\u0000".join(value.casefold() for value in variants if value)


def _build_token_infos(tokenizer: Any, vocab_size: int) -> list[TokenInfo]:
    raw_by_id: list[str | None] = [None] * vocab_size
    try:
        for token, token_id in tokenizer.get_vocab().items():
            token_id = int(token_id)
            if 0 <= token_id < vocab_size:
                raw_by_id[token_id] = str(token)
    except Exception:
        pass

    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", []) if isinstance(value, int)}
    infos: list[TokenInfo] = []
    for token_id in range(vocab_size):
        raw = raw_by_id[token_id]
        present = raw is not None
        if raw is None:
            try:
                converted = tokenizer.convert_ids_to_tokens(token_id)
                raw = str(converted) if converted is not None else None
            except Exception:
                raw = None
        if raw is None:
            raw = f"<missing-token-{token_id}>"
        display = _visible_token(raw)
        infos.append(
            TokenInfo(
                token_id=token_id,
                raw=raw,
                display=display,
                search_blob=_search_variants(raw, display),
                special=token_id in special_ids,
                present_in_tokenizer=present,
            )
        )
    return infos


def _vector_magnitudes(weight: torch.Tensor, block_size: int = 4096) -> torch.Tensor:
    output = torch.empty(weight.shape[0], dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        for start in range(0, weight.shape[0], block_size):
            end = min(start + block_size, weight.shape[0])
            output[start:end] = torch.linalg.vector_norm(weight[start:end].to(dtype=torch.float32), ord=2, dim=1)
    return output


def load_model(
    model_name: str,
    *,
    revision: str = "main",
    matrix_source: str = "auto",
    compute_device: str = "auto",
    allow_download: bool = True,
    force_reload: bool = False,
) -> ModelAssets:
    global _ASSETS
    name = model_name.strip()
    if not name:
        raise ValueError("Model name cannot be blank.")
    source = matrix_source.strip().casefold()
    if source not in {"auto", "input", "output"}:
        raise ValueError("matrix_source must be auto, input, or output.")
    revision = revision.strip() or "main"
    device = resolve_compute_device(compute_device)

    with _ASSETS_LOCK:
        if (
            not force_reload
            and _ASSETS is not None
            and _ASSETS.model_name == name
            and _ASSETS.revision == revision
            and (_ASSETS.matrix_source == source or source == "auto")
            and _ASSETS.compute_device == device
        ):
            return _ASSETS

        _, _, _, AutoTokenizer = _import_transformers()
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            local_files_only=not allow_download,
            trust_remote_code=DEFAULT_TRUST_REMOTE_CODE,
        )
        expected_vocab = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))

        strategy = DEFAULT_LOAD_STRATEGY or "auto"
        if strategy not in {"auto", "weight_only", "safetensors", "full_model"}:
            raise ValueError("MODEL_LOAD_STRATEGY must be auto, weight_only, safetensors, or full_model.")

        weight_error: Exception | None = None
        if strategy in {"auto", "weight_only", "safetensors"}:
            try:
                weight, tensor_name, actual_source = _load_weight_only(
                    name,
                    revision=revision,
                    requested_source=source,
                    allow_download=allow_download,
                )
                used_strategy = "safetensors weight-only"
            except Exception as exc:
                weight_error = exc
                if strategy in {"weight_only", "safetensors"}:
                    raise RuntimeError(f"Weight-only model loading failed: {exc}") from exc
        if strategy == "full_model" or (strategy == "auto" and weight_error is not None):
            try:
                weight, tensor_name, actual_source = _load_full_model(
                    name,
                    revision=revision,
                    requested_source=source,
                    allow_download=allow_download,
                )
                used_strategy = "full Transformers model fallback"
            except Exception as exc:
                if weight_error is not None:
                    raise RuntimeError(
                        f"Could not load a vocabulary matrix. Weight-only loading failed with: {weight_error}. "
                        f"Full-model loading failed with: {exc}"
                    ) from exc
                raise

        weight = _orient_weight(weight, expected_vocab)
        if weight.ndim != 2:
            raise ValueError(f"Loaded tensor has shape {tuple(weight.shape)}, expected [vocab, hidden].")
        if weight.shape[0] <= 1 or weight.shape[1] <= 0:
            raise ValueError(f"Loaded tensor has unusable shape {tuple(weight.shape)}.")
        if not weight.is_floating_point():
            weight = weight.to(dtype=torch.float32)
        weight = weight.cpu().contiguous()
        magnitudes = _vector_magnitudes(weight)
        token_infos = _build_token_infos(tokenizer, int(weight.shape[0]))

        _ASSETS = ModelAssets(
            model_name=name,
            revision=revision,
            matrix_source=actual_source,
            tensor_name=tensor_name,
            tokenizer=tokenizer,
            token_infos=token_infos,
            weight=weight,
            magnitudes=magnitudes,
            compute_device=device,
            load_strategy=used_strategy,
        )
        return _ASSETS


def unload_model() -> None:
    global _ASSETS
    with _ASSETS_LOCK:
        _ASSETS = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def current_assets() -> ModelAssets:
    with _ASSETS_LOCK:
        if _ASSETS is None:
            raise ValueError("No model is loaded. Enter a Hugging Face model ID and click Load model.")
        return _ASSETS


def model_status() -> dict[str, Any]:
    with _ASSETS_LOCK:
        if _ASSETS is None:
            return {
                "loaded": False,
                "model_name": DEFAULT_MODEL_NAME,
                "revision": "main",
                "matrix_source": "auto",
                "tensor_name": None,
                "vocab_size": 0,
                "hidden_dim": 0,
                "dtype": None,
                "compute_device": resolve_compute_device(DEFAULT_COMPUTE_DEVICE),
                "memory_bytes": 0,
                "load_strategy": None,
            }
        return {
            "loaded": True,
            "model_name": _ASSETS.model_name,
            "revision": _ASSETS.revision,
            "matrix_source": _ASSETS.matrix_source,
            "tensor_name": _ASSETS.tensor_name,
            "vocab_size": _ASSETS.vocab_size,
            "hidden_dim": _ASSETS.hidden_dim,
            "dtype": _ASSETS.dtype,
            "compute_device": _ASSETS.compute_device,
            "memory_bytes": _ASSETS.memory_bytes,
            "load_strategy": _ASSETS.load_strategy,
        }


def token_record(assets: ModelAssets, token_id: int, *, magnitude: bool = True) -> dict[str, Any]:
    info = assets.token(int(token_id))
    row: dict[str, Any] = {
        "token_id": info.token_id,
        "raw": info.raw,
        "display": info.display,
        "special": info.special,
        "present_in_tokenizer": info.present_in_tokenizer,
    }
    if magnitude:
        row["magnitude"] = float(assets.magnitudes[token_id].item())
    return row


def _parse_id_expression(expression: str, vocab_size: int) -> list[int]:
    text = expression.strip()
    if not text:
        return []
    output: list[int] = []
    seen: set[int] = set()
    for part in re.split(r"[\s,;]+", text):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        values: Iterable[int]
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = [int(part)]
        for value in values:
            if 0 <= value < vocab_size and value not in seen:
                seen.add(value)
                output.append(value)
            if len(output) > MAX_SEARCH_RESULTS:
                raise ValueError(f"ID expression expands to more than {MAX_SEARCH_RESULTS:,} tokens.")
    return output


def search_tokens(
    assets: ModelAssets,
    pattern: str,
    *,
    mode: str = "regex",
    case_sensitive: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    mode = mode.strip().casefold()
    if mode not in {"regex", "literal", "id"}:
        raise ValueError("Search mode must be regex, literal, or id.")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))
    offset = max(0, int(offset))
    if len(pattern) > MAX_SEARCH_PATTERN_LENGTH:
        raise ValueError(f"Search patterns are limited to {MAX_SEARCH_PATTERN_LENGTH} characters.")

    if mode == "id":
        ids = _parse_id_expression(pattern, assets.vocab_size)
        total = len(ids)
        page = ids[offset : offset + limit]
    else:
        if mode == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}") from exc

            def matches(info: TokenInfo) -> bool:
                return bool(regex.search(info.raw) or regex.search(info.display))
        else:
            needle = pattern if case_sensitive else pattern.casefold()

            def matches(info: TokenInfo) -> bool:
                if case_sensitive:
                    return needle in info.raw or needle in info.display
                return needle in info.search_blob

        ids = []
        total = 0
        page: list[int] = []
        for info in assets.token_infos:
            if not matches(info):
                continue
            if offset <= total < offset + limit:
                page.append(info.token_id)
            total += 1

    return {
        "query": pattern,
        "mode": mode,
        "case_sensitive": bool(case_sensitive),
        "offset": offset,
        "limit": limit,
        "total_matches": total,
        "truncated": offset + len(page) < total,
        "results": [token_record(assets, token_id) for token_id in page],
    }


def _flatten_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer outputs to one flat list of token IDs."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids", [])
    if isinstance(value, tuple):
        value = list(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list):
        raise ValueError("Tokenizer returned an unsupported input_ids value.")
    output: list[int] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            output.extend(int(token_id) for token_id in item)
        else:
            output.append(int(item))
    return output


def tokenize_text(
    assets: ModelAssets,
    text: str,
    *,
    add_special_tokens: bool = False,
    max_tokens: int = 5000,
) -> dict[str, Any]:
    """Tokenize arbitrary text with the active model tokenizer.

    The response preserves sequence order and repeated token occurrences. Token
    IDs outside the loaded vector matrix are returned as non-projectable rather
    than silently discarded, which can happen with added tokenizer tokens.
    """
    source_text = str(text or "")
    if len(source_text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text is limited to {MAX_TEXT_LENGTH:,} characters.")
    limit = max(1, min(int(max_tokens), MAX_TOKENIZE_RESULTS))
    tokenizer = assets.tokenizer
    if tokenizer is None:
        raise ValueError("The active model does not expose a tokenizer.")

    try:
        encoded = tokenizer.encode(source_text, add_special_tokens=bool(add_special_tokens))
    except Exception:
        try:
            encoded = tokenizer(
                source_text,
                add_special_tokens=bool(add_special_tokens),
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except Exception as exc:
            raise ValueError(f"Tokenizer failed on the supplied text: {exc}") from exc

    token_ids = _flatten_token_ids(encoded)
    total_count = len(token_ids)
    returned_ids = token_ids[:limit]
    rows: list[dict[str, Any]] = []
    projectable_ids: set[int] = set()

    for sequence_index, token_id in enumerate(returned_ids):
        projectable = 0 <= token_id < assets.vocab_size
        if projectable:
            row = token_record(assets, token_id)
            projectable_ids.add(token_id)
        else:
            try:
                converted = tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                converted = None
            raw = str(converted) if converted is not None else f"<token-{token_id}>"
            row = {
                "token_id": token_id,
                "raw": raw,
                "display": _visible_token(raw),
                "special": token_id in set(getattr(tokenizer, "all_special_ids", []) or []),
                "present_in_tokenizer": converted is not None,
                "magnitude": None,
            }
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            decoded = row["display"]
        rows.append(
            {
                **row,
                "sequence_index": sequence_index,
                "decoded": str(decoded),
                "projectable": projectable,
            }
        )

    return {
        "text_length": len(source_text),
        "token_count": total_count,
        "returned_count": len(rows),
        "unique_token_count": len(set(token_ids)),
        "projectable_token_count": len(projectable_ids),
        "add_special_tokens": bool(add_special_tokens),
        "truncated": total_count > len(rows),
        "tokens": rows,
    }


@torch.inference_mode()
def nearest_neighbors(
    assets: ModelAssets,
    anchor_id: int,
    *,
    limit: int = 100,
    include_anchor: bool = True,
    block_size: int = 4096,
) -> list[dict[str, Any]]:
    assets.token(anchor_id)
    limit = max(1, min(int(limit), min(MAX_SELECTED_TOKENS, assets.vocab_size)))
    device = assets.compute_device
    anchor = assets.weight[anchor_id].to(device=device, dtype=torch.float32)
    anchor_norm = torch.linalg.vector_norm(anchor).clamp_min(1e-12)
    cosine_cpu = torch.empty(assets.vocab_size, dtype=torch.float32)
    for start in range(0, assets.vocab_size, block_size):
        end = min(start + block_size, assets.vocab_size)
        block = assets.weight[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        norms = assets.magnitudes[start:end].to(device=device, dtype=torch.float32, non_blocking=True).clamp_min(1e-12)
        cosine = (block @ anchor) / (norms * anchor_norm)
        cosine_cpu[start:end] = cosine.clamp(-1.0, 1.0).cpu()
        del block, norms, cosine
    if not include_anchor:
        cosine_cpu[anchor_id] = -float("inf")
    k = min(limit, assets.vocab_size - (0 if include_anchor else 1))
    values, indices = torch.topk(cosine_cpu, k=k, largest=True, sorted=True)
    rows: list[dict[str, Any]] = []
    for rank, (token_id, cosine) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        row = token_record(assets, int(token_id))
        row.update(
            {
                "rank": rank,
                "cosine_similarity": float(cosine),
                "angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cosine))))),
            }
        )
        rows.append(row)
    return rows


def id_window(assets: ModelAssets, center_id: int, count: int) -> list[dict[str, Any]]:
    assets.token(center_id)
    count = max(1, min(int(count), MAX_SELECTED_TOKENS, assets.vocab_size))
    start = center_id - count // 2
    start = max(0, min(start, assets.vocab_size - count))
    return [token_record(assets, token_id) for token_id in range(start, start + count)]


def selected_vectors(assets: ModelAssets, token_ids: list[int]) -> tuple[list[int], np.ndarray]:
    if not token_ids:
        raise ValueError("Select at least two token IDs.")
    if len(token_ids) > MAX_SELECTED_TOKENS:
        raise ValueError(f"At most {MAX_SELECTED_TOKENS:,} tokens can be projected at once.")
    unique: list[int] = []
    seen: set[int] = set()
    for raw_id in token_ids:
        token_id = int(raw_id)
        assets.token(token_id)
        if token_id not in seen:
            seen.add(token_id)
            unique.append(token_id)
    if len(unique) < 2:
        raise ValueError("Select at least two distinct token IDs.")
    vectors = assets.weight[unique].to(dtype=torch.float32, device="cpu").numpy()
    return unique, vectors


def selected_token_rows(
    assets: ModelAssets,
    token_ids: list[int],
    *,
    anchor_id: int | None,
) -> list[dict[str, Any]]:
    rows = [token_record(assets, token_id) for token_id in token_ids]
    if anchor_id is None:
        for row in rows:
            row["cosine_to_anchor"] = None
            row["angle_to_anchor_deg"] = None
        return rows
    assets.token(anchor_id)
    anchor = assets.weight[anchor_id].to(dtype=torch.float32)
    anchor_norm = assets.magnitudes[anchor_id].clamp_min(1e-12)
    vectors = assets.weight[token_ids].to(dtype=torch.float32)
    norms = assets.magnitudes[token_ids].clamp_min(1e-12)
    cosine = ((vectors @ anchor) / (norms * anchor_norm)).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosine))
    for row, cos, angle in zip(rows, cosine.tolist(), angles.tolist()):
        row["cosine_to_anchor"] = float(cos)
        row["angle_to_anchor_deg"] = float(angle)
    return rows


def _candidate_cache_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.getenv(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[str] = set()
    output: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            output.append(path)
    return output


def list_local_models() -> list[LocalModelInfo]:
    records: dict[str, LocalModelInfo] = {}
    try:
        from huggingface_hub import scan_cache_dir

        for cache_dir in _candidate_cache_dirs():
            if not cache_dir.exists():
                continue
            try:
                info = scan_cache_dir(cache_dir=cache_dir)
            except Exception:
                continue
            for repo in getattr(info, "repos", []):
                if getattr(repo, "repo_type", "model") not in {None, "model"}:
                    continue
                repo_id = str(getattr(repo, "repo_id", "") or "")
                if not repo_id:
                    continue
                modified = getattr(repo, "last_modified", None)
                if hasattr(modified, "timestamp"):
                    modified = modified.timestamp()
                records[repo_id] = LocalModelInfo(
                    model_name=repo_id,
                    cache_path=str(getattr(repo, "repo_path", "") or "") or None,
                    size_bytes=int(getattr(repo, "size_on_disk", 0) or 0) or None,
                    last_modified=float(modified) if modified is not None else None,
                )
    except Exception:
        pass

    for cache_dir in _candidate_cache_dirs():
        if not cache_dir.is_dir():
            continue
        for path in cache_dir.glob("models--*"):
            if not path.is_dir():
                continue
            name = path.name[len("models--") :].replace("--", "/")
            if name in records:
                continue
            try:
                modified = path.stat().st_mtime
            except OSError:
                modified = None
            records[name] = LocalModelInfo(name, str(path), None, modified)
    return sorted(records.values(), key=lambda item: item.model_name.casefold())


_AUX_TENSOR_CACHE: dict[tuple[str, str, bool], dict[str, torch.Tensor]] = {}


def _all_safetensor_tensors(model_name: str, *, revision: str, allow_download: bool = True) -> dict[str, torch.Tensor]:
    cache_key = (model_name, revision, allow_download)
    cached = _AUX_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tensors: dict[str, torch.Tensor] = {}
    index_path = _try_repo_file(
        model_name, "model.safetensors.index.json", revision=revision, allow_download=allow_download
    )
    if index_path is not None:
        with index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        weight_map = data.get("weight_map") or {}
        files = sorted(set(str(value) for value in weight_map.values() if isinstance(value, str)))
    else:
        files = _list_safetensors_files(model_name, revision=revision, allow_download=allow_download)
    for filename in files:
        path = _try_repo_file(model_name, filename, revision=revision, allow_download=allow_download)
        if path is None:
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                lname = str(name).casefold()
                if (
                    lname.endswith(("norm.weight", "norm.bias", "layernorm.weight", "layernorm.bias"))
                    or ".v_proj.weight" in lname
                    or ".value.weight" in lname
                    or ".o_proj.weight" in lname
                    or ".out_proj.weight" in lname
                ):
                    tensors[str(name)] = (
                        handle.get_tensor(str(name)).detach().cpu().to(dtype=torch.float32).contiguous()
                    )
    _AUX_TENSOR_CACHE[cache_key] = tensors
    return tensors


def _layer_prefix(tensor_name: str, layer: int) -> str:
    markers = (f"layers.{layer}.", f"h.{layer}.", f"blocks.{layer}.", f"layer.{layer}.")
    for marker in markers:
        if marker in tensor_name:
            return tensor_name.split(marker, 1)[0] + marker
    return ""


def _find_tensor(
    tensors: dict[str, torch.Tensor], needles: Iterable[str], *, layer: int | None = None, ndim: int | None = None
) -> tuple[str, torch.Tensor]:
    names = list(tensors)
    if layer is not None:
        names = [name for name in names if _layer_prefix(name, layer)]
    needle_list = [needle.casefold() for needle in needles]
    matches = [name for name in names if any(needle in name.casefold() for needle in needle_list)]
    if ndim is not None:
        matches = [name for name in matches if tensors[name].ndim == ndim]
    if not matches:
        scope = f" layer {layer}" if layer is not None else ""
        raise ValueError(f"Could not find model tensor matching {', '.join(needles)}{scope}.")
    return sorted(matches, key=lambda name: (len(name), name))[0], tensors[
        sorted(matches, key=lambda name: (len(name), name))[0]
    ]


def _norm_weight_for_args(
    tensors: dict[str, torch.Tensor], args: list[float | str | np.ndarray]
) -> tuple[np.ndarray, torch.Tensor]:
    if len(args) == 2 and str(args[1]).casefold() == "final":
        vector = np.asarray(args[0], dtype=np.float64)
        _name, weight = _find_tensor(
            tensors, ("model.norm.weight", "final_layernorm.weight", "ln_f.weight", "norm.weight"), ndim=1
        )
        return vector, weight
    if len(args) != 4:
        raise ValueError("norm/invnorm expect (vector, layer, 'attn'|'ffn', 'input'|'output') or (vector, 'final').")

    vector = np.asarray(args[0], dtype=np.float64)
    layer = int(args[1])
    block = str(args[2]).casefold()
    position = str(args[3]).casefold()
    if block in {"attn", "attention"} and position in {"input", "before", "in"}:
        needles = ("input_layernorm.weight", "ln_1.weight", "attention_norm.weight")
    elif block in {"attn", "attention"} and position in {"output", "after", "out"}:
        needles = ("post_attention_layernorm.weight", "post_attention_norm.weight", "ln_2.weight")
    elif block in {"ffn", "mlp"} and position in {"input", "before", "in"}:
        needles = ("pre_feedforward_layernorm.weight", "post_attention_layernorm.weight", "ln_2.weight")
    elif block in {"ffn", "mlp"} and position in {"output", "after", "out"}:
        needles = ("post_feedforward_layernorm.weight", "post_feedforward_norm.weight")
    else:
        raise ValueError("norm/invnorm expect (vector, layer, 'attn'|'ffn', 'input'|'output') or (vector, 'final').")
    _name, weight = _find_tensor(tensors, needles, layer=layer, ndim=1)
    return vector, weight


def _apply_rms_norm(vector: np.ndarray, weight: torch.Tensor) -> np.ndarray:
    w = weight.numpy().astype(np.float64)
    if w.shape[0] != vector.shape[0]:
        raise ValueError(f"Norm width {w.shape[0]} does not match vector width {vector.shape[0]}.")
    return vector * w / math.sqrt(float(np.mean(vector * vector)) + 1e-6)


def _apply_inverse_rms_norm(vector: np.ndarray, weight: torch.Tensor) -> np.ndarray:
    w = weight.numpy().astype(np.float64)
    if w.shape[0] != vector.shape[0]:
        raise ValueError(f"Norm width {w.shape[0]} does not match vector width {vector.shape[0]}.")
    if np.any(np.abs(w) <= 1e-15):
        raise ValueError("Cannot invert a norm with zero or near-zero weights.")
    unweighted = vector / w
    normalized_square_mean = float(np.mean(unweighted * unweighted))
    if normalized_square_mean >= 1.0:
        raise ValueError("Cannot invert this norm result because its implied pre-norm magnitude is not finite.")
    scale = math.sqrt(1e-6 / max(1.0 - normalized_square_mean, 1e-15))
    return unweighted * scale


def _attention_head_counts(assets: ModelAssets) -> tuple[int, int]:
    config_path = _try_repo_file(assets.model_name, "config.json", revision=assets.revision, allow_download=True)
    if config_path is None:
        return 1, 1
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    attention_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
    key_value_heads = int(config.get("num_key_value_heads") or attention_heads or 0)
    return max(attention_heads, 1), max(key_value_heads, 1)


def _wov_head_slices(
    *,
    v_output_width: int,
    o_input_width: int,
    attention_heads: int,
    key_value_heads: int,
    head: int,
) -> tuple[slice, slice, int]:
    if attention_heads <= 0:
        attention_heads = 1
    if key_value_heads <= 0:
        key_value_heads = attention_heads
    if o_input_width % attention_heads != 0:
        # Fall back to treating Wo as one concatenated head space when the config
        # does not divide the tensor cleanly.
        attention_heads = 1
    head_dim = o_input_width // attention_heads
    if head_dim <= 0:
        raise ValueError("Wo input width is not usable for head slicing.")
    if head < 0 or head >= attention_heads:
        raise ValueError(f"head must be in [0, {attention_heads - 1}], got {head}.")

    if v_output_width == o_input_width:
        key_value_heads = attention_heads
    elif v_output_width % key_value_heads != 0:
        if v_output_width == head_dim:
            key_value_heads = 1
        elif v_output_width % head_dim == 0:
            key_value_heads = v_output_width // head_dim
        else:
            raise ValueError("Wv output width cannot be partitioned into Wo-compatible heads.")

    v_head_dim = v_output_width // key_value_heads
    if v_head_dim != head_dim:
        raise ValueError("Wv head width and Wo head width do not match.")
    kv_head = min((head * key_value_heads) // attention_heads, key_value_heads - 1)
    v_start = kv_head * head_dim
    o_start = head * head_dim
    return slice(v_start, v_start + head_dim), slice(o_start, o_start + head_dim), attention_heads


def model_vector_function(assets: ModelAssets, name: str, args: list[float | str | np.ndarray]) -> np.ndarray:
    tensors = _all_safetensor_tensors(assets.model_name, revision=assets.revision, allow_download=True)
    if name in {"norm", "invnorm", "inverse_norm"}:
        vector, weight = _norm_weight_for_args(tensors, args)
        if name == "norm":
            return _apply_rms_norm(vector, weight)
        return _apply_inverse_rms_norm(vector, weight)
    if name == "wov":
        if len(args) != 3:
            raise ValueError("wov expects wov(vector, layer, head).")
        vector = np.asarray(args[0], dtype=np.float64)
        layer = int(args[1])
        head = int(args[2])
        _vn, v_weight = _find_tensor(
            tensors,
            ("self_attn.v_proj.weight", "attention.v_proj.weight", "attn.v_proj.weight", "value.weight"),
            layer=layer,
            ndim=2,
        )
        _on, o_weight = _find_tensor(
            tensors,
            ("self_attn.o_proj.weight", "attention.o_proj.weight", "attn.o_proj.weight", "out_proj.weight"),
            layer=layer,
            ndim=2,
        )
        v = v_weight.numpy().astype(np.float64)
        o = o_weight.numpy().astype(np.float64)
        if v.shape[1] != vector.shape[0] or o.shape[0] != vector.shape[0]:
            raise ValueError("Attention projection widths do not match the selected vector space.")
        attention_heads, key_value_heads = _attention_head_counts(assets)
        v_slice, o_slice, _head_count = _wov_head_slices(
            v_output_width=v.shape[0],
            o_input_width=o.shape[1],
            attention_heads=attention_heads,
            key_value_heads=key_value_heads,
            head=head,
        )
        # Multiplying by the selected Wo slice is equivalent to padding the
        # selected Wv-head output with zeros before/after its head slot and then
        # applying the full Wo matrix, but avoids materializing the padded vector.
        return (vector @ v[v_slice, :].T) @ o[:, o_slice].T
    raise ValueError(f"Unknown model function {name!r}.")

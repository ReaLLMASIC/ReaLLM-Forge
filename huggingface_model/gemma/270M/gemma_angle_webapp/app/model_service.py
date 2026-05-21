from __future__ import annotations

import csv
import gc
import io
import json
import os
import threading
import time
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-3-270m")
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")
DEFAULT_ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "eager")
DEFAULT_TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").casefold() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_MODEL_LOAD_STRATEGY = os.getenv("MODEL_LOAD_STRATEGY", "auto").strip().casefold()
DEFAULT_PAIRWISE_BLOCK_SIZE = int(os.getenv("PAIRWISE_BLOCK_SIZE", "2048"))
PAIRWISE_ANGLE_BIN_DEGREES = 5.0
PAIRWISE_MAX_ANGLE_DEGREES = 90.0


@dataclass(frozen=True)
class TokenInfo:
    token_id: int
    raw: str
    display: str
    normalized: str
    present_in_tokenizer: bool = True


@dataclass(frozen=True)
class LocalModelInfo:
    model_name: str
    cache_path: str | None = None
    size_bytes: int | None = None
    last_modified: float | None = None


@dataclass(frozen=True)
class ModelAssets:
    model_name: str
    requested_device: str
    effective_device: str
    tokenizer: PreTrainedTokenizerBase
    token_infos: list[TokenInfo]
    weight: torch.Tensor
    magnitudes: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.weight.shape[1])

    def token(self, token_id: int) -> TokenInfo:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"token_id must be in [0, {self.vocab_size - 1}], got {token_id}")
        return self.token_infos[token_id]


@dataclass(frozen=True)
class PairwiseBinTokenCache:
    """Small in-memory cache of unique token membership for the last bin run.

    This deliberately stores only a ``num_bins × vocab_size`` boolean matrix,
    not the pairwise dot-product/angle matrix and not token-pair rows.
    """

    model_name: str
    vocab_size: int
    hidden_dim: int
    bin_degrees: float
    angle_min_deg: float
    angle_max_deg: float
    block_size: int
    compute_device: str
    include_self: bool
    created_at: float
    token_membership: torch.Tensor



_ASSETS_LOCK = threading.RLock()
_ASSETS: ModelAssets | None = None
_PAIRWISE_BIN_TOKEN_CACHE_LOCK = threading.RLock()
_PAIRWISE_BIN_TOKEN_CACHE: PairwiseBinTokenCache | None = None


def _norm(text: str) -> str:
    return text.casefold().strip()


def _display_token(token: str) -> str:
    cleaned = token.replace("▁", " ")
    return cleaned.encode("utf-8", "replace").decode("utf-8")


_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _dedupe_texts(values: list[str]) -> tuple[str, ...]:
    """Return non-empty strings in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _search_text_variants(raw: str, display: str) -> tuple[str, ...]:
    """Build literal-search variants for a token.

    Search is intentionally case-insensitive substring matching, not pattern matching.
    Besides the raw and display text, we include common token-content aliases so
    byte-fallback tokens like ``<0xF9>`` can still be found with plain ``F9``.
    """
    values: list[str] = []

    for text in (raw, display):
        if text is None:
            continue
        stripped = text.strip()
        values.extend([text, stripped])

        if stripped.startswith("▁"):
            values.append(stripped[1:])

        space_normalized = stripped.replace("▁", " ")
        values.extend([space_normalized, space_normalized.strip()])

        if stripped.startswith("<") and stripped.endswith(">"):
            inner = stripped[1:-1].strip()
            if inner and "<" not in inner and ">" not in inner:
                values.append(inner)
                if inner.casefold().startswith("0x") and len(inner) > 2:
                    hex_part = inner[2:]
                    if all(char in _HEX_DIGITS for char in hex_part):
                        values.append(hex_part)

    return _dedupe_texts(values)


def _normalized_search_text(raw: str, display: str) -> str:
    return " ".join(_norm(value) for value in _search_text_variants(raw, display))

def _build_token_infos(tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> list[TokenInfo]:
    """Return TokenInfo records indexed by token_id.

    The Streamlit version sorted vocab entries and then indexed into the sorted list.
    This version explicitly indexes by token_id so it keeps working if a tokenizer has
    sparse or out-of-order vocabulary IDs.
    """
    infos: list[TokenInfo | None] = [None] * vocab_size
    for tok, idx in tokenizer.get_vocab().items():
        if 0 <= idx < vocab_size:
            display = _display_token(tok)
            infos[idx] = TokenInfo(
                token_id=idx,
                raw=tok,
                display=display,
                normalized=_normalized_search_text(tok, display),
            )

    for idx, info in enumerate(infos):
        if info is None:
            placeholder = f"<missing-token-{idx}>"
            infos[idx] = TokenInfo(
                token_id=idx,
                raw=placeholder,
                display=placeholder,
                normalized=_normalized_search_text(placeholder, placeholder),
                present_in_tokenizer=False,
            )

    return [info for info in infos if info is not None]


def resolve_device(requested_device: str | None = None) -> str:
    requested = (requested_device or DEFAULT_DEVICE or "cpu").strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    # Validate the device string early so bad env vars fail clearly.
    try:
        torch.device(requested)
    except Exception as exc:  # pragma: no cover - exact exception differs by torch version
        raise ValueError(f"Invalid torch device {requested!r}") from exc
    return requested


def _tokenizer_from_pretrained_kwargs(*, allow_download: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": not allow_download}
    if DEFAULT_TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    return kwargs


def _model_from_pretrained_kwargs(*, allow_download: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = _tokenizer_from_pretrained_kwargs(allow_download=allow_download)
    if DEFAULT_ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = DEFAULT_ATTN_IMPLEMENTATION
    return kwargs


def _load_tokenizer(model_name: str, *, allow_download: bool = True) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        model_name,
        **_tokenizer_from_pretrained_kwargs(allow_download=allow_download),
    )


def _load_causal_lm(model_name: str, *, allow_download: bool = True):
    """Load a causal LM, falling back if a model rejects attn_implementation.

    Gemma works with the default eager-attention kwarg, but some other Hugging
    Face model classes either do not accept that kwarg or reject the requested
    implementation. Dropping only that kwarg makes the browser model picker more
    forgiving while preserving the original default behavior when possible.
    """
    kwargs = _model_from_pretrained_kwargs(allow_download=allow_download)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError) as exc:
        if "attn_implementation" not in kwargs:
            raise
        if isinstance(exc, ValueError) and "attn_implementation" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)



_OUTPUT_WEIGHT_CANDIDATE_NAMES: tuple[str, ...] = (
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
    "embed_out.weight",
    "gpt_neox.embed_out.weight",
    "output.weight",
    # Many tied-output models save only the input embedding matrix.  For this
    # app that is still the correct vocabulary vector matrix when weights are
    # tied, and it avoids importing architecture-specific model classes.
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
    "decoder.embed_tokens.weight",
    "embed_tokens.weight",
)

_OUTPUT_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".lm_head.weight",
    ".embed_out.weight",
    ".output.weight",
    ".embed_tokens.weight",
    ".wte.weight",
    ".embed_in.weight",
)


def _local_model_path(model_name: str) -> Path | None:
    """Return an existing local model directory/file path, if model_name is one."""
    try:
        path = Path(model_name).expanduser()
    except (TypeError, ValueError):
        return None
    return path if path.exists() else None


def _download_or_find_repo_file(
    model_name: str,
    filename: str,
    *,
    allow_download: bool,
) -> Path:
    """Resolve a file from either a local model directory or the HF Hub cache."""
    local_path = _local_model_path(model_name)
    if local_path is not None:
        if local_path.is_file():
            if local_path.name == filename:
                return local_path
            raise FileNotFoundError(f"{filename!r} is not available in local file {model_name!r}.")
        candidate = local_path / filename
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{filename!r} is not available in local model directory {str(local_path)!r}.")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - dependency availability varies by env
        raise RuntimeError(
            "huggingface-hub is required to download or resolve remote model files."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_files_only=not allow_download,
        )
    )


def _try_download_or_find_repo_file(
    model_name: str,
    filename: str,
    *,
    allow_download: bool,
) -> Path | None:
    try:
        return _download_or_find_repo_file(model_name, filename, allow_download=allow_download)
    except Exception:
        return None


def _list_repo_safetensors_files(model_name: str, *, allow_download: bool) -> list[str]:
    """Return likely safetensors filenames without downloading tensor contents."""
    local_path = _local_model_path(model_name)
    if local_path is not None:
        if local_path.is_file() and local_path.suffix == ".safetensors":
            return [local_path.name]
        if local_path.is_dir():
            return sorted(path.name for path in local_path.glob("*.safetensors"))
        return []

    if not allow_download:
        return []

    try:
        from huggingface_hub import list_repo_files
    except Exception:
        return []

    try:
        files = list_repo_files(repo_id=model_name, repo_type="model")
    except Exception:
        return []
    return sorted(name for name in files if name.endswith(".safetensors"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {str(path)!r} to contain a JSON object.")
    return data


def _weight_name_rank(name: str) -> tuple[int, int, str]:
    """Sort key for choosing a vocab/output weight tensor from safetensors."""
    for idx, candidate in enumerate(_OUTPUT_WEIGHT_CANDIDATE_NAMES):
        if name == candidate:
            return (0, idx, name)
    for idx, candidate in enumerate(_OUTPUT_WEIGHT_CANDIDATE_NAMES):
        if name.endswith("." + candidate):
            return (1, idx, name)
    for idx, suffix in enumerate(_OUTPUT_WEIGHT_SUFFIXES):
        if name.endswith(suffix):
            return (2, idx, name)
    return (99, 99, name)


def _choose_weight_tensor_name(names: list[str] | tuple[str, ...]) -> str | None:
    ranked = [name for name in names if _weight_name_rank(name)[0] < 99]
    if not ranked:
        return None
    return sorted(ranked, key=_weight_name_rank)[0]


def _safe_open_tensor(path: Path, tensor_name: str) -> torch.Tensor:
    """Read one tensor from a safetensors file without importing the model class."""
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - dependency availability varies by env
        raise RuntimeError(
            "safetensors is required for the weight-only loader. Install it with `pip install safetensors`."
        ) from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        if tensor_name not in available:
            raise KeyError(f"Tensor {tensor_name!r} not found in {str(path)!r}.")
        tensor = handle.get_tensor(tensor_name)
    if tensor.ndim != 2:
        raise ValueError(
            f"Tensor {tensor_name!r} in {str(path)!r} is {tuple(tensor.shape)}, expected a 2D matrix."
        )
    return tensor.detach().to(dtype=torch.float32, device="cpu")


def _load_weight_from_safetensors_index(
    model_name: str,
    *,
    allow_download: bool,
) -> torch.Tensor | None:
    index_path = _try_download_or_find_repo_file(
        model_name,
        "model.safetensors.index.json",
        allow_download=allow_download,
    )
    if index_path is None:
        return None

    index_data = _read_json(index_path)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{str(index_path)!r} does not contain a safetensors weight_map.")

    tensor_name = _choose_weight_tensor_name([str(name) for name in weight_map.keys()])
    if tensor_name is None:
        interesting = ", ".join(sorted(str(name) for name in weight_map.keys())[:20])
        raise ValueError(
            "Could not find an LM-head/output-embedding tensor in model.safetensors.index.json. "
            f"First known tensors: {interesting}"
        )

    shard_name = weight_map.get(tensor_name)
    if not isinstance(shard_name, str) or not shard_name:
        raise ValueError(f"Invalid safetensors shard name for tensor {tensor_name!r}.")
    shard_path = _download_or_find_repo_file(model_name, shard_name, allow_download=allow_download)
    return _safe_open_tensor(shard_path, tensor_name)


def _load_weight_from_single_safetensors(
    model_name: str,
    *,
    allow_download: bool,
) -> torch.Tensor | None:
    filenames = _list_repo_safetensors_files(model_name, allow_download=allow_download)
    preferred = ["model.safetensors", "pytorch_model.safetensors"]
    is_remote = _local_model_path(model_name) is None

    # When we can list files, trust that list.  When we cannot list a remote repo
    # (for example in cache-only mode), try standard single-file names anyway;
    # hf_hub_download with local_files_only=True resolves them if cached and fails
    # cheaply if they are absent.
    if filenames:
        ordered = [name for name in preferred if name in filenames]
        ordered.extend(name for name in filenames if name not in ordered)
    else:
        ordered = list(preferred) if is_remote else []

    # If the repo has many shards but no index, inspecting/downloading all of
    # them would defeat the limited-disk goal.  We inspect local files, and for
    # remote repos we only download when there is a clear single-file checkpoint.
    if is_remote:
        if len(ordered) > 1 and not any(name in preferred for name in ordered):
            return None
        if len(ordered) > 1 and filenames:
            ordered = [name for name in ordered if name in preferred]

    for filename in ordered:
        path = _try_download_or_find_repo_file(model_name, filename, allow_download=allow_download)
        if path is None:
            continue
        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - dependency availability varies by env
            raise RuntimeError(
                "safetensors is required for the weight-only loader. Install it with `pip install safetensors`."
            ) from exc
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tensor_name = _choose_weight_tensor_name(list(handle.keys()))
            if tensor_name is not None:
                return _safe_open_tensor(path, tensor_name)
        except Exception:
            if len(ordered) == 1:
                raise
            continue

    return None


def _load_output_weight_from_safetensors(
    model_name: str,
    *,
    allow_download: bool = True,
) -> torch.Tensor:
    """Load only the vocab/output vector matrix from safetensors.

    This avoids importing architecture-specific classes such as Gemma3ForCausalLM.
    The app only needs the output vectors, so loading the entire model class is a
    fallback rather than the preferred path.
    """
    index_weight = _load_weight_from_safetensors_index(model_name, allow_download=allow_download)
    if index_weight is not None:
        return index_weight

    single_weight = _load_weight_from_single_safetensors(model_name, allow_download=allow_download)
    if single_weight is not None:
        return single_weight

    raise ValueError(
        "Could not locate a usable safetensors LM-head/output-embedding matrix. "
        "Expected tensors such as 'lm_head.weight' or 'model.embed_tokens.weight'."
    )


def _load_output_weight_from_full_model(
    model_name: str,
    *,
    allow_download: bool = True,
    effective_device: str,
) -> torch.Tensor:
    model = _load_causal_lm(model_name, allow_download=allow_download)
    try:
        model.to(effective_device)
        model.eval()
        with torch.inference_mode():
            return _output_embedding_weight(model).detach().to(device="cpu", dtype=torch.float32)
    finally:
        del model
        _release_cached_cuda_memory()


def _load_output_weight_matrix(
    model_name: str,
    *,
    allow_download: bool = True,
    effective_device: str,
) -> torch.Tensor:
    """Load the output weight matrix, preferring a safetensors-only path."""
    strategy = DEFAULT_MODEL_LOAD_STRATEGY or "auto"
    if strategy not in {"auto", "weight_only", "safetensors", "full_model"}:
        raise ValueError(
            "MODEL_LOAD_STRATEGY must be one of: auto, weight_only, safetensors, full_model."
        )

    if strategy in {"auto", "weight_only", "safetensors"}:
        try:
            return _load_output_weight_from_safetensors(model_name, allow_download=allow_download)
        except Exception as weight_only_exc:
            if strategy in {"weight_only", "safetensors"}:
                raise RuntimeError(
                    "Weight-only safetensors loading failed: "
                    f"{_format_model_load_error(weight_only_exc)}"
                ) from weight_only_exc
            fallback_error = weight_only_exc
    else:
        fallback_error = None

    try:
        return _load_output_weight_from_full_model(
            model_name,
            allow_download=allow_download,
            effective_device=effective_device,
        )
    except Exception as full_model_exc:
        if fallback_error is not None:
            raise RuntimeError(
                "Could not load the output vector matrix. "
                "The safetensors weight-only path failed with: "
                f"{_format_model_load_error(fallback_error)}. "
                "The full Transformers model path failed with: "
                f"{_format_model_load_error(full_model_exc)}"
            ) from full_model_exc
        raise


def _format_model_load_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    hints: list[str] = []
    lowered = text.casefold()
    if "gemma3forcausallm" in lowered or "gemma3" in lowered:
        hints.append(
            "Gemma 3 usually requires a recent Transformers build; this app now tries a "
            "safetensors-only weight loader first to avoid importing Gemma3ForCausalLM."
        )
    if "cpp extensions" in lowered or "torchao" in lowered or "fbgemm" in lowered:
        hints.append(
            "An optional compiled extension appears incompatible with the installed Torch; "
            "because the app only needs output vectors, uninstalling/pinning that extension or "
            "using the safetensors loader avoids the full-model import path."
        )
    if hints:
        return text + " Hint: " + " ".join(hints)
    return text


def _output_embedding_weight(model: Any) -> torch.Tensor:
    """Return the LM-head/output-embedding matrix for a causal LM.

    Most decoder-only Hugging Face models expose this as ``model.lm_head.weight``.
    ``get_output_embeddings()`` is the more general interface, so we prefer it and
    keep ``lm_head`` as a fallback.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_embeddings = get_output_embeddings()
        if output_embeddings is not None and getattr(output_embeddings, "weight", None) is not None:
            return output_embeddings.weight

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and getattr(lm_head, "weight", None) is not None:
        return lm_head.weight

    raise ValueError(
        "Could not find an LM-head/output-embedding weight matrix on this model. "
        "Use an AutoModelForCausalLM-compatible Hugging Face repo."
    )


def _release_cached_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vector_magnitudes(weight: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean length of each token vector.

    The output embedding / LM-head weight matrix has shape
    ``[vocab_size, hidden_dim]``. Each row is one token's vector, so reducing over
    ``dim=1`` computes one scalar length per token:

        sqrt(sum_j weight[token_id, j] ** 2)

    This is the vector magnitude / L2 norm / Euclidean length.
    """
    return torch.linalg.vector_norm(weight, ord=2, dim=1)


def _target_model_and_device(
    model_name: str | None,
    requested_device: str | None,
) -> tuple[str, str, str]:
    """Resolve requested model/device, defaulting to the active model when present."""
    if model_name is None and requested_device is None and _ASSETS is not None:
        requested = _ASSETS.requested_device
        return _ASSETS.model_name, requested, _ASSETS.effective_device

    target_model = (model_name or DEFAULT_MODEL_NAME).strip()
    if not target_model:
        raise ValueError("Model name cannot be empty.")

    requested = (requested_device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
    effective = resolve_device(requested)
    return target_model, requested, effective


def get_assets(
    model_name: str | None = None,
    requested_device: str | None = None,
    *,
    force_reload: bool = False,
    allow_download: bool = True,
) -> ModelAssets:
    """Return active model assets, loading or replacing them when needed.

    Calling ``get_assets()`` with no arguments returns the currently active model,
    or lazily loads the configured default model if nothing has been loaded yet.
    Calling it with ``model_name`` switches the active model to that Hugging Face
    repository ID when necessary.
    """
    global _ASSETS

    with _ASSETS_LOCK:
        target_model, requested, effective = _target_model_and_device(model_name, requested_device)

        if (
            not force_reload
            and _ASSETS is not None
            and _ASSETS.model_name == target_model
            and _ASSETS.requested_device == requested
            and _ASSETS.effective_device == effective
        ):
            return _ASSETS

        clear_pairwise_bin_token_cache()

        tokenizer = _load_tokenizer(target_model, allow_download=allow_download)
        weight_cpu = _load_output_weight_matrix(
            target_model,
            allow_download=allow_download,
            effective_device=effective,
        )

        with torch.inference_mode():
            weight = weight_cpu.to(device=effective, dtype=torch.float32, non_blocking=True)
            magnitudes = vector_magnitudes(weight)

        token_infos = _build_token_infos(tokenizer, vocab_size=int(weight.shape[0]))

        # Keep only the tokenizer plus the tensors needed by the app.
        del weight_cpu
        _release_cached_cuda_memory()

        _ASSETS = ModelAssets(
            model_name=target_model,
            requested_device=requested,
            effective_device=effective,
            tokenizer=tokenizer,
            token_infos=token_infos,
            weight=weight,
            magnitudes=magnitudes,
        )
        return _ASSETS


def get_current_assets() -> ModelAssets:
    """Return the already-loaded active model.

    The browser model picker is explicit: the server should not begin a large
    default-model download just because the page loaded or a status request ran.
    Endpoints that need tensors call this helper and return a clear error until
    the user clicks **Load model**.
    """
    with _ASSETS_LOCK:
        if _ASSETS is None:
            raise ValueError(
                "No model is loaded. Enter a Hugging Face model ID and click Load model first."
            )
        return _ASSETS


def get_model_status() -> dict[str, Any]:
    """Return model status without loading any model assets."""
    with _ASSETS_LOCK:
        if _ASSETS is not None:
            return {
                "loaded": True,
                "model_name": _ASSETS.model_name,
                "requested_device": _ASSETS.requested_device,
                "effective_device": _ASSETS.effective_device,
                "vocab_size": _ASSETS.vocab_size,
                "hidden_dim": _ASSETS.hidden_dim,
            }

    requested = (DEFAULT_DEVICE or "cpu").strip() or "cpu"
    try:
        effective = resolve_device(requested)
    except ValueError:
        # Keep the page usable even if DEVICE is misconfigured; the explicit
        # load request will return the detailed validation error.
        effective = requested

    return {
        "loaded": False,
        "model_name": DEFAULT_MODEL_NAME,
        "requested_device": requested,
        "effective_device": effective,
        "vocab_size": 0,
        "hidden_dim": 0,
    }


def load_active_model(
    model_name: str,
    requested_device: str | None = None,
    *,
    force_reload: bool = False,
    allow_download: bool = True,
) -> ModelAssets:
    """Load a Hugging Face causal LM and make it the active app model.

    When allow_download is true, transformers may download missing files from
    the Hub. When false, loading is cache-only via local_files_only=True.
    """
    return get_assets(
        model_name,
        requested_device or DEFAULT_DEVICE,
        force_reload=force_reload,
        allow_download=allow_download,
    )


def active_model_name() -> str:
    """Return the current model name, or the configured default before first load."""
    with _ASSETS_LOCK:
        return _ASSETS.model_name if _ASSETS is not None else DEFAULT_MODEL_NAME



def _coerce_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_hf_cache_dirs() -> list[Path]:
    """Return likely Hugging Face Hub cache directories without requiring hub imports."""
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
    unique: list[Path] = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _record_local_model(
    records: dict[str, LocalModelInfo],
    *,
    model_name: str,
    cache_path: str | None = None,
    size_bytes: int | None = None,
    last_modified: float | None = None,
) -> None:
    if not model_name:
        return

    existing = records.get(model_name)
    if existing is None:
        records[model_name] = LocalModelInfo(
            model_name=model_name,
            cache_path=cache_path,
            size_bytes=size_bytes,
            last_modified=last_modified,
        )
        return

    # Prefer richer metadata when the same repo is seen from multiple scans.
    records[model_name] = LocalModelInfo(
        model_name=model_name,
        cache_path=existing.cache_path or cache_path,
        size_bytes=existing.size_bytes if existing.size_bytes is not None else size_bytes,
        last_modified=(
            max(x for x in [existing.last_modified, last_modified] if x is not None)
            if existing.last_modified is not None or last_modified is not None
            else None
        ),
    )


def _scan_cache_with_hub(records: dict[str, LocalModelInfo]) -> None:
    """Use huggingface_hub.scan_cache_dir when available."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return

    cache_dirs = _candidate_hf_cache_dirs()
    if not cache_dirs:
        cache_dirs = [None]  # type: ignore[list-item]

    for cache_dir in cache_dirs:
        try:
            if cache_dir is not None and not cache_dir.exists():
                continue
            cache_info = scan_cache_dir(cache_dir=cache_dir)
        except Exception:
            continue

        for repo in getattr(cache_info, "repos", []):
            repo_type = getattr(repo, "repo_type", "model")
            if repo_type not in (None, "model"):
                continue
            repo_id = getattr(repo, "repo_id", "") or ""
            repo_path = getattr(repo, "repo_path", None)
            size_on_disk = getattr(repo, "size_on_disk", None)
            last_modified = _coerce_timestamp(getattr(repo, "last_modified", None))
            _record_local_model(
                records,
                model_name=str(repo_id),
                cache_path=str(repo_path) if repo_path else None,
                size_bytes=int(size_on_disk) if size_on_disk is not None else None,
                last_modified=last_modified,
            )


def _scan_cache_manually(records: dict[str, LocalModelInfo]) -> None:
    """Fallback scanner for the standard HF cache layout: models--org--repo."""
    for cache_dir in _candidate_hf_cache_dirs():
        if not cache_dir.exists() or not cache_dir.is_dir():
            continue
        for repo_dir in cache_dir.glob("models--*"):
            if not repo_dir.is_dir():
                continue
            encoded = repo_dir.name[len("models--"):]
            model_name = encoded.replace("--", "/")
            try:
                modified = repo_dir.stat().st_mtime
            except OSError:
                modified = None
            _record_local_model(
                records,
                model_name=model_name,
                cache_path=str(repo_dir),
                size_bytes=None,
                last_modified=modified,
            )


def list_local_models() -> list[LocalModelInfo]:
    """Return Hugging Face model repos already present in the local cache.

    This is a passive cache scan. It does not download anything and it does not
    load model tensors. The standard Hub cache stores model repos as directories
    named like ``models--Qwen--Qwen3.5-0.8B-Base``; those are displayed as
    ``Qwen/Qwen3.5-0.8B-Base`` in the browser dropdown.
    """
    records: dict[str, LocalModelInfo] = {}
    _scan_cache_with_hub(records)
    _scan_cache_manually(records)
    return sorted(records.values(), key=lambda item: item.model_name.casefold())


def _resolve_pairwise_compute_device(requested_device: str | None, assets: ModelAssets) -> str:
    """Resolve the device used for all-pairs binning.

    ``auto`` prefers the tensor's existing CUDA device. If the active model was
    loaded on CPU but CUDA is available, it streams row/column blocks to cuda:0
    so the expensive block matrix multiplies are still GPU accelerated without
    caching the full pairwise matrix.
    """
    requested = (requested_device or "auto").strip().lower()
    if not requested or requested == "auto":
        weight_device = str(assets.weight.device)
        if weight_device.startswith("cuda"):
            return weight_device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    return resolve_device(requested)


def _angle_bin_metadata(
    counts: torch.Tensor,
    *,
    bin_degrees: float,
    token_membership: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Return angle-bin records ordered from the lowest to highest angle range."""
    bins: list[dict[str, Any]] = []
    counts_list = [int(value) for value in counts.detach().cpu().tolist()]
    token_counts: list[int] | None = None
    if token_membership is not None:
        token_counts = [int(value) for value in token_membership.sum(dim=1).detach().cpu().tolist()]

    for bin_index, count in enumerate(counts_list):
        angle_min = float(bin_index * bin_degrees)
        angle_max = float((bin_index + 1) * bin_degrees)
        bins.append(
            {
                # ``rank`` is now the angle-order position, kept for backwards
                # compatibility with the existing response schema.
                "rank": bin_index + 1,
                "bin_index": bin_index,
                "angle_min_deg": angle_min,
                "angle_max_deg": angle_max,
                "label": f"{angle_min:g}–{angle_max:g}°",
                "count": count,
                "token_count": token_counts[bin_index] if token_counts is not None else 0,
            }
        )
    return bins




def _set_pairwise_bin_token_cache(cache: PairwiseBinTokenCache) -> None:
    global _PAIRWISE_BIN_TOKEN_CACHE
    with _PAIRWISE_BIN_TOKEN_CACHE_LOCK:
        _PAIRWISE_BIN_TOKEN_CACHE = cache


def clear_pairwise_bin_token_cache() -> None:
    """Drop the saved unique-token membership for the previous binning run."""
    global _PAIRWISE_BIN_TOKEN_CACHE
    with _PAIRWISE_BIN_TOKEN_CACHE_LOCK:
        _PAIRWISE_BIN_TOKEN_CACHE = None


def _update_pairwise_token_membership(
    token_membership: torch.Tensor,
    bin_indices: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    num_bins: int,
) -> None:
    """Mark every token that participates in at least one pair in each bin.

    ``token_membership`` is CPU bool with shape ``[num_bins, vocab_size]``.
    ``bin_indices``, ``row_indices``, and ``col_indices`` describe the valid
    pairs from the current block on the compute device. This uses one compact
    per-block scatter instead of saving token-pair rows or looping through bins
    with many GPU synchronizations.
    """
    row_len = row_end - row_start
    col_len = col_end - col_start
    if row_len <= 0 or col_len <= 0 or bin_indices.numel() == 0:
        return

    device = bin_indices.device
    row_presence = torch.zeros((num_bins, row_len), dtype=torch.bool, device=device)
    col_presence = torch.zeros((num_bins, col_len), dtype=torch.bool, device=device)
    row_presence[bin_indices, row_indices] = True
    col_presence[bin_indices, col_indices] = True

    token_membership[:, row_start:row_end] |= row_presence.detach().cpu()
    token_membership[:, col_start:col_end] |= col_presence.detach().cpu()
    del row_presence, col_presence


@torch.inference_mode()
def pairwise_angle_distribution(
    assets: ModelAssets,
    *,
    block_size: int = DEFAULT_PAIRWISE_BLOCK_SIZE,
    compute_device: str | None = "auto",
    include_self: bool = False,
    bin_degrees: float = PAIRWISE_ANGLE_BIN_DEGREES,
) -> dict[str, Any]:
    """Return a blockwise all-token acute-angle histogram.

    This computes dot products for unique unordered token pairs ``i < j`` by
    default. Each block is immediately reduced into 5-degree angle bins, so the
    full vocab-by-vocab dot-product matrix is never materialized or written to
    disk. Angles are acute angles in ``[0, 90]`` via ``abs(cosine)``; a signed
    vector angle would instead span ``[0, 180]``.
    """
    vocab_size = assets.vocab_size
    if vocab_size <= 0:
        raise ValueError("The active model has an empty LM-head/output-embedding matrix.")

    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    if bin_degrees <= 0:
        raise ValueError("bin_degrees must be positive.")
    num_bins_float = PAIRWISE_MAX_ANGLE_DEGREES / bin_degrees
    num_bins = int(round(num_bins_float))
    if abs(num_bins_float - num_bins) > 1e-9:
        raise ValueError("bin_degrees must evenly divide 90 degrees.")

    device = _resolve_pairwise_compute_device(compute_device, assets)
    counts = torch.zeros(num_bins, dtype=torch.int64, device="cpu")
    # A compact CPU membership matrix lets the UI show unique tokens per bin
    # without saving pair rows or the full pairwise angle matrix.
    token_membership = torch.zeros((num_bins, vocab_size), dtype=torch.bool, device="cpu")

    # Use cosine thresholds instead of computing arccos for every dot product.
    # Boundaries are increasing: cos(85°), cos(80°), ..., cos(5°).
    inner_angles = torch.arange(
        bin_degrees,
        PAIRWISE_MAX_ANGLE_DEGREES,
        bin_degrees,
        dtype=torch.float32,
        device=device,
    )
    boundaries = torch.cos(torch.deg2rad(torch.flip(inner_angles, dims=[0]))).contiguous()

    started = time.perf_counter()
    weight = assets.weight
    magnitudes = assets.magnitudes

    try:
        for row_start in range(0, vocab_size, block_size):
            row_end = min(row_start + block_size, vocab_size)
            row_weight = weight[row_start:row_end].to(device=device, dtype=torch.float32, non_blocking=True)
            row_norm = magnitudes[row_start:row_end].to(device=device, dtype=torch.float32, non_blocking=True)
            row_vectors = row_weight / row_norm.clamp_min(1e-12).unsqueeze(1)

            for col_start in range(row_start, vocab_size, block_size):
                col_end = min(col_start + block_size, vocab_size)

                if col_start == row_start:
                    col_vectors = row_vectors
                else:
                    col_weight = weight[col_start:col_end].to(device=device, dtype=torch.float32, non_blocking=True)
                    col_norm = magnitudes[col_start:col_end].to(device=device, dtype=torch.float32, non_blocking=True)
                    col_vectors = col_weight / col_norm.clamp_min(1e-12).unsqueeze(1)

                dots = row_vectors @ col_vectors.T
                cos_abs = dots.abs().clamp_(0.0, 1.0).contiguous()
                bucket = torch.bucketize(cos_abs, boundaries, right=False)
                bin_matrix = (num_bins - 1 - bucket).to(torch.int64)

                if col_start == row_start:
                    diagonal = 0 if include_self else 1
                    row_indices, col_indices = torch.triu_indices(
                        bin_matrix.shape[0],
                        bin_matrix.shape[1],
                        offset=diagonal,
                        device=device,
                    )
                    bin_indices = bin_matrix[row_indices, col_indices]
                else:
                    row_len = row_end - row_start
                    col_len = col_end - col_start
                    row_indices = torch.arange(row_len, device=device, dtype=torch.long).repeat_interleave(col_len)
                    col_indices = torch.arange(col_len, device=device, dtype=torch.long).repeat(row_len)
                    bin_indices = bin_matrix.reshape(-1)

                if bin_indices.numel() == 0:
                    del dots, cos_abs, bucket, bin_matrix, bin_indices, row_indices, col_indices
                    if col_start != row_start:
                        del col_vectors
                    continue

                block_counts = torch.bincount(bin_indices, minlength=num_bins)[:num_bins]
                counts += block_counts.detach().cpu()
                _update_pairwise_token_membership(
                    token_membership,
                    bin_indices,
                    row_indices,
                    col_indices,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    num_bins=num_bins,
                )

                del dots, cos_abs, bucket, bin_matrix, bin_indices, row_indices, col_indices, block_counts
                if col_start != row_start:
                    del col_vectors

            del row_weight, row_norm, row_vectors

        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _release_cached_cuda_memory()
            raise RuntimeError(
                f"Pairwise binning ran out of memory on {device} with block_size={block_size}. "
                "Try a smaller block size such as 512 or 1024."
            ) from exc
        raise

    elapsed = time.perf_counter() - started
    total_pairs = vocab_size * (vocab_size - 1) // 2
    if include_self:
        total_pairs += vocab_size

    bins = _angle_bin_metadata(counts, bin_degrees=bin_degrees, token_membership=token_membership)
    _set_pairwise_bin_token_cache(
        PairwiseBinTokenCache(
            model_name=assets.model_name,
            vocab_size=vocab_size,
            hidden_dim=assets.hidden_dim,
            bin_degrees=float(bin_degrees),
            angle_min_deg=0.0,
            angle_max_deg=PAIRWISE_MAX_ANGLE_DEGREES,
            block_size=block_size,
            compute_device=device,
            include_self=include_self,
            created_at=time.time(),
            token_membership=token_membership,
        )
    )

    return {
        "model_name": assets.model_name,
        "vocab_size": vocab_size,
        "hidden_dim": assets.hidden_dim,
        "total_pairs": total_pairs,
        "bin_degrees": float(bin_degrees),
        "angle_min_deg": 0.0,
        "angle_max_deg": PAIRWISE_MAX_ANGLE_DEGREES,
        "block_size": block_size,
        "compute_device": device,
        "include_self": include_self,
        "acute_angle": True,
        "elapsed_seconds": float(elapsed),
        "bins": bins,
    }



def pairwise_angle_bin_tokens(assets: ModelAssets, bin_index: int) -> dict[str, Any]:
    """Return unique tokens participating in the selected bin from the last run.

    The cache is intentionally small: it stores one boolean membership value per
    ``(bin, token_id)`` pair. Token-pair rows and pairwise dot products are never
    saved.
    """
    with _PAIRWISE_BIN_TOKEN_CACHE_LOCK:
        cache = _PAIRWISE_BIN_TOKEN_CACHE
        if cache is None:
            raise ValueError("No pairwise angle-bin token list is available yet. Compute pairwise bins first.")
        if cache.model_name != assets.model_name or cache.vocab_size != assets.vocab_size:
            raise ValueError("The saved pairwise bin token list belongs to a different model. Recompute pairwise bins.")
        num_bins = int(cache.token_membership.shape[0])
        if bin_index < 0 or bin_index >= num_bins:
            raise IndexError(f"bin_index must be in [0, {num_bins - 1}], got {bin_index}")

        membership = cache.token_membership[bin_index].clone()
        bin_degrees = cache.bin_degrees

    token_ids = torch.nonzero(membership, as_tuple=False).flatten().cpu().tolist()
    tokens = [
        {"token_id": int(token_id), "raw": assets.token_infos[int(token_id)].raw, "display": assets.token_infos[int(token_id)].display}
        for token_id in token_ids
    ]
    angle_min = float(bin_index * bin_degrees)
    angle_max = float((bin_index + 1) * bin_degrees)
    return {
        "model_name": assets.model_name,
        "vocab_size": assets.vocab_size,
        "bin_index": int(bin_index),
        "angle_min_deg": angle_min,
        "angle_max_deg": angle_max,
        "label": f"{angle_min:g}–{angle_max:g}°",
        "token_count": len(tokens),
        "tokens": tokens,
    }


def _update_best_cosine_matches(
    best_cos: torch.Tensor,
    best_ids: torch.Tensor,
    target_ids: torch.Tensor,
    candidate_cos: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> None:
    """Update per-token best cosine / nearest-token IDs in place.

    ``best_cos`` and ``best_ids`` live on the compute device.  The arrays are
    intentionally one-dimensional, so this routine keeps only O(vocab) state
    while the caller streams blockwise dot products.  Ties are resolved by the
    lower other-token ID to keep results deterministic.
    """
    if target_ids.numel() == 0:
        return

    current_cos = best_cos[target_ids]
    current_ids = best_ids[target_ids]
    better = (candidate_cos > current_cos) | (
        (candidate_cos == current_cos)
        & ((current_ids < 0) | (candidate_ids < current_ids))
    )
    if not bool(torch.any(better).item()):
        return

    selected = target_ids[better]
    best_cos[selected] = candidate_cos[better]
    best_ids[selected] = candidate_ids[better]


def _minimum_angular_distance_row(
    assets: ModelAssets,
    token_id: int,
    min_angle_deg: float,
    min_angle_rank: int,
    other_token_id: int,
    magnitudes_cpu: list[float],
) -> dict[str, Any]:
    info = assets.token(token_id)
    other = assets.token(other_token_id)
    return {
        "min_angle_rank": int(min_angle_rank),
        "token_id": int(token_id),
        "token_raw": info.raw,
        "token_display": info.display,
        "magnitude": float(magnitudes_cpu[token_id]),
        "min_angle_deg": float(min_angle_deg),
        "other_token_id": int(other_token_id),
        "other_token_raw": other.raw,
        "other_token_display": other.display,
        "other_magnitude": float(magnitudes_cpu[other_token_id]),
    }


@torch.inference_mode()
def minimum_angular_distances(
    assets: ModelAssets,
    *,
    block_size: int = DEFAULT_PAIRWISE_BLOCK_SIZE,
    compute_device: str | None = "auto",
) -> dict[str, Any]:
    """Return the closest non-self token for every vocabulary vector.

    The ordinary signed vector angle in ``[0, 180]`` is used, matching the
    pairwise angle and neighborhood features.  The computation is streamed in
    blocks and keeps only two O(vocab) vectors on the compute device:
    best cosine and best other-token ID.  No pairwise matrix or pair rows are
    cached or written to disk.
    """
    vocab_size = assets.vocab_size
    if vocab_size < 2:
        raise ValueError("At least two tokens are required to compute non-self minimum angles.")

    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    device = _resolve_pairwise_compute_device(compute_device, assets)
    best_cos = torch.full((vocab_size,), -float("inf"), dtype=torch.float32, device=device)
    best_ids = torch.full((vocab_size,), -1, dtype=torch.long, device=device)

    weight = assets.weight
    magnitudes = assets.magnitudes
    started = time.perf_counter()

    try:
        for row_start in range(0, vocab_size, block_size):
            row_end = min(row_start + block_size, vocab_size)
            row_len = row_end - row_start
            row_ids = torch.arange(row_start, row_end, dtype=torch.long, device=device)
            row_weight = weight[row_start:row_end].to(device=device, dtype=torch.float32, non_blocking=True)
            row_norm = magnitudes[row_start:row_end].to(device=device, dtype=torch.float32, non_blocking=True)
            row_vectors = row_weight / row_norm.clamp_min(1e-12).unsqueeze(1)

            for col_start in range(row_start, vocab_size, block_size):
                col_end = min(col_start + block_size, vocab_size)
                col_len = col_end - col_start

                if col_start == row_start:
                    col_vectors = row_vectors
                    col_ids = row_ids
                else:
                    col_ids = torch.arange(col_start, col_end, dtype=torch.long, device=device)
                    col_weight = weight[col_start:col_end].to(device=device, dtype=torch.float32, non_blocking=True)
                    col_norm = magnitudes[col_start:col_end].to(device=device, dtype=torch.float32, non_blocking=True)
                    col_vectors = col_weight / col_norm.clamp_min(1e-12).unsqueeze(1)

                dots = (row_vectors @ col_vectors.T).clamp_(-1.0, 1.0)

                if col_start == row_start:
                    diag = torch.arange(row_len, dtype=torch.long, device=device)
                    dots[diag, diag] = -float("inf")
                    candidate_cos, candidate_local_ids = torch.max(dots, dim=1)
                    candidate_ids = col_ids[candidate_local_ids]
                    _update_best_cosine_matches(
                        best_cos,
                        best_ids,
                        row_ids,
                        candidate_cos,
                        candidate_ids,
                    )
                    del diag, candidate_cos, candidate_local_ids, candidate_ids
                else:
                    row_candidate_cos, row_candidate_local_ids = torch.max(dots, dim=1)
                    row_candidate_ids = col_ids[row_candidate_local_ids]
                    _update_best_cosine_matches(
                        best_cos,
                        best_ids,
                        row_ids,
                        row_candidate_cos,
                        row_candidate_ids,
                    )

                    col_candidate_cos, col_candidate_local_ids = torch.max(dots, dim=0)
                    col_candidate_ids = row_ids[col_candidate_local_ids]
                    _update_best_cosine_matches(
                        best_cos,
                        best_ids,
                        col_ids,
                        col_candidate_cos,
                        col_candidate_ids,
                    )
                    del (
                        row_candidate_cos,
                        row_candidate_local_ids,
                        row_candidate_ids,
                        col_candidate_cos,
                        col_candidate_local_ids,
                        col_candidate_ids,
                    )

                del dots
                if col_start != row_start:
                    del col_ids, col_weight, col_norm, col_vectors

            del row_ids, row_weight, row_norm, row_vectors

        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _release_cached_cuda_memory()
            raise RuntimeError(
                f"Minimum-angle computation ran out of memory on {device} with block_size={block_size}. "
                "Try a smaller block size such as 512 or 1024."
            ) from exc
        raise

    if bool(torch.any(best_ids < 0).item()):
        raise RuntimeError("Could not determine a non-self nearest token for every token.")

    angles = torch.rad2deg(torch.arccos(best_cos.clamp(-1.0, 1.0)))
    angle_order = torch.argsort(angles, stable=True)
    min_angle_ranks = torch.empty(vocab_size, dtype=torch.long, device=device)
    min_angle_ranks[angle_order] = torch.arange(1, vocab_size + 1, dtype=torch.long, device=device)

    elapsed = time.perf_counter() - started
    angles_cpu = angles.detach().cpu().tolist()
    ranks_cpu = min_angle_ranks.detach().cpu().tolist()
    best_ids_cpu = best_ids.detach().cpu().tolist()
    magnitudes_cpu = assets.magnitudes.detach().cpu().tolist()

    rows = [
        _minimum_angular_distance_row(
            assets,
            token_id,
            float(angles_cpu[token_id]),
            int(ranks_cpu[token_id]),
            int(best_ids_cpu[token_id]),
            magnitudes_cpu,
        )
        for token_id in range(vocab_size)
    ]

    return {
        "model_name": assets.model_name,
        "vocab_size": vocab_size,
        "hidden_dim": assets.hidden_dim,
        "total_pairs": vocab_size * (vocab_size - 1) // 2,
        "block_size": block_size,
        "compute_device": device,
        "elapsed_seconds": float(elapsed),
        "rows": rows,
    }

def search_tokens(assets: ModelAssets, query: str) -> list[TokenInfo]:
    """Return every token matching a case-insensitive literal substring query.

    There is deliberately no result cap. A blank query returns the full
    vocabulary in token-id order.
    """
    q = _norm(query)
    if not q:
        return list(assets.token_infos)
    return [info for info in assets.token_infos if q in info.normalized]


@torch.inference_mode()
def angle_degrees(assets: ModelAssets, id_a: int, id_b: int) -> float:
    assets.token(id_a)
    assets.token(id_b)
    denom = (assets.magnitudes[id_a] * assets.magnitudes[id_b]).clamp_min(1e-12)
    cos_val = torch.dot(assets.weight[id_a], assets.weight[id_b]) / denom
    cos_val = torch.clamp(cos_val, -1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos_val)).item())


@torch.inference_mode()
def _angles_for_anchor(assets: ModelAssets, anchor_id: int) -> torch.Tensor:
    assets.token(anchor_id)
    anchor = assets.weight[anchor_id]
    anchor_norm = assets.magnitudes[anchor_id].clamp_min(1e-12)
    denom = (assets.magnitudes * anchor_norm).clamp_min(1e-12)
    cos = (assets.weight @ anchor) / denom
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def _neighborhood_row(assets: ModelAssets, token_id: int, angle_deg: float, rank: int) -> dict[str, Any]:
    info = assets.token(token_id)
    return {
        "rank": rank,
        "token_id": token_id,
        "token_raw": info.raw,
        "token_display": info.display,
        "angle_deg": angle_deg,
        "magnitude": float(assets.magnitudes[token_id].item()),
    }


def _common_close_token_row(
    assets: ModelAssets,
    token_id: int,
    angle_to_token_a_deg: float,
    angle_to_token_b_deg: float,
    rank: int,
) -> dict[str, Any]:
    info = assets.token(token_id)
    return {
        "rank": rank,
        "token_id": token_id,
        "token_raw": info.raw,
        "token_display": info.display,
        "angle_to_token_a_deg": angle_to_token_a_deg,
        "angle_to_token_b_deg": angle_to_token_b_deg,
        "magnitude": float(assets.magnitudes[token_id].item()),
    }


@torch.inference_mode()
def common_close_tokens(
    assets: ModelAssets,
    token_a_id: int,
    token_b_id: int,
    threshold_deg: float = 35.0,
) -> list[dict[str, Any]]:
    """Return tokens whose vector is within ``threshold_deg`` of both anchors.

    This uses the same signed 0°–180° angle definition as the pairwise angle
    endpoint, not the acute ``abs(cosine)`` angle used by the all-pairs global
    distribution. Results are sorted by the worst of the two angles, then the
    sum of both angles, then token ID, so the most jointly close tokens appear
    first.
    """
    assets.token(token_a_id)
    assets.token(token_b_id)
    threshold = float(threshold_deg)
    if threshold < 0:
        raise ValueError("threshold_deg must be non-negative.")

    angles_a = _angles_for_anchor(assets, token_a_id)
    angles_b = _angles_for_anchor(assets, token_b_id)
    mask = (angles_a <= threshold) & (angles_b <= threshold)
    token_ids = torch.nonzero(mask, as_tuple=False).flatten()

    if token_ids.numel() == 0:
        return []

    selected_angles_a = angles_a[token_ids]
    selected_angles_b = angles_b[token_ids]
    max_angles = torch.maximum(selected_angles_a, selected_angles_b)
    sum_angles = selected_angles_a + selected_angles_b

    # Lexicographic sort: max(angle_to_A, angle_to_B), then sum, then token_id.
    # ``torch.nonzero`` returns IDs in ascending order, so stable sorts preserve
    # token ID as the final tie-breaker.
    order = torch.argsort(sum_angles, stable=True)
    token_ids = token_ids[order]
    selected_angles_a = selected_angles_a[order]
    selected_angles_b = selected_angles_b[order]
    max_angles = max_angles[order]

    order = torch.argsort(max_angles, stable=True)
    token_ids = token_ids[order]
    selected_angles_a = selected_angles_a[order]
    selected_angles_b = selected_angles_b[order]

    ids_cpu = token_ids.detach().cpu().tolist()
    angles_a_cpu = selected_angles_a.detach().cpu().tolist()
    angles_b_cpu = selected_angles_b.detach().cpu().tolist()

    return [
        _common_close_token_row(
            assets,
            int(token_id),
            float(angle_a),
            float(angle_b),
            rank,
        )
        for rank, (token_id, angle_a, angle_b) in enumerate(
            zip(ids_cpu, angles_a_cpu, angles_b_cpu),
            start=1,
        )
    ]


@torch.inference_mode()
def nearest_neighbors(
    assets: ModelAssets,
    anchor_id: int,
    limit: int,
    *,
    include_self: bool = True,
) -> list[dict[str, Any]]:
    angles = _angles_for_anchor(assets, anchor_id)
    k = min(assets.vocab_size, max(1, limit + (0 if include_self else 1)))
    values, indices = torch.topk(angles, k=k, largest=False, sorted=True)

    rows: list[dict[str, Any]] = []
    rank = 1
    for value, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        token_id = int(idx)
        if not include_self and token_id == anchor_id:
            continue
        rows.append(_neighborhood_row(assets, token_id, float(value), rank))
        rank += 1
        if len(rows) >= limit:
            break
    return rows


@torch.inference_mode()
def iter_neighborhood_csv(assets: ModelAssets, anchor_id: int) -> Iterator[str]:
    angles = _angles_for_anchor(assets, anchor_id)
    order = torch.argsort(angles, stable=True).detach().cpu().tolist()
    angles_cpu = angles.detach().cpu().tolist()
    magnitudes_cpu = assets.magnitudes.detach().cpu().tolist()

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["rank", "token_id", "angle_deg", "magnitude", "token_raw", "token_display"])
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    for rank, token_id in enumerate(order, start=1):
        info = assets.token(int(token_id))
        writer.writerow(
            [
                rank,
                int(token_id),
                float(angles_cpu[token_id]),
                float(magnitudes_cpu[token_id]),
                info.raw,
                info.display,
            ]
        )
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

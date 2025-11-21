"""Utilities for constructing and applying snap-to-grid projections."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


COMP_ATTENTION = "attn"
COMP_MLP = "mlp"


def _normalize_rows(weight: torch.Tensor, target_dim: Optional[int] = None) -> torch.Tensor:
    """Return a row-wise L2 normalised view of ``weight``.

    When ``target_dim`` is provided the function ensures that the returned
    matrix has exactly that many columns by transposing the weight matrix when
    necessary. This prevents concatenation errors when mixing vectors whose
    natural orientation differs (e.g. attention vs. MLP projection weights).
    """

    tensor = weight.detach().float().cpu()

    if target_dim is None:
        matrix = tensor.reshape(tensor.shape[0], -1)
    else:
        # Prefer views whose trailing dimension already matches ``target_dim``.
        if tensor.ndim >= 2 and tensor.shape[-1] == target_dim:
            matrix = tensor.reshape(-1, target_dim)
        elif tensor.ndim >= 2 and tensor.shape[0] == target_dim:
            matrix = tensor.transpose(0, 1).reshape(-1, target_dim)
        else:
            flat = tensor.reshape(tensor.shape[0], -1)
            if flat.shape[-1] != target_dim:
                raise ValueError(
                    f"Unable to reshape tensor with shape {tuple(tensor.shape)} "
                    f"to have {target_dim} features."
                )
            matrix = flat

    return F.normalize(matrix, p=2, dim=-1)


def _gather_base_vectors(
    model: torch.nn.Module,
    upto_layer: int,
    component: str,
) -> Optional[torch.Tensor]:
    """Collect all source vectors required for ``upto_layer``.

    Parameters
    ----------
    model:
        The GPT model containing the parameters.
    upto_layer:
        Index of the block (0-indexed) whose pre-norm input will snap to the grid.
    component:
        Either ``"attn"`` or ``"mlp"`` depending on the target sub-module.

    Returns
    -------
    Optional[torch.Tensor]
        A tensor of shape ``(N, d)`` where each row has unit norm. ``None`` when
        there are no available vectors (should not normally happen).
    """

    vectors: List[torch.Tensor] = []
    upto_layer = max(int(upto_layer), 0)
    target_dim = getattr(getattr(model, "config", None), "n_embd", None)

    for name, param in model.named_parameters():
        if param.ndim != 2 or not name.endswith("weight"):
            continue

        if "transformer.wte" in name:
            vectors.append(_normalize_rows(param, target_dim))
            continue

        if "transformer.h." not in name:
            continue

        try:
            layer_str = name.split("transformer.h.")[1].split(".")[0]
            layer_idx = int(layer_str)
        except (IndexError, ValueError):
            continue

        if COMP_ATTENTION in name and name.endswith("attn.c_proj.weight"):
            if layer_idx <= upto_layer:
                vectors.append(_normalize_rows(param, target_dim))
        elif COMP_MLP in name and name.endswith("mlp.c_proj.weight"):
            limit = upto_layer if component == COMP_MLP else upto_layer - 1
            if layer_idx <= limit:
                vectors.append(_normalize_rows(param, target_dim))

    if not vectors:
        return None

    return torch.cat(vectors, dim=0)


def _ensure_component_list(component: str | Sequence[str]) -> List[str]:
    if isinstance(component, str):
        if component == "both":
            return [COMP_ATTENTION, COMP_MLP]
        return [component]
    return list(component)


@dataclass
class SnapToGridRegistry:
    """Mapping of layer/component pairs to snap-to-grid vectors."""

    grids: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def set_grid(self, layer_idx: int, component: str, grid: torch.Tensor) -> None:
        self.grids[(layer_idx, component)] = grid.cpu()

    def get_grid(self, layer_idx: int, component: str) -> Optional[torch.Tensor]:
        return self.grids.get((layer_idx, component))

    def clear(self) -> None:
        self.grids.clear()

    def state_dict(self) -> Dict[str, object]:
        return {"grids": self.grids, "metadata": self.metadata}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.grids = {k: v.cpu() for k, v in state.get("grids", {}).items()}
        self.metadata = state.get("metadata", {})


def generate_snap_to_grid_registry(
    model: torch.nn.Module,
    layers: Optional[Iterable[int]],
    component_selection: str | Sequence[str],
    size: int,
    generator: Optional[torch.Generator] = None,
) -> SnapToGridRegistry:
    """Create a registry for the provided ``size`` across all ``layers``."""

    if size is None or size <= 0:
        registry = SnapToGridRegistry()
        registry.metadata["size"] = 0
        return registry

    if layers is None:
        n_layers = getattr(getattr(model, "config", None), "n_layer", None)
        layers = range(1, n_layers or 0)

    component_list = _ensure_component_list(component_selection)

    registry = SnapToGridRegistry(metadata={"size": int(size)})

    base_rng = generator or torch.Generator(device="cpu")
    for layer_idx in layers:
        if layer_idx <= 0:
            continue
        for component in component_list:
            source = _gather_base_vectors(model, layer_idx, component)
            if source is None or source.numel() == 0:
                continue
            coeffs = torch.randn((size, source.size(0)), generator=base_rng)
            combos = coeffs @ source
            combos = F.normalize(combos, p=2, dim=-1)
            registry.set_grid(layer_idx, component, combos)

    return registry


def apply_snap_to_grid_tensor(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Project activations ``x`` onto the closest vector from ``grid``."""

    if grid is None:
        return x

    target = grid.to(device=x.device, dtype=x.dtype, non_blocking=True)
    x_norm = F.normalize(x, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    sims = torch.matmul(x_norm, target_norm.t())
    best = sims.argmax(dim=-1)
    snapped = target_norm.index_select(0, best.reshape(-1))
    snapped = snapped.reshape(*x.shape)
    return snapped


def save_registry(path: str, registry: SnapToGridRegistry) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(registry.state_dict(), path)


def load_registry(path: str) -> SnapToGridRegistry:
    data = torch.load(path, map_location="cpu")
    registry = SnapToGridRegistry()
    registry.load_state_dict(data)
    return registry


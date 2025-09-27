"""Block mask strategy variations for attention layers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

DEFAULT_BLOCK_MASK = "global"
_SENTINEL_NONE = {"", "none", "null"}


class BlockMaskBase(nn.Module):
    """Base class for block mask strategies.

    Subclasses are responsible for producing additive attention biases for
    scaled-dot-product attention, dense masks for the fallback attention path
    (where we manually apply masks before the softmax), and block masks for
    FlexAttention when requested.
    """

    name: str = "global"

    def __init__(self, config, window_size: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        if window_size is not None:
            self._configured_window = window_size
        else:
            self._configured_window = getattr(config, "window_size", None)
        self.selected_name = self.name

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """Whether this strategy applies any masking beyond causal masking."""
        return False

    @property
    def is_sliding(self) -> bool:
        """Whether this strategy represents a sliding-window style mask."""
        return False

    @property
    def supports_flex(self) -> bool:
        """Whether FlexAttention block masks are supported."""
        return False

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    def current_window(self, device=None, dtype=None) -> Optional[torch.Tensor]:
        """Returns the (possibly learned) window size as a tensor."""
        return None

    def discrete_window(self) -> Optional[int]:
        """Returns an integer window size, if defined."""
        return None

    def sdpa_bias(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Returns an additive bias suitable for scaled_dot_product_attention."""
        return None

    def fallback_mask(
        self,
        seq_len: int,
        device: torch.device,
        base_causal_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Returns a dense mask for the fallback attention implementation."""
        return base_causal_mask

    def flex_block_mask(
        self, seq_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Returns a flex attention block mask if supported."""
        return None

    def reset_sequence_cache(self) -> None:
        """Hook to clear any cached tensors tied to sequence length."""
        return None

    # ------------------------------------------------------------------
    # nn.Module niceties
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        window = self.current_window()
        if window is None:
            return ""
        if isinstance(window, torch.Tensor):
            value = float(window.detach().cpu().item())
        else:
            value = float(window)
        return f"window={value:.3f}"


class GlobalBlockMask(BlockMaskBase):
    """No additional masking beyond causal masking."""

    name = "global"


class SlidingBlockMask(BlockMaskBase):
    """Fixed sliding-window causal block mask."""

    name = "sliding"

    def __init__(self, config, window_size: Optional[int] = None) -> None:
        super().__init__(config, window_size)
        resolved = self._configured_window
        if resolved is None:
            raise ValueError("Sliding block mask requires window_size to be set.")
        resolved = int(resolved)
        if resolved <= 0:
            raise ValueError("window_size must be positive for sliding block masks")
        self._window_size = resolved
        self._sdpa_cache: Dict[Tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}
        self._fallback_cache: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
        self._flex_cache: Dict[Tuple[torch.device, int], torch.Tensor] = {}

    # Capabilities -----------------------------------------------------
    @property
    def is_active(self) -> bool:
        return True

    @property
    def is_sliding(self) -> bool:
        return True

    @property
    def supports_flex(self) -> bool:
        return True

    # Mask builders ----------------------------------------------------
    def current_window(self, device=None, dtype=None) -> torch.Tensor:
        tensor = torch.tensor(
            float(self._window_size),
            device=device if device is not None else None,
            dtype=dtype if dtype is not None else torch.float32,
        )
        return tensor

    def discrete_window(self) -> int:
        return self._window_size

    def sdpa_bias(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device, dtype, q_len, kv_len)
        if key not in self._sdpa_cache:
            q_positions = torch.arange(q_len, device=device)
            kv_positions = torch.arange(kv_len, device=device)
            mask = kv_positions.unsqueeze(0) < (
                q_positions.unsqueeze(1) - self._window_size
            )
            bias = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=dtype)
            finfo = torch.finfo(dtype)
            bias = bias.masked_fill(mask, finfo.min)
            self._sdpa_cache[key] = bias
        return self._sdpa_cache[key]

    def fallback_mask(
        self,
        seq_len: int,
        device: torch.device,
        base_causal_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device, base_causal_mask.dtype, seq_len)
        if key not in self._fallback_cache:
            ones = torch.ones((1, 1, seq_len, seq_len), device=device, dtype=base_causal_mask.dtype)
            diag_mask = torch.triu(ones, diagonal=-self._window_size)
            mask = base_causal_mask * diag_mask
            self._fallback_cache[key] = mask
        return self._fallback_cache[key]

    def flex_block_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        key = (device, seq_len)
        if key not in self._flex_cache:
            window = self._window_size

            def _sliding_window_causal(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                within = (q_idx - kv_idx) <= window
                return causal & within

            block_mask = create_block_mask(
                _sliding_window_causal,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
            self._flex_cache[key] = block_mask
        return self._flex_cache[key]

    def reset_sequence_cache(self) -> None:
        self._flex_cache.clear()


class LearnedSlidingBlockMask(BlockMaskBase):
    """Sliding window with a learnable window size."""

    name = "learned_sliding"

    def __init__(self, config, mode: str = "softplus") -> None:
        super().__init__(config)
        base_window = self._configured_window
        if base_window is None:
            raise ValueError(
                "Learned sliding block masks require window_size to provide the initial value."
            )

        self.mode = mode
        self.temperature = float(getattr(config, "block_mask_learned_temperature", 1.0))
        self.penalty = float(getattr(config, "block_mask_learned_penalty", 30.0))
        self.min_window = float(getattr(config, "block_mask_learned_min_window", 1.0))
        max_window_cfg = getattr(config, "block_mask_learned_max_window", None)
        if max_window_cfg is None:
            max_window_cfg = float(config.block_size)
        self.max_window = float(max(max_window_cfg, self.min_window + 1.0))

        base_window = max(float(base_window), self.min_window)
        init_param = self._inverse_transform(base_window)
        self.window_param = nn.Parameter(init_param)

    # Capabilities -----------------------------------------------------
    @property
    def is_active(self) -> bool:
        return True

    @property
    def is_sliding(self) -> bool:
        return True

    @property
    def supports_flex(self) -> bool:
        return True

    # Helpers ----------------------------------------------------------
    def _inverse_transform(self, target_window: float) -> torch.Tensor:
        offset = target_window - self.min_window
        if self.mode == "softplus":
            offset = max(offset, 1e-4)
            return torch.log(torch.expm1(torch.tensor(offset, dtype=torch.float32)))
        if self.mode == "exp":
            offset = max(offset, 1e-4)
            return torch.log(torch.tensor(offset, dtype=torch.float32))
        if self.mode == "sigmoid":
            span = self.max_window - self.min_window
            if span <= 0:
                raise ValueError(
                    "block_mask_learned_max_window must be larger than min_window for sigmoid strategy"
                )
            ratio = offset / span
            ratio = float(min(max(ratio, 1e-6), 1 - 1e-6))
            return torch.logit(torch.tensor(ratio, dtype=torch.float32))
        raise ValueError(f"Unknown learned sliding mode '{self.mode}'")

    def current_window(self, device=None, dtype=None) -> torch.Tensor:
        param = self.window_param
        if device is not None or dtype is not None:
            param = param.to(device=device or param.device, dtype=dtype or param.dtype)
        if self.mode == "softplus":
            offset = F.softplus(param)
        elif self.mode == "exp":
            offset = torch.exp(param)
        elif self.mode == "sigmoid":
            span = self.max_window - self.min_window
            offset = span * torch.sigmoid(param)
        else:
            raise ValueError(f"Unknown learned sliding mode '{self.mode}'")
        window = self.min_window + offset
        return torch.clamp(window, max=self.max_window)

    def discrete_window(self) -> int:
        window = self.current_window().detach()
        window = torch.clamp(window, min=self.min_window, max=self.max_window)
        return int(torch.round(window).item())

    # Mask builders ----------------------------------------------------
    def sdpa_bias(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        window = self.current_window(device=device, dtype=torch.float32)
        q_positions = torch.arange(q_len, device=device, dtype=window.dtype)
        kv_positions = torch.arange(kv_len, device=device, dtype=window.dtype)
        distance = q_positions.unsqueeze(1) - kv_positions.unsqueeze(0)
        temperature = max(self.temperature, 1e-6)
        gating = torch.sigmoid((window - distance) / temperature)
        bias = torch.log(gating.clamp(min=1e-6)) * self.penalty
        return bias.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    def fallback_mask(
        self,
        seq_len: int,
        device: torch.device,
        base_causal_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        window = self.discrete_window()
        ones = torch.ones((1, 1, seq_len, seq_len), device=device, dtype=base_causal_mask.dtype)
        diag_mask = torch.triu(ones, diagonal=-window)
        return base_causal_mask * diag_mask

    def flex_block_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        window = self.discrete_window()

        def _sliding_window_causal(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            within = (q_idx - kv_idx) <= window
            return causal & within

        return create_block_mask(
            _sliding_window_causal,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )


BLOCK_MASK_REGISTRY: Dict[str, Callable[[Any], BlockMaskBase]] = {
    "global": lambda config: GlobalBlockMask(config),
    "sliding": lambda config: SlidingBlockMask(config),
    "learned_sliding": lambda config: LearnedSlidingBlockMask(config, mode="softplus"),
    "learned_sliding_softplus": lambda config: LearnedSlidingBlockMask(config, mode="softplus"),
    "learned_sliding_sigmoid": lambda config: LearnedSlidingBlockMask(config, mode="sigmoid"),
    "learned_sliding_exp": lambda config: LearnedSlidingBlockMask(config, mode="exp"),
}

# Alias for disabling masking entirely.
BLOCK_MASK_REGISTRY["none"] = BLOCK_MASK_REGISTRY["global"]
BLOCK_MASK_REGISTRY["null"] = BLOCK_MASK_REGISTRY["global"]


def normalize_block_mask_name(name: Optional[str]) -> str:
    if name is None:
        return DEFAULT_BLOCK_MASK
    normalized = str(name).strip().lower()
    if normalized in _SENTINEL_NONE:
        return DEFAULT_BLOCK_MASK
    return normalized


def build_block_mask_strategy(config, name: Optional[str]) -> BlockMaskBase:
    normalized = normalize_block_mask_name(name)
    if normalized not in BLOCK_MASK_REGISTRY:
        available = ", ".join(sorted(BLOCK_MASK_REGISTRY.keys()))
        raise ValueError(f"Unsupported block_mask '{name}'. Available: {available}")
    strategy = BLOCK_MASK_REGISTRY[normalized](config)
    strategy.selected_name = normalized
    return strategy


blockmask_dictionary = BLOCK_MASK_REGISTRY


"""Self-contained Hugging Face model for ReaLLM-Forge csv_mc_int multicontext checkpoints.

    CsvMcIntForCausalLM.load_dir("out/ecg_acc_6L_384D_b256/hf")      # or from_pretrained(..., trust_remote_code=True)
    tokens = model.generate_forecast(context, horizon=130, num_samples=8, top_k=100)

Same computation as ReaLLM-Forge `model.py` (multicontext, <= 64 lanes) with
`attention_variant=causal`: the K lane embeddings are summed (stack + sum, as model.py), then
pre-RMSNorm blocks (RoPE multi-head attention, MLP), a final RMSNorm, and one tied head per lane.
The state_dict names are ReaLLM-Forge's, so checkpoints load strictly.

Additions for live use: a KV cache (`past_key_values`) and `generate_forecast`, which samples
`num_samples` futures of `horizon` rows for each context. When context + horizon fit in
`block_size` the cache gives exactly the windowed model's predictions; longer horizons slide a
`block_size` window like ReaLLM-Forge's sample.py (slower: one full forward per row).

Inputs are `(B, K, T)` token ids (batch, lane, row) or a list of K `(B, T)` tensors;
token_id = value - int_min for each lane (see processing_csv_mc_int.py).
Depends only on torch and transformers.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

try:
    from .configuration_csv_mc_int import CsvMcIntConfig
except ImportError:  # loaded as a plain module (copied model dir, scripts)
    from configuration_csv_mc_int import CsvMcIntConfig

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class CsvMcIntOutput(ModelOutput):
    """loss: weighted mean of lane_losses (plain mean by default, as train.py); lane_losses: (K,);
    logits: K tensors (B, T or 1, V_i); past_key_values: per-layer (k, v) when use_cache;
    hidden_states: final normed (B, T, n_embd) when asked for."""

    loss: Optional[torch.FloatTensor] = None
    lane_losses: Optional[torch.FloatTensor] = None
    logits: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[KVCache] = None
    hidden_states: Optional[torch.FloatTensor] = None


class RMSNorm(nn.Module):
    """ReaLLM-Forge RMSNorm (gain only)."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain


def rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n: int) -> torch.Tensor:
    """Rotates the interleaved pairs (0,1), (2,3), ... of the first n dims of x by the given angles."""
    x_even, x_odd = x[..., :n:2], x[..., 1:n:2]
    out = torch.empty_like(x)
    out[..., :n:2], out[..., 1:n:2] = x_even * cos - x_odd * sin, x_even * sin + x_odd * cos
    if n < x.shape[-1]:
        out[..., n:] = x[..., n:]
    return out


class RotaryEmbedding(nn.Module):
    """ReaLLM-Forge RoPE (interleaved pairs, base 10000), at positions offset .. offset + T - 1."""

    def __init__(self, dim: int, rope_length: Optional[int] = None, fp32: bool = False):
        super().__init__()
        if dim % 2:
            raise ValueError("rotary dim must be even")
        self.dim = dim
        self.rope_length = rope_length if rope_length else dim
        self.fp32 = fp32          # config.rope_fp32: angles and rotation in float32 even under autocast
        self.inv_freq: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if self.inv_freq is None or self.inv_freq.device != x.device:
            half = self.rope_length // 2
            self.inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
        if self.fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                pos = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
                angles = pos[:, None] * self.inv_freq[None, :]
                return rotate_pairs(x.float(), angles.cos(), angles.sin(), self.rope_length).to(x.dtype)
        # As trained before rope_fp32: under bf16 autocast this einsum runs in bf16 (rounded angles)
        pos = torch.arange(offset, offset + x.shape[-2], device=x.device).type_as(self.inv_freq)
        angles = torch.einsum("i,d->id", pos, self.inv_freq)
        sin, cos = angles.sin(), angles.cos()
        n = self.rope_length
        x_even, x_odd = x[..., :n:2], x[..., 1:n:2]
        out = torch.empty_like(x)
        out[..., :n:2], out[..., 1:n:2] = x_even * cos - x_odd * sin, x_even * sin + x_odd * cos
        if n < self.dim:
            out[..., n:] = x[..., n:]
        return out


class CausalSelfAttention(nn.Module):
    """ReaLLM-Forge `CausalSelfAttention` (attention_variant=causal) with a KV cache."""

    def __init__(self, config: CsvMcIntConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_group = config.n_kv_group
        self.head_dim = config.n_embd // config.n_head
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_k = nn.Linear(config.n_embd, self.head_dim * self.n_kv_group, bias=config.bias)
        self.c_attn_v = nn.Linear(config.n_embd, self.head_dim * self.n_kv_group, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj._reallm_residual_proj = True
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.use_qk_norm = config.use_qk_norm
        self.use_qk_norm_scale = config.use_qk_norm_scale
        self.softmax_variant_attn = config.softmax_variant_attn
        self.relu2max_divisor = config.relu2max_divisor
        self.div_by_seq_len = config.div_by_seq_len
        self.rotary_q = self.rotary_k = None
        if config.use_rotary_embeddings:
            fp32 = bool(getattr(config, "rope_fp32", False))
            self.rotary_q = RotaryEmbedding(self.head_dim, config.rope_length, fp32)
            self.rotary_k = RotaryEmbedding(self.head_dim, config.rope_length, fp32)
        if self.use_qk_norm_scale:
            L = config.block_size
            self.qk_norm_factor = nn.Parameter(torch.tensor(math.log2(L * L - L)))
        group = self.n_head // self.n_kv_group
        self.register_buffer("head_to_kv", torch.arange(self.n_kv_group).repeat_interleave(group), persistent=False)

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.n_kv_group == self.n_head else t.index_select(1, self.head_to_kv)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(prefix + "bias", None)    # ReaLLM-Forge's causal-mask buffer (non-flash configs)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        B, T, C = x.size()
        q = self.c_attn_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_attn_k(x).view(B, T, self.n_kv_group, self.head_dim).transpose(1, 2)
        v = self.c_attn_v(x).view(B, T, self.n_kv_group, self.head_dim).transpose(1, 2)
        offset = 0 if past is None else past[0].shape[2]
        if self.rotary_q is not None:
            q = self.rotary_q(q, offset)
            k = self.rotary_k(k, offset)
        if self.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        k_attn, v_attn = self._expand(k), self._expand(v)
        Tk = k_attn.shape[2]
        if self.softmax_variant_attn == "softmax" and self.n_kv_group == self.n_head:
            if self.use_qk_norm_scale:   # ReaLLM-Forge's flash path pre-scales q this way
                q = q * (self.qk_norm_factor * math.sqrt(math.sqrt(k_attn.size(-1))))
            drop = self.dropout if self.training else 0.0
            if past is None:
                y = F.scaled_dot_product_attention(q, k_attn, v_attn, dropout_p=drop, is_causal=True)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k_attn, v_attn, dropout_p=drop)
            else:
                mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
                y = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask, dropout_p=drop)
        else:   # ReaLLM-Forge's manual path (GQA or relu2max)
            att = q @ k_attn.transpose(-2, -1)
            att = att * self.qk_norm_factor if self.use_qk_norm_scale else att / math.sqrt(k_attn.size(-1))
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            att = att.masked_fill(~mask, float("-inf"))
            if self.softmax_variant_attn == "softmax":
                att = F.softmax(att, dim=-1)
            else:
                att = torch.relu(att) ** 2 / self.relu2max_divisor
                if self.div_by_seq_len:
                    att = att / att.shape[-1]
            att = self.attn_dropout(att).to(v_attn.dtype)
            y = att @ v_attn
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.relu(x), 2)


class ActivationWrapper(nn.Module):
    """Same module path as ReaLLM-Forge's activations (mlp.activation_variant.activation)."""

    def __init__(self, cls):
        super().__init__()
        self.activation = cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class MLP(nn.Module):
    """ReaLLM-Forge `OriginalMLP`: c_fc -> activation(x - x_offset) - y_offset -> c_proj."""

    def __init__(self, config: CsvMcIntConfig):
        super().__init__()
        act = config.activation_variant
        self.activation_variant = (SquaredReLU() if act in ("squared_relu", "relu2") else
                                   ActivationWrapper({"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[act]))
        for name, value, learn in (("activation_x_offset", config.mlp_x_offset, config.learn_mlp_x_offset),
                                   ("activation_y_offset", config.mlp_y_offset, config.learn_mlp_y_offset)):
            if learn:
                setattr(self, name, nn.Parameter(torch.tensor(float(value))))
            else:
                self.register_buffer(name, torch.tensor(float(value)))
        size = config.mlp_size if config.mlp_size is not None else config.mlp_expansion_factor * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, size, bias=config.bias)
        self.c_proj = nn.Linear(size, config.n_embd, bias=config.bias)
        self.c_proj._reallm_residual_proj = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    """Pre-RMSNorm block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config: CsvMcIntConfig):
        super().__init__()
        self.pre_ln_attn = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.pre_ln_mlp = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past=None, use_cache: bool = False):
        a, present = self.attn(self.pre_ln_attn(x), past, use_cache)
        x = x + a
        x = x + self.mlp(self.pre_ln_mlp(x))
        return x, present


def tied_weights_keys(num_lanes: int):
    """Each lane's head shares its embedding table. transformers 4.x wants the tied names as a list
    (it then saves safetensors without the duplicates), 5.x a {tied: source} mapping (a list there makes
    the model fail to build in post_init)."""
    import transformers
    heads = [f"transformer.lm_head_{i}.weight" for i in range(num_lanes)]
    if int(transformers.__version__.split(".")[0]) >= 5:
        return {h: f"transformer.wte_{i}.weight" for i, h in enumerate(heads)}
    return heads


class CsvMcIntForCausalLM(PreTrainedModel):
    """ReaLLM-Forge csv_mc_int multicontext GPT as a Hugging Face model (see the module docstring)."""

    config_class = CsvMcIntConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block"]
    # the heads are tied to the embeddings, so a weights file without them is complete
    _keys_to_ignore_on_load_missing = [r"transformer\.lm_head_\d+\.weight"]
    CODE_FILES = ("configuration_csv_mc_int.py", "modeling_csv_mc_int.py", "processing_csv_mc_int.py")

    def __init__(self, config: CsvMcIntConfig):
        super().__init__(config)
        self.num_lanes = len(config.vocab_sizes)
        self._tied_weights_keys = tied_weights_keys(self.num_lanes)   # tied lane by lane in tie_weights()
        mods = {}
        for i, v in enumerate(config.vocab_sizes):
            mods[f"wte_{i}"] = nn.Embedding(v, config.n_embd)
        mods["drop"] = nn.Dropout(config.dropout)
        mods["h"] = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        mods["ln_f"] = RMSNorm(config.n_embd)
        for i, v in enumerate(config.vocab_sizes):
            mods[f"lm_head_{i}"] = nn.Linear(config.n_embd, v, bias=False)
        self.transformer = nn.ModuleDict(mods)
        self.post_init()
        self.tie_weights()

    # ------------------------------------------------------------------ HF plumbing
    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", True) and hasattr(self, "transformer"):
            for i in range(self.num_lanes):
                self.transformer[f"lm_head_{i}"].weight = self.transformer[f"wte_{i}"].weight

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer["wte_0"]

    def set_input_embeddings(self, value: nn.Module):
        self.transformer["wte_0"] = value

    def _init_weights(self, module: nn.Module):
        """ReaLLM-Forge GPT init: N(0, 0.02); residual c_proj N(0, 0.02 / sqrt(2 n_layer)); gains 1."""
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            if getattr(module, "_reallm_residual_proj", False):
                std = std / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gain)
        elif isinstance(module, CausalSelfAttention) and module.use_qk_norm_scale:
            L = self.config.block_size
            with torch.no_grad():
                module.qk_norm_factor.fill_(math.log2(L * L - L))
        elif isinstance(module, MLP):
            with torch.no_grad():
                module.activation_x_offset.fill_(float(self.config.mlp_x_offset))
                module.activation_y_offset.fill_(float(self.config.mlp_y_offset))

    @torch.no_grad()
    def reset_parameters_like_reallm(self) -> "CsvMcIntForCausalLM":
        """Re-initialise every weight exactly like ReaLLM-Forge's GPT (training from scratch)."""
        self.apply(self._init_weights)
        self.tie_weights()
        return self

    # ------------------------------------------------------------------ forward
    def _lanes_bkt(self, ids: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        """Token ids as (B, K, T). Accepts (B, K, T), (K, T) for one context, or K tensors (B, T)."""
        if isinstance(ids, (list, tuple)):
            ids = torch.stack(list(ids), dim=1)
        if ids.dim() == 2:
            ids = ids.unsqueeze(0)
        if ids.dim() != 3 or ids.shape[1] != self.num_lanes:
            raise ValueError(f"expected token ids (B, K={self.num_lanes}, T), got {tuple(ids.shape)}")
        return ids

    def embed(self, ids_bkt: torch.Tensor) -> torch.Tensor:
        """Sum of the lane embeddings, in model.py's order (one lane: as is; else stack + sum)."""
        reps = [self.transformer[f"wte_{i}"](ids_bkt[:, i]) for i in range(self.num_lanes)]
        return reps[0] if len(reps) == 1 else torch.stack(reps, dim=0).sum(dim=0)

    def forward(
        self,
        input_ids: Union[torch.Tensor, Sequence[torch.Tensor]],
        labels: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
        last_step_only: Optional[bool] = None,
        return_logits: bool = True,
        output_hidden_states: bool = False,
        lane_loss_weights: Optional[Sequence[float]] = None,
        return_dict: bool = True,
        **_unused: Any,
    ) -> Union[CsvMcIntOutput, Tuple]:
        """input_ids / labels: (B, K, T) token ids (labels -1 = ignored). past_key_values: the cache
        from a previous call with use_cache=True (positions continue after it). last_step_only:
        logits for the last row only (default: when there are no labels)."""
        ids = self._lanes_bkt(input_ids)
        B, K, T = ids.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_len + T > self.config.block_size:
            raise ValueError(f"{past_len} cached + {T} new rows > block_size {self.config.block_size}")
        if last_step_only is None:
            last_step_only = labels is None

        x = self.transformer["drop"](self.embed(ids))
        presents = [] if use_cache else None
        for li, block in enumerate(self.transformer["h"]):
            x, present = block(x, None if past_key_values is None else past_key_values[li], use_cache)
            if use_cache:
                presents.append(present)
        x = self.transformer["ln_f"](x)

        logits, lane_losses, loss = None, None, None
        if labels is not None:
            y = self._lanes_bkt(labels)
            full = [F.linear(x, self.transformer[f"lm_head_{i}"].weight) for i in range(K)]
            lane_losses = torch.stack([
                F.cross_entropy(full[i].view(-1, full[i].size(-1)), y[:, i].reshape(-1), ignore_index=-1)
                for i in range(K)])
            w = lane_loss_weights if lane_loss_weights is not None else self.config.lane_loss_weights
            if w is None:
                loss = lane_losses.mean()
            else:
                wt = torch.as_tensor(w, dtype=lane_losses.dtype, device=lane_losses.device)
                loss = (lane_losses * wt).sum() / wt.sum()
            if return_logits:
                logits = [f[:, -1:, :] for f in full] if last_step_only else full
        elif return_logits:
            xs = x[:, -1:, :] if last_step_only else x
            logits = [F.linear(xs, self.transformer[f"lm_head_{i}"].weight) for i in range(K)]

        cache = tuple(presents) if use_cache else None
        hidden = x if output_hidden_states else None
        if not return_dict:
            return loss, lane_losses, logits, cache, hidden
        return CsvMcIntOutput(loss=loss, lane_losses=lane_losses, logits=logits, past_key_values=cache,
                              hidden_states=hidden)

    # ------------------------------------------------------------------ forecasting
    @staticmethod
    def sample_next(logits: List[torch.Tensor], temperature: float = 1.0, top_k: Optional[int] = None,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """One token per lane from each lane's last-row logits -> (N, K, 1). temperature <= 0: argmax.
        (top-k then softmax then a multinomial draw, as ReaLLM-Forge's sample.py.)"""
        out = []
        for lg in logits:
            lg = lg[:, -1, :].float()
            if temperature <= 0:
                out.append(lg.argmax(dim=-1, keepdim=True))
                continue
            lg = lg / temperature
            if top_k is not None and top_k < lg.size(-1):
                v, _ = torch.topk(lg, top_k)
                lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
            out.append(torch.multinomial(F.softmax(lg, dim=-1), 1, generator=generator))
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def generate_forecast(self, context: torch.Tensor, horizon: int, num_samples: int = 1,
                          temperature: float = 1.0, top_k: Optional[int] = None,
                          generator: Optional[torch.Generator] = None, use_cache: Optional[bool] = None
                          ) -> torch.Tensor:
        """`num_samples` sampled continuations of each context.

        context: (B, K, T) or (K, T) token ids. Returns (B, num_samples, K, horizon) token ids.
        With context + horizon - 1 <= block_size (and use_cache not False) the KV cache is used and
        every row sees exactly the rows before it, as in training; otherwise a block_size window
        slides along (sample.py's way: the oldest rows drop out of view).
        """
        ctx = self._lanes_bkt(context)
        B, K, T = ctx.shape
        L = self.config.block_size
        if T > L:
            ctx = ctx[:, :, -L:]
            T = L
        x = ctx.repeat_interleave(num_samples, dim=0)
        fits = T + horizon - 1 <= L and not (self.config.div_by_seq_len and
                                             self.config.softmax_variant_attn == "relu2max")
        steps = []
        if fits and use_cache is not False:
            out = self(x, use_cache=True, last_step_only=True)
            nxt = self.sample_next(out.logits, temperature, top_k, generator)
            steps.append(nxt)
            past = out.past_key_values
            for _ in range(horizon - 1):
                out = self(nxt, past_key_values=past, use_cache=True, last_step_only=True)
                past = out.past_key_values
                nxt = self.sample_next(out.logits, temperature, top_k, generator)
                steps.append(nxt)
        else:
            seq = x
            for _ in range(horizon):
                out = self(seq[:, :, -L:], last_step_only=True)
                nxt = self.sample_next(out.logits, temperature, top_k, generator)
                steps.append(nxt)
                seq = torch.cat([seq, nxt], dim=-1)
        return torch.cat(steps, dim=-1).view(B, num_samples, K, horizon)

    # ------------------------------------------------------------------ files
    @classmethod
    def from_reallm_checkpoint(cls, ckpt_path: str, lanes: Optional[List[dict]] = None,
                               map_location: Union[str, torch.device] = "cpu", **config_extra
                               ) -> "CsvMcIntForCausalLM":
        """A ReaLLM-Forge .pt checkpoint (best_ckpt.pt / ckpt.pt), loaded strictly."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = CsvMcIntConfig.from_reallm_model_args(ckpt["model_args"], lanes=lanes, **config_extra)
        model = cls(config)
        prefix = "_orig_mod."
        sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=True)
        model.tie_weights()
        return model.to(map_location).eval()

    def save_dir(self, path: str, extra_files: Optional[dict] = None) -> None:
        """save_pretrained (torch format: keeps the tied lanes shared) + this package's code files,
        so the folder loads with trust_remote_code=True or load_dir() anywhere."""
        os.makedirs(path, exist_ok=True)
        self.save_pretrained(path, safe_serialization=False)
        here = os.path.dirname(os.path.abspath(__file__))
        for name in self.CODE_FILES:
            src = os.path.join(here, name)
            if os.path.isfile(src):
                with open(src, "rb") as fi, open(os.path.join(path, name), "wb") as fo:
                    fo.write(fi.read())
        for name, payload in (extra_files or {}).items():
            with open(os.path.join(path, name), "w") as f:
                f.write(payload if isinstance(payload, str) else json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load_dir(cls, path: str, map_location: Union[str, torch.device] = "cpu",
                 dtype: Optional[torch.dtype] = None) -> "CsvMcIntForCausalLM":
        """Load a saved model folder with plain torch (no from_pretrained machinery), so it works across
        transformers versions. Weights: pytorch_model.bin or model.safetensors."""
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        for k in ("architectures", "auto_map", "transformers_version", "model_type", "torch_dtype", "dtype"):
            cfg.pop(k, None)
        model = cls(CsvMcIntConfig(**cfg))
        bin_path, st_path = os.path.join(path, "pytorch_model.bin"), os.path.join(path, "model.safetensors")
        if os.path.isfile(bin_path):
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        elif os.path.isfile(st_path):
            from safetensors.torch import load_file
            sd = load_file(st_path)
        else:
            raise FileNotFoundError(f"no pytorch_model.bin or model.safetensors in {path}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("transformer.lm_head_")]   # tied to wte_*
        if missing or unexpected:
            raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
        model.tie_weights()
        model = model.to(map_location)
        if dtype is not None:
            model = model.to(dtype)
        return model.eval()

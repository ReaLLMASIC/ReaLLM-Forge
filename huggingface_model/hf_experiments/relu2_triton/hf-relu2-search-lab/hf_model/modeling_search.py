"""Independent-head HF adapter around the unchanged optimized attention kernels.

Q/K have H*d_qk projection width; V and the concatenation have H*d_v width.
Neither width has to equal d_model. Unequal feature widths are zero-padded to
max(d_qk,d_v) for the existing fused kernels, then output is sliced to d_v.
Padding changes work/memory, not attention mathematics; backend labels expose it.
"""
import math
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .configuration_search import SearchConfig
from .modeling_comparison import (ComparisonPreTrainedModel, ComparisonModel,
                                 ComparisonForCausalLM, StableRMSNorm)
from .modeling_nanogpt import _apply_rope, NanoGPTMLP
from .output_norm import CappedHypersphereNorm
from .triton_relu2_attention import can_use_triton_relu2_attention, triton_relu2_attention
from .triton_relu2linear_fixed import triton_relu2linear16_attention
from .relu2linear_activation import relu2linear
# Explicit imports preserve the transitive source graph in HF 4.44 exports.
from .triton_relu2linear_attention import triton_relu2linear_attention
from .triton_relu2max import triton_relu2max
from .configuration_comparison import ComparisonConfig
from .configuration_nanogpt import NanoGPTConfig
from .context_attention import context_attention_reference
from .triton_context_attention import triton_context_attention
from .kda_attention import KimiDeltaAttention


def pad_head_features(q, k, v):
    """Differentiable, identity-preserving adapter; cache actual-width K/V.

Call AFTER QK normalization and RoPE. Always pass the original scale explicitly
to the attention kernel, since its default would depend on the padded width.
"""
    if q.shape[-1] != k.shape[-1] or q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K must agree in feature width and Q/K/V in batch/head dimensions")
    width = max(q.shape[-1], v.shape[-1])
    return tuple(F.pad(x, (0, width - x.shape[-1])) if x.shape[-1] != width else x for x in (q, k, v))


def search_attention_reference(q, k, v, scale, variant, divisor=256.0,
                               attention_mask=None, causal_offset=None):
    """FP32 score/activation oracle with input-dtype attention weights."""
    offset = k.shape[-2] - q.shape[-2] if causal_offset is None else causal_offset
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q.float() @ k.float().transpose(-2, -1)) * scale
        keep = torch.arange(k.shape[-2], device=q.device)[None] <= torch.arange(q.shape[-2], device=q.device)[:, None] + offset
        keep = keep[None, None]
        if attention_mask is not None:
            keep = keep & attention_mask[:, None, None, :k.shape[-2]].bool()
        if variant == "softmax_sdpa":
            # SDPA semantics for completely masked rows: output and gradients zero.
            scores = scores.masked_fill(~keep, -torch.inf)
            scores = torch.where(keep.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, -1).masked_fill(~keep, 0.0)
        elif variant in ("relu2", "linear16"):
            scores = scores.masked_fill(~keep, 0.0)
            weights = (scores.relu().square() if variant == "relu2" else relu2linear(scores, 16.0)) / divisor
        else:
            raise ValueError(variant)
        return weights.to(v.dtype) @ v


class SearchAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.n_qk_head_dim
        h, dq, dv, width = config.num_attention_heads, config.n_qk_head_dim, config.n_v_head_dim, config.hidden_size
        self.q_proj = nn.Linear(width, h * dq, bias=config.bias)
        self.k_proj = nn.Linear(width, h * dq, bias=config.bias)
        self.v_proj = nn.Linear(width, h * dv, bias=config.bias)
        self.out_proj = nn.Linear(h * dv, width, bias=config.bias)
        if config.use_qk_norm_scale:
            self.qk_norm_factor = nn.Parameter(torch.tensor(float(config.qk_norm_scale_init)))
        self.last_backend = None
        self.diagnostic_callback = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False):
        c = self.config
        b, n, _ = hidden_states.shape
        q, k = [p(hidden_states).view(b, n, c.num_attention_heads, c.n_qk_head_dim).transpose(1, 2)
                for p in (self.q_proj, self.k_proj)]
        v = self.v_proj(hidden_states).view(b, n, c.num_attention_heads, c.n_v_head_dim).transpose(1, 2)
        past = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if position_ids is None:
            position_ids = torch.arange(past, past + n, device=q.device)[None]
        if c.use_rotary_embeddings:
            q = _apply_rope(q, position_ids, c.rope_length, c.rope_theta)
            k = _apply_rope(k, position_ids, c.rope_length, c.rope_theta)
        if c.use_qk_norm:
            q = (q.float() / (q.float().norm(dim=-1, keepdim=True) + c.qk_norm_eps)).to(q.dtype)
            k = (k.float() / (k.float().norm(dim=-1, keepdim=True) + c.qk_norm_eps)).to(k.dtype)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None
        if attention_mask is not None and (attention_mask.ndim != 2 or attention_mask.shape[0] != b or attention_mask.shape[1] < k.shape[-2]):
            raise ValueError("attention_mask must be a [batch, >= key_length] keep mask")
        scale = self.qk_norm_factor if c.use_qk_norm_scale else 1 / math.sqrt(c.n_qk_head_dim)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(q.detach(), k.detach(), scale, past)
        divisor = c.relu2max_divisor * (k.shape[-2] if c.relu2max_divide_by_sequence_length else 1)
        padded = c.n_qk_head_dim != c.n_v_head_dim
        qp, kp, vp = pad_head_features(q, k, v)
        if c.variant == "softmax_sdpa":
            mask, causal = None, past == 0 and n > 1
            if attention_mask is not None:
                if q.is_cuda and c.sdpa_backend == "flash":
                    raise ValueError("Forced Flash SDPA requires an unpadded batch in this adapter. "
                                     "Use sdpa_backend='auto' for real key-padding masks; the search "
                                     "runner uses packed unpadded tokens for its Flash baseline.")
                mask = attention_mask[:, None, None, :k.shape[-2]].bool()
                allowed = torch.arange(k.shape[-2], device=q.device)[None] <= torch.arange(n, device=q.device)[:, None] + past
                mask, causal = mask & allowed[None, None], False
            elif past and n > 1:
                from torch.nn.attention.bias import causal_lower_right
                mask, causal = causal_lower_right(n, k.shape[-2]), False
            from torch.nn.attention import sdpa_kernel, SDPBackend
            ctx = nullcontext() if c.sdpa_backend == "auto" else sdpa_kernel(
                SDPBackend.FLASH_ATTENTION if c.sdpa_backend == "flash" else SDPBackend.MATH)
            # Learned g is applied to Q because SDPA's scale argument is a float.
            with ctx:
                out = F.scaled_dot_product_attention((qp * scale).to(qp.dtype), kp, vp,
                    attn_mask=mask, is_causal=causal, dropout_p=0.0, scale=1.0)
            self.last_backend = "sdpa_" + c.sdpa_backend + ("_padded" if padded else "")
        elif c.relu2max_accelerator != "torch" and can_use_triton_relu2_attention(qp, kp, vp, attention_mask):
            fn = triton_relu2_attention if c.variant == "relu2" else triton_relu2linear16_attention
            out = fn(qp, kp, vp, scale=scale, divisor=divisor, causal=True,
                     causal_offset=past, attention_mask=attention_mask)
            self.last_backend = "triton_fused_padded" if padded else "triton_fused"
        else:
            if c.require_fused:
                raise RuntimeError("Fused search attention unavailable: use NVIDIA SM80+, Triton, supported input dtype and head widths <=256")
            out = search_attention_reference(q, k, v, scale, c.variant, divisor, attention_mask, past)
            self.last_backend = "torch_reference"
        out = out[..., :c.n_v_head_dim]
        return self.out_proj(out.transpose(1, 2).reshape(b, n, c.num_attention_heads * c.n_v_head_dim)), present


class SearchBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.attn = SearchAttention(config)
        self.mlp_norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = NanoGPTMLP(config)
        self.attn_output_norm = CappedHypersphereNorm(config.hidden_size, config.output_norm == "capped")
        self.ffn_output_norm = CappedHypersphereNorm(config.hidden_size, config.output_norm == "capped")

    def forward(self, x, **kwargs):
        y, present = self.attn(self.attn_norm(x), **kwargs)
        x = x + self.attn_output_norm(y)
        return x + self.ffn_output_norm(self.mlp(self.mlp_norm(x))), present


class SearchModel(ComparisonModel):
    config_class = SearchConfig
    _no_split_modules = ["SearchBlock"]

    def __init__(self, config):
        ComparisonPreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SearchBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, return_dict=True, **kwargs):
        # Tokenizers/generate commonly supply an all-ones mask. Preserve Flash's
        # compact causal path rather than materializing an unsupported dense
        # mask. This incurs one host sync per masked CUDA model call, not one
        # per layer. Packed-stream training supplies no mask and incurs none.
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != input_ids.shape[0]:
                raise ValueError("attention_mask must be a [batch, >= key_length] keep mask")
            past = 0 if not past_key_values else past_key_values[0][0].shape[-2]
            length = past + input_ids.shape[-1]
            if attention_mask.shape[-1] < length:
                raise ValueError("attention_mask is shorter than the key sequence")
            attention_mask = attention_mask[:, :length]
            if bool(attention_mask.bool().all()):
                attention_mask = None
        return super().forward(input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, return_dict=return_dict, **kwargs)


class SearchForCausalLM(ComparisonForCausalLM):
    config_class = SearchConfig
    _no_split_modules = ["SearchBlock"]

    def __init__(self, config):
        ComparisonPreTrainedModel.__init__(self, config)
        self.model = SearchModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


SearchForCausalLM.register_for_auto_class("AutoModelForCausalLM")

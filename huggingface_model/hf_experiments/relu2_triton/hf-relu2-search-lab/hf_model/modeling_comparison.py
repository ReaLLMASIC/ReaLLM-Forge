"""Separate experiment model. Existing NanoGPT files are intentionally untouched.

Same parameter names for every arm. FP32 RMS reductions + epsilon apply equally
to all arms; custom attention dispatch fails rather than silently timing fallback.
"""
import math
from contextlib import nullcontext
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_comparison import ComparisonConfig
from .modeling_nanogpt import _apply_rope, NanoGPTMLP
# Explicit transitive dependency for Transformers 4.44's local dynamic loader.
from .triton_relu2max import triton_relu2max
from .triton_relu2_attention import can_use_triton_relu2_attention, triton_relu2_attention
from .triton_relu2linear_attention import triton_relu2linear8_attention, triton_relu2linear_attention
from .triton_relu2linear_fixed import (triton_relu2linear2_attention, triton_relu2linear4_attention,
                                      triton_relu2linear16_attention)
from .relu2linear_activation import relu2linear
from .output_norm import CappedHypersphereNorm
from .kda_attention import KimiDeltaAttention
from .context_attention import context_attention_reference
from .triton_context_attention import triton_context_attention


class StableRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class ComparisonAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias))
        if config.use_qk_norm_scale:
            self.qk_norm_factor = nn.Parameter(torch.tensor(float(config.qk_norm_scale_init)))
        if config.learned_subtractive_bias:
            self.subtractive_bias = nn.Parameter(torch.full((config.num_attention_heads,), config.subtractive_bias_init))
        self.last_backend = None
        self.diagnostic_callback = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False):
        c = self.config
        b, n, _ = hidden_states.shape
        q, k, v = [proj(hidden_states).view(b, n, c.num_attention_heads, self.head_dim).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj)]
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
        scale = self.qk_norm_factor if c.use_qk_norm_scale else 1 / math.sqrt(self.head_dim)
        if self.diagnostic_callback is not None:
            self.diagnostic_callback(q.detach(), k.detach(), scale, past)
        divisor = c.relu2max_divisor * (k.shape[-2] if c.relu2max_divide_by_sequence_length else 1)
        if c.variant == "softmax_sdpa":
            mask, causal = None, past == 0 and n > 1
            if attention_mask is not None:
                mask = attention_mask[:, None, None, :k.shape[-2]].bool()
                allowed = torch.arange(k.shape[-2], device=q.device)[None] <= torch.arange(n, device=q.device)[:, None] + past
                mask = mask & allowed[None, None]
                causal = False
            elif past and n > 1:
                from torch.nn.attention.bias import causal_lower_right
                mask, causal = causal_lower_right(n, k.shape[-2]), False
            from torch.nn.attention import sdpa_kernel, SDPBackend
            ctx = (nullcontext() if c.sdpa_backend == "auto" else
                   sdpa_kernel(SDPBackend.FLASH_ATTENTION if c.sdpa_backend == "flash" else SDPBackend.MATH))
            # SDPA scale is a host float. Multiplying Q preserves d(scale).
            with ctx:
                out = F.scaled_dot_product_attention((q * scale).to(q.dtype), k, v,
                      attn_mask=mask, is_causal=causal, dropout_p=0.0, scale=1.0)
            self.last_backend = "sdpa_" + c.sdpa_backend
        elif c.relu2max_accelerator != "torch" and can_use_triton_relu2_attention(q, k, v, attention_mask):
            kwargs = dict(scale=scale, divisor=divisor, causal=True,
                          causal_offset=past, attention_mask=attention_mask)
            if c.is_context_attention:
                out = triton_context_attention(q, k, v, bias=getattr(self, "subtractive_bias", None),
                    alpha=c.causal_length_alpha, anchor=c.causal_length_anchor, **kwargs)
            elif c.variant == "relu2":
                out = triton_relu2_attention(q, k, v, **kwargs)
            elif c.variant == "linear8":
                out = triton_relu2linear8_attention(q, k, v, **kwargs)
            elif c.variant == "linear2":
                out = triton_relu2linear2_attention(q, k, v, **kwargs)
            elif c.variant == "linear4":
                out = triton_relu2linear4_attention(q, k, v, **kwargs)
            elif c.variant == "linear16":
                out = triton_relu2linear16_attention(q, k, v, **kwargs)
            else:
                out = triton_relu2linear_attention(q, k, v, threshold=c.effective_threshold, **kwargs)
            self.last_backend = "triton_fused"
        else:
            if c.require_fused:
                raise RuntimeError("Fused attention unavailable for this input. Use CUDA SM80+ and BF16; CPU is smoke-only.")
            if c.is_context_attention:
                out = context_attention_reference(q, k, v, scale=scale,
                    bias=getattr(self, "subtractive_bias", None), divisor=divisor,
                    alpha=c.causal_length_alpha, anchor=c.causal_length_anchor,
                    causal=True, causal_offset=past, attention_mask=attention_mask)
                self.last_backend = "torch_reference"
                return self.out_proj(out.transpose(1, 2).reshape(b, n, -1)), present
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = (q.float() @ k.float().transpose(-2, -1)) * scale
                keep = torch.arange(k.shape[-2], device=q.device)[None] <= torch.arange(n, device=q.device)[:, None] + past
                keep = keep[None, None]
                if attention_mask is not None:
                    keep = keep & attention_mask[:, None, None, :k.shape[-2]].bool()
                scores = scores.masked_fill(~keep, 0)
                weights = (scores.relu().square() if c.variant == "relu2" else relu2linear(scores, c.effective_threshold)) / divisor
            out = weights.to(v.dtype) @ v
            self.last_backend = "torch_reference"
        return self.out_proj(out.transpose(1, 2).reshape(b, n, -1)), present


class ComparisonBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.attn = KimiDeltaAttention(config) if config.variant == "kda" else ComparisonAttention(config)
        self.mlp_norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = NanoGPTMLP(config)
        self.attn_output_norm = CappedHypersphereNorm(config.hidden_size, config.output_norm == "capped")
        self.ffn_output_norm = CappedHypersphereNorm(config.hidden_size, config.output_norm == "capped")

    def forward(self, x, **kwargs):
        y, present = self.attn(self.attn_norm(x), **kwargs)
        x = x + self.attn_output_norm(y)
        return x + self.ffn_output_norm(self.mlp(self.mlp_norm(x))), present


class ComparisonPreTrainedModel(PreTrainedModel):
    config_class = ComparisonConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ComparisonBlock"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


class ComparisonModel(ComparisonPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ComparisonBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = StableRMSNorm(config.hidden_size, config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, return_dict=True, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        past_key_values = past_key_values or [None] * len(self.layers)
        if len(past_key_values) != len(self.layers):
            raise ValueError("KV cache layer count mismatch")
        past = 0 if self.config.variant == "kda" or past_key_values[0] is None else past_key_values[0][0].shape[-2]
        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
            else:
                position_ids = torch.arange(past, past + input_ids.shape[1], device=input_ids.device)[None]
        x, presents = self.embed_tokens(input_ids), []
        for layer, cache in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                # Bind each layer now; late-bound lambdas recompute the wrong layer.
                call = partial(layer, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                x, _ = checkpoint(call, x, use_reentrant=False)
                use_cache = False
            else:
                x, p = layer(x, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_value=cache, use_cache=use_cache)
                if use_cache:
                    presents.append(p)
        out = BaseModelOutputWithPast(last_hidden_state=self.norm(x), past_key_values=tuple(presents) if use_cache else None)
        return out if return_dict else out.to_tuple()


class ComparisonForCausalLM(ComparisonPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ComparisonModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, value): self.lm_head = value

    def forward(self, input_ids, attention_mask=None, labels=None, logits_to_keep=0, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        out = self.model(input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        hidden = out.last_hidden_state
        if logits_to_keep and labels is None:
            hidden = hidden[:, -logits_to_keep:]
        logits = self.lm_head(hidden)
        loss = None if labels is None else F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1))
        result = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return dict(input_ids=input_ids, past_key_values=past_key_values,
                    attention_mask=attention_mask, use_cache=True, logits_to_keep=1)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer) for layer in past_key_values)


ComparisonConfig.register_for_auto_class()
ComparisonForCausalLM.register_for_auto_class("AutoModelForCausalLM")

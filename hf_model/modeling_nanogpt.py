"""Hugging Face causal LM matching nanoGPT's QK-norm attention path."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_nanogpt import NanoGPTConfig
from .triton_relu2max import can_use_triton_relu2max, triton_relu2max


class NanoGPTRMSNorm(nn.Module):
    """RMSNorm matching ``variations.norm_variations.RMSNorm`` exactly."""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return x / rms * self.weight


def _repeat_kv(x, num_heads, num_kv_heads):
    if num_heads == num_kv_heads:
        return x
    # Match nanoGPT's contiguous, approximately-even head-to-group mapping.
    sizes = [num_heads // num_kv_heads + (i < num_heads % num_kv_heads) for i in range(num_kv_heads)]
    return torch.cat([x[:, i : i + 1].expand(-1, size, -1, -1) for i, size in enumerate(sizes)], dim=1)


def _apply_rope(x, position_ids, rotary_dim, theta):
    """Apply the repo's interleaved (even/odd), optionally partial RoPE."""
    if not rotary_dim:
        return x
    inv_freq = 1.0 / theta ** (
        torch.arange(rotary_dim // 2, device=x.device, dtype=torch.float32) / (rotary_dim // 2)
    )
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    rotated, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    even, odd = rotated[..., 0::2], rotated[..., 1::2]
    out = torch.empty_like(rotated)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return torch.cat((out, tail), dim=-1)


class NanoGPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        kv_size = self.head_dim * config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        if config.use_qk_norm_scale:
            self.qk_norm_factor = nn.Parameter(torch.tensor(float(config.qk_norm_scale_init)))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        batch, length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, length, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + length, device=hidden_states.device)[None, :]
        if self.config.use_rotary_embeddings:
            q = _apply_rope(q, position_ids, self.config.rope_length, self.config.rope_theta)
            k = _apply_rope(k, position_ids, self.config.rope_length, self.config.rope_theta)
        if self.config.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + self.config.qk_norm_eps)
            k = k / (k.norm(dim=-1, keepdim=True) + self.config.qk_norm_eps)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None
        k, v = (_repeat_kv(t, self.config.num_attention_heads, self.config.num_key_value_heads) for t in (k, v))

        scores = q @ k.transpose(-2, -1)
        scores = scores * self.qk_norm_factor if self.config.use_qk_norm_scale else scores / math.sqrt(self.head_dim)
        key_length = k.shape[-2]
        causal = torch.arange(key_length, device=scores.device)[None, :] <= (
            torch.arange(length, device=scores.device)[:, None] + past_length
        )
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :key_length] == 0, torch.finfo(scores.dtype).min)

        if self.config.attention_normalizer == "relu2max":
            divisor = self.config.relu2max_divisor
            if self.config.relu2max_divide_by_sequence_length:
                divisor *= key_length
            use_triton = self.config.relu2max_accelerator == "triton" or (
                self.config.relu2max_accelerator == "auto" and can_use_triton_relu2max(scores)
            )
            if use_triton:
                weights = triton_relu2max(scores, divisor)
            else:
                weights = F.relu(scores).square() / divisor
        else:
            weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, self.config.attention_dropout, self.training)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, -1)
        return self.resid_dropout(self.out_proj(output)), present


class NanoGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(self.activation(self.up_proj(x))))


class NanoGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = NanoGPTRMSNorm(config.hidden_size)
        self.attn = NanoGPTAttention(config)
        self.mlp_norm = NanoGPTRMSNorm(config.hidden_size)
        self.mlp = NanoGPTMLP(config)

    def forward(self, x, **kwargs):
        attn, present = self.attn(self.attn_norm(x), **kwargs)
        x = x + attn
        return x + self.mlp(self.mlp_norm(x)), present


class NanoGPTPreTrainedModel(PreTrainedModel):
    config_class = NanoGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanoGPTBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)


class NanoGPTModel(NanoGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size) if config.use_absolute_position_embeddings else None
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([NanoGPTBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = NanoGPTRMSNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, return_dict=None, **kwargs):
        del kwargs
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        past_key_values = past_key_values or [None] * len(self.layers)
        past_length = 0 if past_key_values[0] is None else past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)[None, :]
        x = self.embed_tokens(input_ids)
        if self.embed_positions is not None:
            x = x + self.embed_positions(position_ids)
        x = self.dropout(x)
        presents = []
        for layer, past in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False
                x, _ = self._gradient_checkpointing_func(
                    lambda states: layer(states, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)[0], x
                ), None
            else:
                x, present = layer(x, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past, use_cache=use_cache)
                if use_cache:
                    presents.append(present)
        output = BaseModelOutputWithPast(last_hidden_state=self.norm(x), past_key_values=tuple(presents) if use_cache else None)
        return output if return_dict else output.to_tuple()


class NanoGPTForCausalLM(NanoGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NanoGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask, "use_cache": kwargs.get("use_cache", True)}

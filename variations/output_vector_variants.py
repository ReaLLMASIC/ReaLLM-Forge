# variations/output_vector_variants.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from variations.position_encoding_variations import RotaryEmbedding


class LinearMixer(nn.Module):
    """Learned linear combination of all layer outputs."""

    def __init__(self, config):
        super().__init__()
        mix_init = torch.zeros(config.n_layer + 1)
        mix_init[-1] = 1.0
        self.weights = nn.Parameter(mix_init)

    def forward(self, layer_outputs):
        stack = torch.stack(layer_outputs, dim=0)
        weights = self.weights.view(-1, 1, 1, 1)
        return (weights * stack).sum(dim=0)


class RouterTop1(nn.Module):
    """Select the single best layer output using a router."""

    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.n_embd, config.n_layer + 1)

    def forward(self, layer_outputs):
        stack = torch.stack(layer_outputs, dim=0)
        stack_perm = stack.permute(1, 2, 0, 3)
        logits = self.router(layer_outputs[-1])
        indices = logits.argmax(dim=-1)
        gather_idx = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, stack_perm.size(-1))
        selected = stack_perm.gather(2, gather_idx).squeeze(2)
        return selected


class RouterTopK(nn.Module):
    """Softly combine the top-k layer outputs using router weights."""

    def __init__(self, config):
        super().__init__()
        self.k = config.ln_f_mixer_top_k
        self.router = nn.Linear(config.n_embd, config.n_layer + 1)

    def forward(self, layer_outputs):
        stack = torch.stack(layer_outputs, dim=0)
        stack_perm = stack.permute(1, 2, 0, 3)
        logits = self.router(layer_outputs[-1])
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1).unsqueeze(-1)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, self.k, stack_perm.size(-1))
        selected = stack_perm.gather(2, gather_idx)
        mixed = (weights * selected).sum(dim=2)
        return mixed


class DecoderMixer(nn.Module):
    """Apply a full-attention decoder layer across block outputs."""

    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layer + 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rotary = RotaryEmbedding(config, size=config.n_embd // config.n_head)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, layer_outputs):
        B, T, C = layer_outputs[0].shape
        seq = torch.stack(layer_outputs, dim=1)
        seq = seq.permute(0, 2, 1, 3).reshape(B * T, self.n_layers, C)
        q = self.q_proj(seq)
        k = self.k_proj(seq)
        v = self.v_proj(seq)
        q = q.view(B * T, self.n_layers, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B * T, self.n_layers, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B * T, self.n_layers, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.rotary(q)
        k = self.rotary(k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.softmax(dim=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B * T, self.n_layers, C)
        out = self.proj(out)
        out = self.resid_dropout(out)
        last = out[:, -1, :].view(B, T, C)
        return last


output_vector_variant_dict = {
    'linear': LinearMixer,
    'router_top1': RouterTop1,
    'router_topk': RouterTopK,
    'decoder': DecoderMixer,
}

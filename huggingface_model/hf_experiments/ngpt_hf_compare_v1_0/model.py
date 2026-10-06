"""Dense GPT and original-paper nGPT. Pure PyTorch core; HF adapter: hf_model.py.

Reference: Loshchilov et al., arXiv:2410.01131v2, Sections 2.1--2.6.
Concepts and naming also follow NVIDIA/ngpt (MIT); see THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    width: int = 512
    layers: int = 8
    heads: int = 8
    context: int = 1024
    variant: str = "ngpt"
    rope_theta: float = 10000.0
    alpha_init: float = 0.05
    norm_eps: float = 1e-12
    rms_eps: float = 1e-6
    activation_checkpointing: bool = False
    weight_storage: str = "fp32"

    def __post_init__(self):
        if min(self.vocab_size, self.width, self.layers, self.heads, self.context) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must be divisible by heads; head dimension must be even.")
        if self.variant not in {"gpt", "ngpt"}:
            raise ValueError("variant must be gpt or ngpt")
        if self.weight_storage not in {"fp32", "reference_bf16"}:
            raise ValueError("Unknown weight-storage policy.")


def unit(x, dim=-1, eps=1e-12):
    # FP32 normalization and residual arithmetic even inside BF16 autocast.
    return F.normalize(x.float(), p=2, dim=dim, eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        d = c.width
        # Identically shaped backbone matrices in the two variants; no biases.
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.up_gate = nn.Linear(d, 8*d, bias=False)  # two 4d SwiGLU branches
        self.down_proj = nn.Linear(4*d, d, bias=False)
        if c.variant == "ngpt":
            # Stored scale and effective initialization are NOT the same thing.
            base = d ** -0.5
            self.alpha_attn = nn.Parameter(torch.full((d,), base))
            self.alpha_mlp = nn.Parameter(torch.full((d,), base))
            self.s_qk = nn.Parameter(torch.full((c.heads, d//c.heads), base))
            self.s_u = nn.Parameter(torch.ones(4*d))
            self.s_v = nn.Parameter(torch.ones(4*d))
        else:
            self.attn_norm = RMSNorm(d, c.rms_eps)
            self.mlp_norm = RMSNorm(d, c.rms_eps)

    def mix(self, h, suggestion, raw_alpha):
        alpha = (raw_alpha * (self.c.alpha_init * math.sqrt(self.c.width))).abs()
        a = unit(h, eps=self.c.norm_eps)
        b = unit(suggestion, eps=self.c.norm_eps)
        # Normalized linear interpolation, not a replacement with exact SLERP.
        return unit(a + alpha * (b - a), eps=self.c.norm_eps)

    def forward(self, h, cos, sin):
        c = self.c
        ng = c.variant == "ngpt"
        b, t, d = h.shape
        x = h if ng else self.attn_norm(h)
        def split(v):
            return v.view(b, t, c.heads, d//c.heads).transpose(1, 2)
        q, k, v = (split(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj))
        def rotate(z):
            # Correct interleaved RoPE, with positions computed in FP32.
            a, b = z.float()[..., 0::2], z.float()[..., 1::2]
            return torch.stack((a*cos-b*sin, a*sin+b*cos), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        if ng:
            s = self.s_qk[None, :, None, :] * math.sqrt(d)
            q = unit(q, eps=c.norm_eps) * s
            k = unit(k, eps=c.norm_eps) * s
        # Explicit scale: SDPA's default 1/sqrt(head_dim) is WRONG for nGPT.
        scale = math.sqrt(d//c.heads) if ng else 1/math.sqrt(d//c.heads)
        y = F.scaled_dot_product_attention(
            q.to(v.dtype), k.to(v.dtype), v,
            dropout_p=0.0, is_causal=True, scale=scale,
        ).transpose(1, 2).reshape(b, t, d)
        suggestion = self.o_proj(y)
        h = self.mix(h, suggestion, self.alpha_attn) if ng else h + suggestion.float()
        x = h if ng else self.mlp_norm(h)
        u, gate = self.up_gate(x).chunk(2, dim=-1)
        if ng:
            # Paper Eqs. 20--21: sqrt(d) on the SiLU gate branch only.
            u = u.float() * self.s_u
            gate = gate.float() * self.s_v * math.sqrt(d)
        suggestion = self.down_proj(u * F.silu(gate))
        return self.mix(h, suggestion, self.alpha_mlp) if ng else h + suggestion.float()


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c = config
        c = config
        self.embed = nn.Embedding(c.vocab_size, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.lm_head = nn.Linear(c.width, c.vocab_size, bias=False)  # untied
        if c.variant == "ngpt":
            self.s_z = nn.Parameter(torch.full((c.vocab_size,), c.width**-0.5))
        else:
            self.final_norm = RMSNorm(c.width, c.rms_eps)
        # Shared RNG order across variants. Scalar initializations consume no RNG.
        std = c.width**-0.5 if c.variant == "ngpt" else 0.02
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=std)
        for block in self.blocks:
            for projection in (block.o_proj, block.down_proj):
                projection.weight.data.div_(math.sqrt(2*c.layers))
        self.register_buffer("inv_freq", 1.0 / (c.rope_theta ** (
            torch.arange(0, c.width//c.heads, 2).float() / (c.width//c.heads)
        )), persistent=False)
        if c.weight_storage == "reference_bf16":
            # Public NVIDIA implementation: BF16 block linear weights; FP32
            # input/output embeddings and learned scale / normalization vectors.
            for block in self.blocks:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        module.to(dtype=torch.bfloat16)
        self.project_weights_()

    def constrained_weights(self):
        """PyTorch weights are [out, in]; residual-output matrices use DIM 0."""
        yield "embed.weight", self.embed.weight, 1
        yield "lm_head.weight", self.lm_head.weight, 1
        for i, b in enumerate(self.blocks):
            for name in ("q_proj", "k_proj", "v_proj", "up_gate"):
                yield f"blocks.{i}.{name}.weight", getattr(b, name).weight, 1
            for name in ("o_proj", "down_proj"):
                yield f"blocks.{i}.{name}.weight", getattr(b, name).weight, 0

    @torch.no_grad()
    def project_weights_(self):
        if self.c.variant == "ngpt":
            for _, w, dim in self.constrained_weights():
                # Keep Parameter identities: these are exactly the optimizer's weights.
                w.copy_(unit(w, dim=dim, eps=self.c.norm_eps).to(w.dtype))

    @torch.no_grad()
    def diagnostics(self):
        if self.c.variant != "ngpt":
            return {}
        error = max((w.float().norm(dim=dim) - 1).abs().max().item()
                    for _, w, dim in self.constrained_weights())
        d = self.c.width
        alpha = torch.cat([b.alpha_attn.abs() for b in self.blocks]).mean().item()
        return {"unit_norm_max_error": error,
                "alpha_attn_mean": alpha*self.c.alpha_init*math.sqrt(d),
                "s_z_mean": (self.s_z*math.sqrt(d)).mean().item()}

    def hidden(self, input_ids):
        if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= self.c.context:
            raise ValueError("input_ids must be [batch, time] within configured context.")
        pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.float32)
        angles = torch.outer(pos, self.inv_freq.float())
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        h = self.embed(input_ids).float()
        for block in self.blocks:
            if self.c.activation_checkpointing and self.training and torch.is_grad_enabled():
                h = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        return h if self.c.variant == "ngpt" else self.final_norm(h)

    def project_logits(self, h):
        logits = self.lm_head(h)
        if self.c.variant == "ngpt":
            logits = logits.float() * (self.s_z * math.sqrt(self.c.width))
        return logits

    def forward(self, input_ids):
        return self.project_logits(self.hidden(input_ids))

    def next_token_loss(self, tokens, chunk_tokens=128):
        """tokens: [B, context+1]. Exactly ONE shift. No full vocabulary logits retained.

        Recompute each chunk in backward; mere slicing without recomputation would
        retain all vocabulary logits and not deliver the intended memory reduction.
        """
        if tokens.ndim != 2 or tokens.shape[1] < 2 or chunk_tokens < 1:
            raise ValueError("Need [batch, >=2] tokens and positive chunk size.")
        h = self.hidden(tokens[:, :-1]).reshape(-1, self.c.width)
        targets = tokens[:, 1:].reshape(-1)
        def chunk_loss(states, labels):
            return F.cross_entropy(self.project_logits(states).float(), labels, reduction="sum")
        total = h.new_zeros(())
        for start in range(0, h.shape[0], chunk_tokens):
            args = (h[start:start+chunk_tokens], targets[start:start+chunk_tokens])
            if torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, *args, use_reentrant=False)
            else:
                total = total + chunk_loss(*args)
        return total / targets.numel()

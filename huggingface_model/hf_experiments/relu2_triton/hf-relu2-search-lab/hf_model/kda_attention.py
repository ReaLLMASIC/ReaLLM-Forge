"""Kimi Delta Attention adapter for the small-model sweep.

CUDA: pinned FLA Triton kernels for causal convolution, chunkwise KDA with
backward, recurrent inference and gated head RMSNorm. CPU: explicit mathematical
reference, reserved for tests. No automatic CUDA-to-reference fallback.

The adapter follows Kimi Linear section 4 / FLA's KimiDeltaAttention layer:
rank=head_dim forget/output gates, beta in (0,1), four-tap SiLU convolutions,
L2-normalized Q/K, per-head RMSNorm then sigmoid output gating, and no RoPE.
It is a pure KDA arm in our dense backbone, not the 3:1 KDA/MLA Kimi model.
"""
# Neural parameterization adapted from FLA KimiDeltaAttention (MIT); see
# third_party/FLA_LICENSE. The installed FLA kernels are unmodified.
import importlib
import importlib.metadata
import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

FLA_VERSION = "0.5.2"


@lru_cache(maxsize=1)
def triton_ops():
    # importlib keeps the optional CUDA dependency out of HF's static scanner,
    # so CPU save/load and the other arms remain usable without FLA installed.
    try:
        version = importlib.metadata.version("fla-core")
        if version != FLA_VERSION:
            raise RuntimeError(f"Expected fla-core=={FLA_VERSION}, found {version}")
        kda = importlib.import_module("fla.ops.kda")
        conv = importlib.import_module("fla.modules.conv.causal_conv1d")
        norm = importlib.import_module("fla.modules.fused_norm_gate")
        return kda.chunk_kda, kda.fused_recurrent_kda, conv.causal_conv1d, norm.rms_norm_gated
    except Exception as exc:
        raise RuntimeError("KDA Triton backend unavailable. Run scripts/install_kda.sh with the same "
                           "RELU2_PYTHON used for training; no reference fallback is used on CUDA.") from exc


def kda_reference(q, k, v, log_decay, beta, initial_state=None, scale=None):
    """Independent differentiable recurrence, tensors [B,T,H,D], state [B,H,K,V].

    Inputs q/k are already normalized; log_decay is log(alpha), beta is sigmoid.
    Decay the old state FIRST, then correct its prediction for the current key.
    Read AFTER writing the current token (causal, including the diagonal).
    FP64 is retained for derivative tests; otherwise state/reductions use FP32.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, log_decay, beta = [x.to(dtype) for x in (q, k, v, log_decay, beta)]
    b, t, h, d = q.shape
    scale = d ** -0.5 if scale is None else scale
    state = q.new_zeros(b, h, d, v.shape[-1]) if initial_state is None else initial_state.to(dtype)
    outputs = []
    for i in range(t):
        decayed = log_decay[:, i].exp().unsqueeze(-1) * state
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, i], decayed)
        error = beta[:, i].unsqueeze(-1) * (v[:, i] - prediction)
        state = decayed + k[:, i].unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i] * scale, state))
    return torch.stack(outputs, dim=1), state


class KDAShortConv(nn.Module):
    """Depthwise filters stored as [channel,tap] matrices for native Muon.

    Equivalent to Conv1d[D,1,W]. The CPU reference and Triton kernel both keep
    W raw projection values per channel in their cache, oldest first.
    """
    _muon_matrix = True

    def __init__(self, width, kernel_size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, kernel_size))
        self.kernel_size = kernel_size
        nn.init.uniform_(self.weight, -1 / math.sqrt(kernel_size), 1 / math.sqrt(kernel_size))

    def forward(self, x, cache=None, use_cache=False):
        if x.is_cuda:
            conv = triton_ops()[2]
            return conv(x=x, weight=self.weight, initial_state=cache,
                        output_final_state=use_cache, activation="silu", backend="triton")
        raw = x.transpose(1, 2)
        w = self.kernel_size
        history = raw.new_zeros(raw.shape[0], raw.shape[1], w) if cache is None else cache
        joined = torch.cat((history.to(raw.dtype), raw), dim=-1)
        # Only W-1 prior samples are needed for convolution; W are saved for FLA.
        y = F.conv1d(joined[..., 1:], self.weight[:, None, :], groups=raw.shape[1])
        return F.silu(y.transpose(1, 2)), joined[..., -w:].contiguous() if use_cache else None


class KimiDeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, h = config.hidden_size, config.num_attention_heads
        self.head_dim, self.heads = d // h, h
        rank = self.head_dim
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(d, d, bias=False))
        for name in ("q_conv", "k_conv", "v_conv"):
            setattr(self, name, KDAShortConv(d, config.kda_conv_size))
        self.f_proj = nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False))
        self.b_proj = nn.Linear(d, h, bias=False)
        self.g_proj = nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=True))
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(d) * math.log(100) + math.log(.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.head_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.last_backend = None

    def _empty_cache(self, x):
        b, _, d = x.shape
        # The training model has FP32 parameters even under BF16 autocast.
        dtype = torch.get_autocast_dtype("cuda") if x.is_cuda and torch.is_autocast_enabled() else x.dtype
        state = torch.zeros(b, self.heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        return (state,) + tuple(torch.zeros(b, d, self.config.kda_conv_size, device=x.device, dtype=dtype) for _ in range(3))

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False):
        x = hidden_states
        b, t, d = x.shape
        if not x.is_cuda and self.config.require_fused:
            raise RuntimeError("KDA requires CUDA for training; CPU reference is smoke-only")
        if x.is_cuda:
            triton_ops()  # fail clearly before any expensive work
        # Padding is removed, not merely zeroed: padding must not decay the state
        # or advance a convolution. Packed sweep batches take the fast mask=None path.
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != b or attention_mask.shape[1] < t:
                raise ValueError("KDA accepts a 2D padding mask covering the current tokens")
            mask = attention_mask[:, -t:].bool()
            if not bool(mask.all()):
                outputs, states = [], []
                for row in range(b):
                    indices = mask[row].nonzero(as_tuple=True)[0]
                    cache = None if past_key_value is None else tuple(s[row:row+1] for s in past_key_value)
                    if len(indices):
                        y, cache = self.forward(x[row:row+1, indices], past_key_value=cache, use_cache=use_cache)
                        padded = y.new_zeros(1, t, d).index_copy(1, indices, y)
                    else:
                        padded = x.new_zeros(1, t, d)
                        cache = cache if cache is not None else self._empty_cache(x[row:row+1])
                    outputs.append(padded)
                    if use_cache: states.append(cache)
                present = tuple(torch.cat([s[i] for s in states]) for i in range(4)) if use_cache else None
                return torch.cat(outputs), present
        previous = (None,) * 4 if past_key_value is None else past_key_value
        if len(previous) != 4:
            raise ValueError("KDA cache must contain recurrent state and three convolution states")
        q, cq = self.q_conv(self.q_proj(x), previous[1], use_cache)
        k, ck = self.k_conv(self.k_proj(x), previous[2], use_cache)
        v, cv = self.v_conv(self.v_proj(x), previous[3], use_cache)
        q, k, v = [a.reshape(b, t, self.heads, self.head_dim) for a in (q, k, v)]
        g = self.f_proj(x).reshape_as(q)
        beta = self.b_proj(x)
        gate = self.g_proj(x).reshape_as(v)
        if x.is_cuda:
            if q.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError("KDA Triton requires BF16/FP16 projections; enable the sweep's BF16 autocast")
            chunk, recurrent, _, norm = triton_ops()
            # Recurrent inference has no backward; eval() alone must never select it.
            use_recurrent = not torch.is_grad_enabled() and not self.training and t <= 64
            op = recurrent if use_recurrent else chunk
            out, state = op(q=q, k=k, v=v, g=g, beta=beta, A_log=self.A_log,
                dt_bias=self.dt_bias, scale=self.head_dim ** -.5, initial_state=previous[0],
                output_final_state=use_cache, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=False)
            out = norm(out, gate, self.head_norm_weight, None, activation="sigmoid", eps=self.config.norm_eps)
            self.last_backend = "fla_triton_kda_recurrent" if use_recurrent else "fla_triton_kda_chunk"
        else:
            work = torch.float64 if q.dtype == torch.float64 else torch.float32
            q, k = [a.to(work) * torch.rsqrt(a.to(work).square().sum(-1, keepdim=True) + 1e-6) for a in (q, k)]
            log_decay = -self.A_log.to(work).exp()[None, None, :, None] * F.softplus(g.to(work) + self.dt_bias.to(work).reshape(self.heads, self.head_dim))
            out, state = kda_reference(q, k, v, log_decay, beta.to(work).sigmoid(), previous[0])
            out = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + self.config.norm_eps)
            out = (out * self.head_norm_weight.to(work) * gate.to(work).sigmoid()).to(v.dtype)
            self.last_backend = "kda_torch_reference"
        present = (state, cq, ck, cv) if use_cache else None
        return self.out_proj(out.reshape(b, t, d)), present

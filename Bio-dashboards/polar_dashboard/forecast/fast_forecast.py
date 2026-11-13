#!/usr/bin/env python3
"""Fast forecasting for csv_mc_int HF models: a static KV cache and a CUDA-graph decode step.

generate_forecast() (modeling_csv_mc_int.py) runs one Python-level forward per generated row; at this
model size that is ~7.5 ms per row on an A100, almost all kernel-launch overhead, so a 1 s forecast
(130 rows) takes ~1 s. FastForecaster removes the overhead for the live dashboard:

  * the KV cache for `num_samples` continuations of one context is allocated once (block_size rows);
  * prefill: one ordinary forward of the context (the model's own code, bf16 autocast as in training)
    fills the cache and gives the first row's logits;
  * decode: one row per step for all samples -- lane-embedding sum, the blocks against the cache, the
    tied lane heads and the sampler -- captured once as a CUDA graph and replayed per row. The position,
    the input tokens and the output row index live in device tensors that the step itself advances.

Numerics follow the bf16-autocast forward the models were trained and evaluated with (fp32 residual
stream and RMSNorm, bf16 matmuls / attention, and bf16 rotary angles -- autocast runs the model's
einsum(pos, inv_freq) in bf16, so training saw rounded angles; a model with config.rope_fp32 computes
its angles and rotation in fp32 instead); with dtype=float32 everything is fp32.
The sampler draws from softmax(top_k(logits) / temperature) like
sample_next(), using the Gumbel-max trick (the same distribution; graph-friendly). temperature <= 0 is
greedy (argmax). Supported: softmax attention without qk-norm (the H10 models); otherwise use
generate_forecast().

    fc = FastForecaster(model, num_samples=8, top_k=100)          # model: CsvMcIntForCausalLM on CUDA
    toks = fc.forecast(context_tokens, horizon=130)              # (K, T) ids -> (num_samples, K, horizon)

    python3 hf_csv_mc_int/fast_forecast.py --model out/hf_models/best/ecg --dataset data/csv_mc_int_ecg --check
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from modeling_csv_mc_int import CsvMcIntForCausalLM  # noqa: E402


class FastForecaster:
    def __init__(self, model: CsvMcIntForCausalLM, num_samples: int = 8, top_k: Optional[int] = 100,
                 temperature: float = 1.0, dtype: torch.dtype = torch.bfloat16, use_cuda_graph: bool = True):
        cfg = model.config
        unsupported = [name for name, bad in (
            ("softmax_variant_attn != softmax", cfg.softmax_variant_attn != "softmax"),
            ("use_qk_norm", bool(cfg.use_qk_norm)),
            ("use_qk_norm_scale", bool(getattr(cfg, "use_qk_norm_scale", False))),
            ("use_rotary_embeddings off", not cfg.use_rotary_embeddings),
        ) if bad]
        if unsupported:
            raise NotImplementedError(f"FastForecaster does not support: {', '.join(unsupported)}")
        self.model = model.eval()
        self.dev = next(model.parameters()).device
        self.dtype = dtype if self.dev.type == "cuda" else torch.float32
        self.N, self.top_k, self.temperature = int(num_samples), top_k, float(temperature)
        self.L, self.K = cfg.block_size, len(cfg.vocab_sizes)
        self.vocab = list(cfg.vocab_sizes)
        t = model.transformer
        self.blocks = list(t["h"])
        att = self.blocks[0].attn
        self.nh, self.nkv, self.hd = att.n_head, att.n_kv_group, att.head_dim
        self.C = cfg.n_embd
        rl = att.rotary_q.rope_length
        self.rope_n = rl
        self.rope_fp32 = bool(getattr(att.rotary_q, "fp32", False))      # config.rope_fp32
        half = rl // 2
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=self.dev).float() / half))

        # Matmul weights in the compute dtype (what autocast would cast them to on every call)
        cast = (lambda w: w.detach().to(self.dtype)) if self.dtype != torch.float32 else (lambda w: w.detach())
        self.w = []
        for b in self.blocks:
            a, m = b.attn, b.mlp
            self.w.append({k: (cast(lin.weight), None if lin.bias is None else cast(lin.bias)) for k, lin in (
                ("q", a.c_attn_q), ("k", a.c_attn_k), ("v", a.c_attn_v), ("o", a.c_proj),
                ("fc", m.c_fc), ("proj", m.c_proj))})
        self.heads = [cast(t[f"lm_head_{i}"].weight) for i in range(self.K)]

        N, L = self.N, self.L
        self.k_cache = [torch.zeros(N, self.nkv, L, self.hd, device=self.dev, dtype=self.dtype) for _ in self.blocks]
        self.v_cache = [torch.zeros(N, self.nkv, L, self.hd, device=self.dev, dtype=self.dtype) for _ in self.blocks]
        self.tok = torch.zeros(N, self.K, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros((), dtype=torch.long, device=self.dev)
        self.out_idx = torch.zeros((), dtype=torch.long, device=self.dev)
        self.out = torch.zeros(N, self.K, L, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(L, device=self.dev)
        self.head_to_kv = att.head_to_kv
        self.teacher = None                 # (N, K) tokens to feed instead of samples (check mode, eager only)
        self.last_logits: List[torch.Tensor] = []
        self.graph = None
        self.use_cuda_graph = bool(use_cuda_graph and self.dev.type == "cuda")

    # ------------------------------------------------------------------ pieces
    def _linear(self, x: torch.Tensor, wb) -> torch.Tensor:
        w, b = wb
        return F.linear(x.to(w.dtype), w, b)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.rope_fp32:
            return self._rope_pairs(x.float(), cos, sin).to(x.dtype)
        return self._rope_pairs(x, cos, sin)

    def _rope_pairs(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = self.rope_n
        x_even, x_odd = x[..., :n:2], x[..., 1:n:2]
        out = torch.empty_like(x)
        out[..., :n:2], out[..., 1:n:2] = x_even * cos - x_odd * sin, x_even * sin + x_odd * cos
        if n < x.shape[-1]:
            out[..., n:] = x[..., n:]
        return out

    def _sample(self, lg: torch.Tensor) -> torch.Tensor:
        """(N, V) float32 logits -> (N,) ids: Gumbel-max over softmax(top_k(lg) / temperature)."""
        if self.temperature <= 0:
            return lg.argmax(dim=-1)
        lg = lg / self.temperature
        if self.top_k is not None and self.top_k < lg.size(-1):
            v, _ = torch.topk(lg, self.top_k)
            lg = lg.masked_fill(lg < v[:, -1:], float("-inf"))
        u = torch.rand(lg.shape, device=lg.device, dtype=torch.float32).clamp_(min=1e-12)
        return (lg - torch.log(-torch.log(u))).argmax(dim=-1)

    def _step(self) -> None:
        """One decode row for all samples at position self.pos (device tensor); advances the state."""
        t = self.model.transformer
        reps = [t[f"wte_{i}"](self.tok[:, i]) for i in range(self.K)]
        x = (reps[0] if self.K == 1 else torch.stack(reps, dim=0).sum(dim=0)).unsqueeze(1)   # (N, 1, C) fp32
        N, nh, nkv, hd, C = self.N, self.nh, self.nkv, self.hd, self.C
        # Rotary angles as the model computes them. rope_fp32: in float32 (and the rotation too). Otherwise
        # einsum(pos, inv_freq), which bf16 autocast runs in bf16 (so training used bf16-rounded angles); the
        # elementwise product in that dtype is bit-identical.
        if self.rope_fp32:
            angles = self.pos.to(torch.float32).view(1, 1) * self.inv_freq.view(1, -1)
        else:
            angles = self.pos.to(self.dtype).view(1, 1) * self.inv_freq.to(self.dtype).view(1, -1)    # (1, half)
        cos, sin = angles.cos(), angles.sin()
        mask = (self.ar <= self.pos).view(1, 1, 1, self.L)
        idx = self.pos.view(1)
        for li, b in enumerate(self.blocks):
            w = self.w[li]
            h = b.pre_ln_attn(x)
            q = self._linear(h, w["q"]).view(N, 1, nh, hd).transpose(1, 2)
            k = self._linear(h, w["k"]).view(N, 1, nkv, hd).transpose(1, 2)
            v = self._linear(h, w["v"]).view(N, 1, nkv, hd).transpose(1, 2)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            self.k_cache[li].index_copy_(2, idx, k)
            self.v_cache[li].index_copy_(2, idx, v)
            kc, vc = self.k_cache[li], self.v_cache[li]
            if nkv != nh:
                kc, vc = kc.index_select(1, self.head_to_kv), vc.index_select(1, self.head_to_kv)
            y = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
            x = x + self._linear(y.transpose(1, 2).reshape(N, 1, C), w["o"])
            m = b.mlp
            z = self._linear(b.pre_ln_mlp(x), w["fc"])
            z = m.activation_variant(z - m.activation_x_offset) - m.activation_y_offset
            x = x + self._linear(z, w["proj"])
        x = t["ln_f"](x)[:, 0, :]
        logits = [F.linear(x.to(hw.dtype), hw).float() for hw in self.heads]
        if self.teacher is not None:          # check mode: keep the logits, feed the given tokens
            self.last_logits = logits
            nxt = self.teacher
        else:
            nxt = torch.stack([self._sample(lg) for lg in logits], dim=1)             # (N, K)
        self.tok.copy_(nxt)
        self.out.index_copy_(2, self.out_idx.view(1), nxt.unsqueeze(-1))
        self.pos.add_(1)
        self.out_idx.add_(1)

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                # warm-up (also builds cuBLAS / SDPA workspaces)
                self.pos.fill_(0)
                self.out_idx.fill_(0)
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.pos.fill_(0)
        self.out_idx.fill_(0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def prefill(self, context: torch.Tensor) -> List[torch.Tensor]:
        """Fill the cache from one context (K, T) and return its last-row logits (float32, one per lane)."""
        ctx = context.to(self.dev).long()
        if ctx.dim() == 3:
            if ctx.shape[0] != 1:
                raise ValueError("one context at a time")
            ctx = ctx[0]
        T = ctx.shape[-1]
        amp = (torch.autocast(device_type="cuda", dtype=self.dtype)
               if self.dev.type == "cuda" and self.dtype != torch.float32 else torch.autocast("cpu", enabled=False))
        with amp:
            out = self.model(ctx.unsqueeze(0), use_cache=True, last_step_only=True)
        for li, (pk, pv) in enumerate(out.past_key_values):
            self.k_cache[li][:, :, :T].copy_(pk.expand(self.N, -1, -1, -1))
            self.v_cache[li][:, :, :T].copy_(pv.expand(self.N, -1, -1, -1))
        self.pos.fill_(T)
        self.out_idx.fill_(0)
        return [lg[0, -1].float() for lg in out.logits]

    @torch.no_grad()
    def forecast(self, context: torch.Tensor, horizon: int, generator_seed: Optional[int] = None) -> torch.Tensor:
        """num_samples sampled continuations of `context` (K, T): returns (num_samples, K, horizon) ids."""
        T = context.shape[-1]
        if T + horizon - 1 > self.L:
            raise ValueError(f"context {T} + horizon {horizon} - 1 > block_size {self.L}")
        if self.use_cuda_graph and self.graph is None:
            self._capture()
        if generator_seed is not None:
            torch.manual_seed(generator_seed)
        last = self.prefill(context)
        first = torch.stack([self._sample(lg.unsqueeze(0).expand(self.N, -1).contiguous()) for lg in last], dim=1)
        self.tok.copy_(first)
        self.out[:, :, 0] = first
        self.out_idx.fill_(1)
        for _ in range(horizon - 1):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
        return self.out[:, :, :horizon].clone()

    @torch.no_grad()
    def teacher_logits(self, context: torch.Tensor, continuation: torch.Tensor) -> List[torch.Tensor]:
        """Check mode (eager): logits for every continuation row given the true previous rows.
        context (K, T), continuation (K, H) -> per lane (H, V) float32 (row j predicts continuation[:, j])."""
        last = self.prefill(context)
        rows = [[lg] for lg in last]
        cont = continuation.to(self.dev).long()
        try:
            for j in range(cont.shape[-1] - 1):
                self.teacher = cont[:, j].unsqueeze(0).expand(self.N, -1).contiguous()
                self.tok.copy_(self.teacher)
                self._step()
                for i in range(self.K):
                    rows[i].append(self.last_logits[i][0])
        finally:
            self.teacher = None
        return [torch.stack(r) for r in rows]


# ---------------------------------------------------------------------- checks / benchmark
def _load_val(dataset: Path):
    import json
    import numpy as np
    man = json.load(open(dataset / "manifest.json"))
    lanes = [n.split("/")[-1] for n in man["multicontext_datasets"]]
    return np.stack([np.fromfile(dataset / ln / "val.bin", dtype=np.uint16) for ln in lanes]).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--context", type=int, default=126)
    ap.add_argument("--horizon", type=int, default=130)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--windows", type=int, default=8, help="held-out windows for the checks")
    ap.add_argument("--repeats", type=int, default=20, help="timed forecasts")
    ap.add_argument("--check", action="store_true", help="compare with the model's own forward / generate_forecast")
    a = ap.parse_args()

    import numpy as np
    dev = torch.device("cuda")
    model = CsvMcIntForCausalLM.load_dir(a.model, map_location=dev)
    val = _load_val(Path(a.dataset))
    rng = np.random.default_rng(1234)
    L = a.context + a.horizon
    starts = np.sort(rng.integers(0, val.shape[1] - L, size=a.windows))
    print(f"[fast_forecast] {a.model}: {sum(p.numel() for p in model.parameters()):,} params, lanes {model.config.vocab_sizes}, "
          f"{a.context} context rows -> {a.horizon} rows x {a.samples} samples", flush=True)

    if a.check:
        torch.backends.cuda.matmul.allow_tf32 = False

        def amp_ctx(dtype):
            return (torch.autocast("cuda", dtype=dtype) if dtype != torch.float32
                    else torch.autocast("cuda", enabled=False))

        def ref_full(win, dtype):          # one forward over context + continuation
            with torch.no_grad(), amp_ctx(dtype):
                lg = model(win[:, :L - 1].unsqueeze(0), last_step_only=False).logits
            return [x[0, a.context - 1:].float() for x in lg]

        def ref_incremental(ctx, cont, dtype):   # the model's own KV-cache path, one row at a time
            with torch.no_grad(), amp_ctx(dtype):
                out = model(ctx.unsqueeze(0), use_cache=True, last_step_only=True)
                rows = [[x[0, -1].float()] for x in out.logits]
                past = out.past_key_values
                for j in range(cont.shape[-1] - 1):
                    out = model(cont[:, j:j + 1].unsqueeze(0), past_key_values=past, use_cache=True,
                                last_step_only=True)
                    past = out.past_key_values
                    for i, x in enumerate(out.logits):
                        rows[i].append(x[0, -1].float())
            return [torch.stack(r) for r in rows]

        def compare(xs, ys, cont):
            worst = mean = agree = n = 0.0
            nll_x = nll_y = 0.0
            for x_l, y_l, c in zip(xs, ys, cont):
                for x, y, t in zip(x_l, y_l, c):
                    d = (x - y).abs()
                    worst, mean = max(worst, float(d.max())), mean + float(d.mean()) * x.shape[0]
                    agree += float((x.argmax(-1) == y.argmax(-1)).sum())
                    n += x.shape[0]
                    nll_x -= float(F.log_softmax(x, -1).gather(-1, t.unsqueeze(-1)).sum())
                    nll_y -= float(F.log_softmax(y, -1).gather(-1, t.unsqueeze(-1)).sum())
            return (f"max |diff| {worst:.4f}, mean |diff| {mean / n:.5f}, argmax agree {agree / n:.2%}, "
                    f"true-row loss {nll_x / n:.5f} vs {nll_y / n:.5f}")

        for dtype in (torch.float32, torch.bfloat16):
            fc = FastForecaster(model, num_samples=1, top_k=a.top_k, dtype=dtype, use_cuda_graph=False)
            fast, full, inc, conts = [], [], [], []
            for s in starts:
                win = torch.from_numpy(val[:, s:s + L]).to(dev)
                ctx, cont = win[:, :a.context], win[:, a.context:]
                fast.append(fc.teacher_logits(ctx, cont))
                full.append(ref_full(win, dtype))
                inc.append(ref_incremental(ctx, cont, dtype))
                conts.append(cont)
            name = "float32" if dtype == torch.float32 else "bf16"
            print(f"  [{name}] fast vs full forward:        {compare(fast, full, conts)}", flush=True)
            print(f"  [{name}] fast vs model's KV-cache path: {compare(fast, inc, conts)}", flush=True)
            print(f"  [{name}] model's KV-cache vs full:      {compare(inc, full, conts)}", flush=True)
        # greedy decode: CUDA graph vs generate_forecast (temperature 0), float32 so near-ties don't flip
        fg = FastForecaster(model, num_samples=1, temperature=0.0, dtype=torch.float32)
        same, tot = 0, 0
        for s in starts:
            ctx = torch.from_numpy(val[:, s:s + a.context]).to(dev)
            mine = fg.forecast(ctx, a.horizon)[0]
            with torch.no_grad():
                ref = model.generate_forecast(ctx, a.horizon, num_samples=1, temperature=0.0)[0, 0]
            first_diff = (mine != ref).any(dim=0).nonzero()
            same += int(first_diff[0]) if len(first_diff) else a.horizon
            tot += a.horizon
        print(f"  [float32, CUDA graph] greedy rows identical to generate_forecast before the first difference: "
              f"{same / tot:.1%} of {tot}", flush=True)

    # 3) speed: CUDA-graph forecasts vs generate_forecast
    fc = FastForecaster(model, num_samples=a.samples, top_k=a.top_k)
    ctx = torch.from_numpy(val[:, starts[0]:starts[0] + a.context]).to(dev)
    fc.forecast(ctx, a.horizon)                                      # capture + warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for r in range(a.repeats):
        s = starts[r % len(starts)]
        fc.forecast(torch.from_numpy(val[:, s:s + a.context]).to(dev), a.horizon)
    torch.cuda.synchronize()
    fast_s = (time.perf_counter() - t0) / a.repeats
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.generate_forecast(ctx, a.horizon, num_samples=a.samples, top_k=a.top_k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for r in range(3):
            model.generate_forecast(ctx, a.horizon, num_samples=a.samples, top_k=a.top_k)
        torch.cuda.synchronize()
    ref_s = (time.perf_counter() - t0) / 3
    print(f"  one forecast ({a.samples} x {a.horizon} rows): CUDA graph {fast_s * 1000:.1f} ms "
          f"({fast_s / a.horizon * 1000:.2f} ms/row) vs generate_forecast {ref_s * 1000:.1f} ms -> {ref_s / fast_s:.1f}x",
          flush=True)


if __name__ == "__main__":
    main()

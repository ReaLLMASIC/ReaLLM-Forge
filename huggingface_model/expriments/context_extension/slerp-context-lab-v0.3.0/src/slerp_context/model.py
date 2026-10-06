"""Full-weight Hunyuan dense, Llama and Pythia recurrent wrappers.

Original pretrained modules are reused. K/V caches contain pre-RoPE keys.
Hunyuan Q/K norms are applied AFTER RoPE, exactly as in Transformers.
Backbone weights and updates are FP32; CUDA autocast computes matmuls in BF16.
"""
from functools import partial
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from .memory import SphericalMemory, DeltaMemory
from .state import LayerState, StreamState


def apply_rope(x, cos, sin):
    dim = cos.shape[-1]
    rot, tail = x[..., :dim], x[..., dim:]
    half = dim // 2
    twisted = torch.cat((-rot[..., half:], rot[..., :half]), -1)
    return torch.cat((rot * cos[:, None] + twisted * sin[:, None], tail), -1)


def build_base(cfg, base_config=None):
    torch.manual_seed(cfg.seed)
    if base_config is not None:
        bc = AutoConfig.from_pretrained(base_config, local_files_only=True)
        base = AutoModelForCausalLM.from_config(bc, attn_implementation="sdpa").float()
        if (Path(base_config)/"generation_config.json").exists():
            base.generation_config=GenerationConfig.from_pretrained(base_config,local_files_only=True)
        return base
    if cfg.tiny:
        common = dict(vocab_size=259, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=2048, bos_token_id=0, eos_token_id=0,
                      pad_token_id=0, attention_dropout=0.0)
        if cfg.tiny_arch == "gpt_neox":
            common.update(rotary_pct=0.5, hidden_dropout=0.0, use_parallel_residual=True)
        else:
            common.update(num_key_value_heads=2, head_dim=12, tie_word_embeddings=True)
        bc = AutoConfig.for_model(cfg.tiny_arch, **common)
        return AutoModelForCausalLM.from_config(bc, attn_implementation="sdpa").float()
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, revision=cfg.revision, dtype=torch.float32,
        attn_implementation="sdpa", use_safetensors=True, trust_remote_code=False)
    cfg.revision = base.config._commit_hash or cfg.revision
    return base


class RecurrentLM(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.cfg = cfg.validate()
        bc = base.config
        self.arch = bc.model_type
        if self.arch not in {"gpt_neox", "llama", "hunyuan_v1_dense"}:
            raise ValueError(f"Unsupported architecture: {self.arch}")
        if self.arch == "gpt_neox" and (not bc.use_parallel_residual or bc.hidden_dropout):
            raise ValueError("Pythia requires parallel residuals and zero dropout")
        if bc.attention_dropout != 0:
            raise ValueError("Zero attention dropout is required")
        self.backbone = base.float().requires_grad_(True)
        self.heads = bc.num_attention_heads
        self.kv_heads = getattr(bc, "num_key_value_heads", self.heads)
        self.dim = bc.hidden_size // self.heads if self.arch == "gpt_neox" else self.layers[0].self_attn.head_dim
        self.groups = self.heads // self.kv_heads
        self.native_context = bc.max_position_embeddings
        if cfg.window > self.native_context and cfg.method != "pi":
            raise ValueError("Local window exceeds native context")
        memory_class = DeltaMemory if cfg.method == "delta" else SphericalMemory
        self.memories = nn.ModuleList([
            memory_class(self.kv_heads, self.dim, cfg) for _ in self.layers
        ]) if cfg.method not in {"local", "pi", "native"} else nn.ModuleList()

    @property
    def original(self):
        return self.backbone

    @property
    def transformer(self):
        return self.backbone.gpt_neox if self.arch == "gpt_neox" else self.backbone.model

    @property
    def layers(self):
        return self.transformer.layers

    @property
    def embedding(self):
        return self.backbone.get_input_embeddings()

    @property
    def lm_head(self):
        return self.backbone.get_output_embeddings()

    @classmethod
    def build(cls, cfg, device="cpu", base_config=None):
        if cfg.method == "rmt":
            from .rmt import RMTLM
            return RMTLM(build_base(cfg, base_config=base_config), cfg).to(device)
        return cls(build_base(cfg, base_config=base_config), cfg).to(device)

    def save_pretrained(self, directory):
        from .storage import save_weights
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        save_weights(self, path)

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        from .config import Config
        from .storage import checkpoint_path, load_weights
        path = checkpoint_path(directory)
        model = cls.build(Config.load(path / "config.json"), device, path / "backbone_config")
        load_weights(model, path)
        return model

    def initial_state(self):
        p = self.embedding.weight
        states = []
        for _ in self.layers:
            kv = p.new_zeros((1, self.kv_heads, 0, self.dim))
            count = self.dim if self.cfg.method == "delta" else self.cfg.slots
            m = p.new_zeros((1, self.kv_heads, count if self.memories else 0, self.dim))
            if self.cfg.method != "delta":
                m[..., 0] = 1
            r = p.new_zeros((*m.shape[:-1], 1))
            states.append(LayerState(kv, kv, m, r, torch.zeros_like(r, dtype=torch.bool)))
        return StreamState(0, tuple(states))

    def next_chunk_size(self, state, remaining):
        return min(remaining, self.cfg.chunk - state.consumed % self.cfg.chunk)

    def _layer(self, x, k, v, m, lr, valid, cos, sin, *, layer_id, evict, memory_enabled):
        layer = self.layers[layer_id]
        if evict:
            if self.memories and memory_enabled:
                m, lr, valid = self.memories[layer_id].write(
                    k[:, :, :self.cfg.chunk], v[:, :, :self.cfg.chunk], m, lr, valid)
            k, v = k[:, :, self.cfg.chunk:], v[:, :, self.cfg.chunk:]
        residual = x
        h = layer.input_layernorm(x)
        if self.arch == "gpt_neox":
            a = layer.attention
            qkv = a.query_key_value(h).view(1, x.shape[1], self.heads, 3*self.dim)
            q, nk, nv = qkv.permute(0, 2, 1, 3).split(self.dim, dim=-1)
        else:
            a = layer.self_attn
            shape = (1, x.shape[1], -1, self.dim)
            q = a.q_proj(h).view(shape).transpose(1, 2)
            nk = a.k_proj(h).view(shape).transpose(1, 2)
            nv = a.v_proj(h).view(shape).transpose(1, 2)
        # Avoid promoting BF16 projected keys via the empty FP32 initial cache.
        old = k.shape[-2]
        k = torch.cat((k.to(nk.dtype), nk), -2)
        v = torch.cat((v.to(nv.dtype), nv), -2)
        qr = apply_rope(q, cos[:, old:].to(q.dtype), sin[:, old:].to(q.dtype))
        kr = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        if self.arch == "hunyuan_v1_dense":
            qr, kr = a.query_layernorm(qr), a.key_layernorm(kr)
        positions = torch.arange(k.shape[-2], device=x.device)
        allowed = positions[None, :] <= positions[old:, None]
        kr = kr.repeat_interleave(self.groups, dim=1) if self.groups > 1 else kr
        vr = v.repeat_interleave(self.groups, dim=1) if self.groups > 1 else v
        out = F.scaled_dot_product_attention(qr, kr, vr, attn_mask=allowed, dropout_p=0.0)
        if self.memories and memory_enabled:
            out = out + self.memories[layer_id].read(q, m, lr, valid).to(out.dtype)
        out = out.transpose(1, 2).reshape(1, x.shape[1], -1)
        if self.arch == "gpt_neox":
            out = a.dense(out)
            x = layer.mlp(layer.post_attention_layernorm(residual)) + out + residual
        else:
            x = residual + a.o_proj(out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k, v, m, lr, valid

    def consume_chunk(self, ids, state, memory_enabled=True):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Batch 1, unpadded, nonempty streams only")
        if ids.shape[1] > self.next_chunk_size(state, ids.shape[1]):
            raise ValueError("Use consume() to split logical chunk boundaries")
        unbounded = self.cfg.method in {"pi", "native"}
        if self.cfg.method == "native" and state.consumed + ids.shape[1] > self.native_context:
            raise ValueError("Native context exceeded")
        evict = not unbounded and state.layers[0].k.shape[-2] == self.cfg.window
        x = self.embedding(ids)
        old = state.layers[0].k.shape[-2] - (self.cfg.chunk if evict else 0)
        positions = torch.arange(old + ids.shape[1], device=x.device)[None]
        if self.cfg.method == "pi":
            positions = positions.float() / self.cfg.pi_factor
        # Native RoPE outside checkpoint closures avoids rotary-state mutation
        # during backward recomputation. Native context metadata is never enlarged.
        cos, sin = self.transformer.rotary_emb(x, positions)
        new = []
        for i, st in enumerate(state.layers):
            fn = partial(self._layer, layer_id=i, evict=evict, memory_enabled=memory_enabled)
            args = (x, *st.tensors(), cos, sin)
            out = checkpoint(fn, *args, use_reentrant=False) if self.training and self.cfg.checkpoint and torch.is_grad_enabled() else fn(*args)
            x, k, v, m, lr, valid = out
            if self.cfg.detach_local:
                k, v = k.detach(), v.detach()
            new.append(LayerState(k, v, m, lr, valid))
        norm = self.transformer.final_layer_norm if self.arch == "gpt_neox" else self.transformer.norm
        return norm(x), StreamState(state.consumed + ids.shape[1], tuple(new))

    def consume(self, ids, state=None, memory_enabled=True, all_logits=True):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Batch 1, unpadded, nonempty streams only")
        state = self.initial_state() if state is None else state
        chunks, start = [], 0
        while start < ids.shape[1]:
            n = self.next_chunk_size(state, ids.shape[1]-start)
            h, state = self.consume_chunk(ids[:, start:start+n], state, memory_enabled)
            if all_logits:
                chunks.append(self.lm_head(h))
            start += n
        logits = torch.cat(chunks, 1) if all_logits else self.lm_head(h[:, -1:])
        return logits, state

    def tail_logits(self, ids, last_n):
        state, start, tail = self.initial_state(), 0, None
        while start < ids.shape[1]:
            n = self.next_chunk_size(state, ids.shape[1]-start)
            h, state = self.consume_chunk(ids[:, start:start+n], state)
            tail = h[:, -last_n:] if tail is None else torch.cat((tail, h), 1)[:, -last_n:]
            start += n
        return self.lm_head(tail)

    def episode_loss(self, ids, labels, memory_enabled=True):
        if ids.shape != labels.shape:
            raise ValueError("Expected aligned input and next-token labels")
        state, start, total = self.initial_state(), 0, None
        scored = int((labels != -100).sum())
        if scored == 0:
            raise ValueError("Episode has no supervised tokens")
        while start < ids.shape[1]:
            n = self.next_chunk_size(state, ids.shape[1]-start)
            h, state = self.consume_chunk(ids[:, start:start+n], state, memory_enabled)
            y = labels[:, start:start+n]
            mask = y != -100
            if mask.any():
                selected, target = h[mask], y[mask]
                for j in range(0, len(target), self.cfg.loss_chunk_tokens):
                    def loss_fn(hidden, targets):
                        return F.cross_entropy(self.lm_head(hidden).float(), targets, reduction="sum")
                    args = (selected[j:j+self.cfg.loss_chunk_tokens], target[j:j+self.cfg.loss_chunk_tokens])
                    loss = checkpoint(loss_fn, *args, use_reentrant=False) if self.training and self.cfg.checkpoint else loss_fn(*args)
                    total = loss if total is None else total + loss
            start += n
        return total / scored, state

    def completion_logps(self, prompt, completion):
        all_ids = torch.cat((prompt, completion), 1)
        state, start, selected = self.initial_state(), 0, []
        inp, targets = all_ids[:, :-1], all_ids[:, 1:]
        while start < inp.shape[1]:
            n = self.next_chunk_size(state, inp.shape[1]-start)
            h, state = self.consume_chunk(inp[:, start:start+n], state)
            begin = max(0, prompt.shape[1] - 1 - start)
            for j in range(begin, n, self.cfg.loss_chunk_tokens):
                end = min(n, j+self.cfg.loss_chunk_tokens)
                def logp_fn(hidden, target):
                    return -F.cross_entropy(self.lm_head(hidden).float().reshape(-1, self.original.config.vocab_size), target.flatten(), reduction="none")
                args = (h[:, j:end], targets[:, start+j:start+end])
                selected.append(checkpoint(logp_fn, *args, use_reentrant=False) if self.training and self.cfg.checkpoint else logp_fn(*args))
            start += n
        return torch.cat(selected)

    @torch.no_grad()
    def generate_stream(self, prompt, max_new_tokens, temperature=0.0, top_k=0,
                        state=None, eos_token_id=None, memory_enabled=True):
        self.eval()
        if max_new_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError("Generation settings must be nonnegative")
        logits, state = self.consume(prompt, state, memory_enabled, all_logits=False)
        generated = []
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        for _ in range(max_new_tokens):
            scores = logits[:, -1].float()
            if temperature > 0:
                scores = scores / temperature
                if top_k:
                    cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, -torch.inf)
                token = torch.multinomial(scores.softmax(-1), 1)
            else:
                token = scores.argmax(-1, keepdim=True)
            generated.append(token)
            logits, state = self.consume(token, state, memory_enabled, all_logits=False)
            if token.item() in eos:
                break
        return (torch.cat(generated, 1) if generated else prompt[:, :0]), state

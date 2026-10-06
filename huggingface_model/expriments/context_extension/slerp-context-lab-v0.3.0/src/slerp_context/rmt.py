"""Full-weight causal recurrent soft-memory transformer.

Every segment reads m hidden vectors before its tokens and writes m vectors
after them. Source-token hidden states cannot see the write positions. Segment
positions reset; the pretrained native context metadata is never changed.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .storage import tensor_bytes


def cache_bytes(cache):
    if cache is None:
        return 0
    if hasattr(cache, "layers"):
        return sum(tensor_bytes(getattr(layer, name, None))
                   for layer in cache.layers for name in ("keys", "values"))
    return tensor_bytes(cache)


@dataclass
class RMTState:
    consumed: int
    memory: torch.Tensor
    cache: object = None
    segment_tokens: int = 0

    def nbytes(self):
        return tensor_bytes(self.memory) + cache_bytes(self.cache)


class RMTLM(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        if base.config.model_type not in {"llama", "hunyuan_v1_dense"}:
            raise ValueError("RMT currently supports Llama and Hunyuan dense")
        self.backbone, self.cfg = base.float().requires_grad_(True), cfg.validate()
        self.native_context = base.config.max_position_embeddings
        if cfg.chunk + 2 * cfg.slots > min(cfg.window, self.native_context):
            raise ValueError("RMT chunk + 2*slots must fit both window and native context")
        d = base.config.hidden_size
        scale = base.config.initializer_range
        self.memories = nn.ParameterDict({
            "initial": nn.Parameter(torch.randn(1, cfg.slots, d) * scale),
            "write": nn.Parameter(torch.randn(1, cfg.slots, d) * scale)})

    @property
    def original(self): return self.backbone

    @property
    def embedding(self): return self.backbone.get_input_embeddings()

    @property
    def lm_head(self): return self.backbone.get_output_embeddings()

    def initial_state(self):
        return RMTState(0, self.memories["initial"])

    def save_pretrained(self,directory):
        from .storage import save_weights
        save_weights(self,directory)

    def segment(self, ids, memory):
        m = self.cfg.slots
        embeds = torch.cat((memory, self.embedding(ids), self.memories["write"]), dim=1)
        h = self.backbone.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
        return h[:, m:m+ids.shape[1]], h[:, -m:]

    def episode_loss(self, ids, labels, memory_enabled=True):
        if ids.shape != labels.shape or ids.shape[0] != 1:
            raise ValueError("Batch-one aligned next-token labels required")
        scored = int((labels != -100).sum())
        if not scored: raise ValueError("No supervised tokens")
        memory, total = self.memories["initial"], None
        for start in range(0, ids.shape[1], self.cfg.chunk):
            x, y = ids[:, start:start+self.cfg.chunk], labels[:, start:start+self.cfg.chunk]
            args = (x, memory)
            h, memory = checkpoint(self.segment, *args, use_reentrant=False) if self.cfg.checkpoint and self.training else self.segment(*args)
            mask = y != -100
            if mask.any():
                hidden, target = h[mask], y[mask]
                for j in range(0, len(target), self.cfg.loss_chunk_tokens):
                    def loss_fn(a, b): return F.cross_entropy(self.lm_head(a).float(), b, reduction="sum")
                    args = (hidden[j:j+self.cfg.loss_chunk_tokens], target[j:j+self.cfg.loss_chunk_tokens])
                    loss = checkpoint(loss_fn, *args, use_reentrant=False) if self.cfg.checkpoint and self.training else loss_fn(*args)
                    total = loss if total is None else total + loss
        return total / scored, RMTState(ids.shape[1], memory)

    def tail_logits(self, ids, last_n):
        memory, tail = self.memories["initial"], None
        for start in range(0, ids.shape[1], self.cfg.chunk):
            h, memory = self.segment(ids[:, start:start+self.cfg.chunk], memory)
            tail = h[:, -last_n:] if tail is None else torch.cat((tail, h), 1)[:, -last_n:]
        return self.lm_head(tail)

    @torch.no_grad()
    def prefill(self, ids):
        memory = self.memories["initial"]
        # Keep a nonempty last segment as the live generation cache.
        start = 0
        while ids.shape[1] - start > self.cfg.chunk:
            _, memory = self.segment(ids[:, start:start+self.cfg.chunk], memory)
            start += self.cfg.chunk
        last = ids[:, start:]
        out = self.backbone.model(inputs_embeds=torch.cat((memory, self.embedding(last)), 1), use_cache=True)
        state = RMTState(ids.shape[1], memory, out.past_key_values, last.shape[1])
        return self.lm_head(out.last_hidden_state[:, -1:]), state

    @torch.no_grad()
    def step(self, token, state):
        memory, cache, count = state.memory, state.cache, state.segment_tokens
        if count == self.cfg.chunk:
            out = self.backbone.model(inputs_embeds=self.memories["write"], past_key_values=cache, use_cache=True)
            memory, cache, count = out.last_hidden_state, None, 0
        embeds = self.embedding(token)
        if cache is None: embeds = torch.cat((memory, embeds), 1)
        out = self.backbone.model(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        return self.lm_head(out.last_hidden_state[:, -1:]), RMTState(
            state.consumed+1, memory, out.past_key_values, count+1)

    @torch.no_grad()
    def generate_stream(self, prompt, max_new_tokens, eos_token_id=None, **kwargs):
        self.eval()
        logits, state = self.prefill(prompt)
        outputs = []
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        for i in range(max_new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True); outputs.append(token)
            if token.item() in eos or i+1 == max_new_tokens: break
            logits, state = self.step(token, state)
        return torch.cat(outputs, 1) if outputs else prompt[:, :0], state

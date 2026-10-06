"""Unmodified Hugging Face generation baseline; no custom attention forward."""
from dataclasses import dataclass
import torch
from torch import nn
from .model import build_base


@dataclass
class NativeState:
    consumed: int
    cache_bytes: int

    def nbytes(self):
        return self.cache_bytes


class NativeLM(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.backbone, self.cfg = base, cfg
        self.native_context = base.config.max_position_embeddings

    @classmethod
    def build(cls, cfg, device):
        return cls(build_base(cfg).to(device), cfg).eval()

    @torch.no_grad()
    def generate_stream(self, prompt, max_new_tokens, eos_token_id=None, memory_enabled=True):
        if prompt.shape[1] + max_new_tokens > self.native_context:
            raise ValueError("Prompt plus generation exceeds the unchanged native context")
        result = self.backbone.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, logits_to_keep=1,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            temperature=None, top_p=None, top_k=None,
            eos_token_id=eos_token_id, pad_token_id=self.backbone.config.pad_token_id or 0,
            return_dict_in_generate=True)
        cache = result.past_key_values
        size = 0
        for layer in getattr(cache, "layers", []):
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if tensor is not None: size += tensor.numel()*tensor.element_size()
        consumed = int(cache.get_seq_length()) if cache is not None else 0
        # HF's returned cache usually excludes the final generated token. Report
        # its actual occupancy rather than claiming equivalence to StreamState.
        return result.sequences[:, prompt.shape[1]:], NativeState(consumed, size)

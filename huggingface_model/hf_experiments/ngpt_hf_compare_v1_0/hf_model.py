"""Hugging Face AutoConfig / AutoModelForCausalLM integration.

Import this module before AutoModelForCausalLM.from_pretrained(local_directory).
This registers the model locally; remote-code execution is not necessary.
"""
from dataclasses import asdict, fields
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from model import ModelConfig, Transformer


class NGPTConfig(PretrainedConfig):
    model_type = "ngpt_matched_comparison"

    def __init__(self, **kwargs):
        names = {f.name for f in fields(ModelConfig)}
        settings = {k: kwargs.pop(k) for k in list(kwargs) if k in names}
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        for k, v in asdict(ModelConfig(**settings)).items():
            setattr(self, k, v)
        self.hidden_size = self.width
        self.num_hidden_layers = self.layers
        self.num_attention_heads = self.heads
        self.max_position_embeddings = self.context
        self.use_cache = False

    def core_config(self):
        return ModelConfig(**{f.name: getattr(self, f.name) for f in fields(ModelConfig)})


class NGPTForCausalLM(PreTrainedModel):
    config_class = NGPTConfig
    base_model_prefix = "net"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.net = Transformer(config.core_config())
        self.post_init()

    def _init_weights(self, module):
        # Initialization/projection already performed by Transformer, in identical
        # RNG order for the two variants. Do not overwrite it in post_init().
        pass

    def get_input_embeddings(self):
        return self.net.embed

    def set_input_embeddings(self, value):
        self.net.embed = value

    def get_output_embeddings(self):
        return self.net.lm_head

    def set_output_embeddings(self, value):
        self.net.lm_head = value

    def next_token_loss(self, tokens, chunk_tokens=128):
        return self.net.next_token_loss(tokens, chunk_tokens)

    def project_weights_(self):
        self.net.project_weights_()

    def diagnostics(self):
        return self.net.diagnostics()

    def forward(self, input_ids=None, labels=None, attention_mask=None, return_dict=None, **kwargs):
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise ValueError("This small comparison model supports unpadded packed sequences only.")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("KV caching is not implemented in this comparison model.")
        logits = self.net(input_ids)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or input_ids.shape[1] < 2:
                raise ValueError("HF labels must match input_ids with at least two positions.")
            # Standard Hugging Face convention: labels=input_ids; shift internally.
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        if return_dict is False or (return_dict is None and not self.config.use_return_dict):
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits)


AutoConfig.register(NGPTConfig.model_type, NGPTConfig)
AutoModelForCausalLM.register(NGPTConfig, NGPTForCausalLM)

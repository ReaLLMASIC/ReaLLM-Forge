"""A shared HF GPT-NeoX decoder with explicit tokenizer conditioning.

Training can omit global logits to project only the selected lexical rows.
Generation returns standard global-shaped masked logits and preserves route IDs
through the KV cache. Labels follow the usual HF unshifted causal-LM convention.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)

try:
    from .configuration_mixture import MixtureConfig, route_index
except ImportError:
    from configuration_mixture import MixtureConfig, route_index


@dataclass
class MixtureCausalLMOutput(CausalLMOutputWithPast):
    route_losses: Optional[torch.FloatTensor] = None
    route_target_counts: Optional[torch.LongTensor] = None


class MixtureForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    config_class = MixtureConfig
    base_model_prefix = "gpt_neox"
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.tokenizer_embeddings = nn.Embedding(5, config.hidden_size)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        output_mask = torch.zeros(5, config.vocab_size, dtype=torch.bool)
        inverse = torch.full((5, config.vocab_size), -1, dtype=torch.long)
        for route in range(5):
            allowed = torch.tensor(config.allowed_ids(route), dtype=torch.long)
            output_mask[route, allowed] = True
            inverse[route, allowed] = torch.arange(len(allowed))
        self.register_buffer("allowed_output_mask", output_mask, persistent=False)
        self.register_buffer("global_to_local", inverse, persistent=False)
        self.post_init()
        self._matched_initialization()

    def _matched_initialization(self):
        """Keep common backbone/byte/lexical-prefix weights matched across K.

        Independent RNG streams prevent a larger WTE from advancing the
        backbone RNG. For the nested tokenizer export, each language's retained
        lexical prefix has the same values in both size conditions.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initializer_seed)
            # Skip input/output embeddings. Their shapes differ across K.
            for module in self.gpt_neox.layers.modules():
                self._init_weights(module)
            self._init_weights(self.gpt_neox.final_layer_norm)
            weight = self.get_input_embeddings().weight
            if weight.device.type == "meta":
                return
            with torch.no_grad():
                torch.manual_seed(self.config.initializer_seed + 100)
                nn.init.normal_(weight[:256], std=self.config.initializer_range)
                for route in range(1, 5):
                    torch.manual_seed(self.config.initializer_seed + 100 + route)
                    block = self.config.lexical_range(route)
                    nn.init.normal_(weight[block.start:block.stop], std=self.config.initializer_range)
                torch.manual_seed(self.config.initializer_seed + 200)
                nn.init.normal_(weight[256 + 4 * self.config.lexical_size:], std=self.config.initializer_range)
                torch.manual_seed(self.config.initializer_seed + 300)
                nn.init.normal_(self.tokenizer_embeddings.weight, std=self.config.initializer_range)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.gpt_neox.embed_in

    def set_input_embeddings(self, value):
        self.gpt_neox.embed_in = value

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, value):
        self.embed_out = value

    def _routes(self, tokenizer_ids, batch_size, device):
        if tokenizer_ids is None:
            raise ValueError("tokenizer_ids is required; choose byte/zh/ko/ja/en explicitly")
        if isinstance(tokenizer_ids, str):
            tokenizer_ids = route_index(tokenizer_ids)
        routes = torch.as_tensor(tokenizer_ids, device=device)
        if routes.dtype == torch.bool or routes.is_floating_point():
            raise ValueError("tokenizer_ids must contain integer route IDs")
        routes = routes.to(torch.long)
        if routes.ndim == 0:
            routes = routes.expand(batch_size)
        if routes.shape != (batch_size,):
            raise ValueError(f"tokenizer_ids must be scalar or shape [{batch_size}], got {tuple(routes.shape)}")
        if torch.any((routes < 0) | (routes > 4)):
            raise ValueError("tokenizer_ids must be in 0..4")
        return routes

    def _validate_input_ids(self, input_ids, routes):
        if torch.any((input_ids < 0) | (input_ids >= self.config.vocab_size)):
            raise ValueError("input_ids outside the global vocabulary")
        allowed = self.allowed_output_mask[routes].gather(1, input_ids)
        controls = 256 + 4 * self.config.lexical_size + 3 * routes[:, None]
        allowed = allowed | (input_ids == controls) | (input_ids == controls + 2)
        if not bool(allowed.all()):
            raise ValueError("Input contains a private token/control belonging to another route")

    def forward(
        self,
        input_ids=None,
        tokenizer_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        return_logits=True,
        **kwargs,
    ):
        # Trainer may pass num_items_in_batch; route balancing intentionally
        # uses a separate target count per route rather than a pooled count.
        kwargs.pop("num_items_in_batch", None)
        if kwargs:
            raise TypeError(f"Unexpected model kwargs: {sorted(kwargs)}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        routes = self._routes(tokenizer_ids, source.shape[0], source.device)
        if input_ids is not None:
            self._validate_input_ids(input_ids, routes)
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if self.config.use_tokenizer_embeddings:
            inputs_embeds = inputs_embeds + self.tokenizer_embeddings(routes)[:, None, :]
        outputs = self.gpt_neox(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state
        logits = None
        if return_logits:
            logits = self.embed_out(hidden).float()
            logits = logits.masked_fill(~self.allowed_output_mask[routes, None, :], float("-inf"))

        loss = route_losses = counts = None
        if labels is not None:
            if labels.shape != hidden.shape[:2]:
                raise ValueError("labels must match the current input shape, before the causal shift")
            labels = labels.to(hidden.device)
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask[:, -labels.shape[1]:] == 0, -100)
            target = labels[:, 1:].contiguous()
            counts = torch.zeros(5, dtype=torch.long, device=hidden.device)
            route_losses = torch.full((5,), float("nan"), device=hidden.device)
            losses = []
            for route in range(5):
                rows = routes == route
                route_target = target[rows]
                valid = route_target != -100
                count = valid.sum()
                counts[route] = count
                if not count:
                    continue
                selected_target = route_target[valid]
                if torch.any((selected_target < 0) | (selected_target >= self.config.vocab_size)):
                    raise ValueError("A label is outside the global vocabulary")
                local_target = self.global_to_local[route, selected_target]
                if torch.any(local_target < 0):
                    raise ValueError(f"Label is not predictable under tokenizer route {route}")
                if logits is not None:
                    selected_logits = logits[rows, :-1][valid]
                    route_loss = F.cross_entropy(selected_logits, selected_target)
                else:
                    allowed = self.allowed_output_mask[route].nonzero(as_tuple=True)[0]
                    selected_hidden = hidden[rows, :-1][valid]
                    weight = self.get_output_embeddings().weight.index_select(0, allowed)
                    selected_logits = F.linear(selected_hidden, weight).float()
                    route_loss = F.cross_entropy(selected_logits, local_target)
                losses.append(route_loss)
                route_losses[route] = route_loss.detach()
            if not losses:
                raise ValueError("Batch has no valid next-token targets after the causal shift")
            loss = torch.stack(losses).mean()

        if return_dict is False or (return_dict is None and not self.config.use_return_dict):
            tail = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + tail) if loss is not None else tail
        return MixtureCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            route_losses=route_losses,
            route_target_counts=counts,
        )

    def prepare_inputs_for_generation(self, input_ids, tokenizer_ids=None, **kwargs):
        model_inputs = GPTNeoXForCausalLM.prepare_inputs_for_generation(self, input_ids, **kwargs)
        model_inputs["tokenizer_ids"] = tokenizer_ids
        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        source = inputs if inputs is not None else kwargs.get("input_ids")
        if source is None:
            raise ValueError("generate requires an explicit input_ids prompt (at least route BOS)")
        routes = self._routes(kwargs.get("tokenizer_ids"), source.shape[0], source.device)
        if not bool((routes == routes[0]).all()):
            raise ValueError("Generation batches must use one route; group requests by tokenizer")
        route = int(routes[0])
        kwargs["tokenizer_ids"] = routes
        for name, value in {
            "bos_token_id": self.config.bos_id(route),
            "eos_token_id": self.config.eos_id(route),
            "pad_token_id": self.config.pad_id(route),
        }.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"{name} must equal {value} for route {route}")
            kwargs[name] = value
        return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)


AutoModelForCausalLM.register(MixtureConfig, MixtureForCausalLM, exist_ok=True)

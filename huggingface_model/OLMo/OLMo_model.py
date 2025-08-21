# OLMo_custom_attention.py
import torch
import math
import torch.nn.functional as F
from transformers.models.olmo.configuration_olmo import OlmoConfig
from transformers.models.olmo.modeling_olmo import OlmoAttention, repeat_kv, apply_rotary_pos_emb

class OlmoCustomAttention(OlmoAttention):
    """
    Custom attention as determined by the attention_variant parameter.
    """
    def __init__(self, config, max_steps, layer_idx, clip, gate, obo_variant, attention_variant, zeta=1.0, gamma=0.0):
        super().__init__(config, layer_idx=layer_idx)
        if attention_variant not in ["softmax", "relu", "relu_norm", "softplus", "softplus_norm", "sigmoid", "sigmoid_norm", 
                                     "gradual_relu", "gradual_softplus", "gradual_sigmoid"]:
            raise ValueError("attention_variant isn't one of the valid possible options")
        self.attention_variant = attention_variant
        self.clip = clip
        self.gate = gate
        self.obo_variant = obo_variant
        self.zeta = zeta
        self.gamma = gamma
        self.max_steps = max_steps
        self.num_forward = 0
        self.curr_steps = 0
        self.num_heads = config.num_attention_heads
        # Initialize gating parameters if gated attention is chosen.
        if self.gate:
            # Each head gets its own linear transformation (d_head -> 1)
            head_dim = config.hidden_size // config.num_attention_heads
            self.gate_weight = torch.nn.Parameter(torch.randn(config.num_attention_heads, head_dim) * (1.0 / math.sqrt(head_dim)))
            self.gate_bias = torch.nn.Parameter(torch.zeros(config.num_attention_heads, 1))
        if self.obo_variant == "learned_obo_per_layer":
            self.obo_param = torch.nn.Parameter(torch.tensor(0.0))
        elif self.obo_variant == "learned_obo_per_layer_per_head":
            self.obo_param = torch.nn.Parameter(torch.zeros(self.num_heads, 1))

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
        if self.training:
            self.num_forward += 1
            if (self.num_forward - 1) % 4 == 0:
                self.curr_steps += 1
        transition_percentage = self.curr_steps / self.max_steps
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project hidden states to query, key, and value.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Apply rotary positional embeddings.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Expand key and value for key/value groups.
        key_states_exp = repeat_kv(key_states, self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)
        attn_scores = torch.matmul(query_states, key_states_exp.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states_exp.shape[-2]]
            attn_scores = attn_scores + causal_mask

        if self.obo_variant != "none" and not (self.attention_variant == "softmax" or self.attention_variant.endswith("norm") or self.attention_variant.startswith("gradual")):
            raise ValueError("Can't perform Off-By-One on non-normalized attention variant")
        if self.attention_variant == "relu":
            # ReLU-based attention
            attn_weights = F.relu(attn_scores)
        elif self.attention_variant == "relu_norm":
            attn_weights = F.relu(attn_scores)
            # Normalize the weights so they sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            if self.obo_variant == "obo":
                attn_weights = attn_weights / (1.0 + attn_sum)
            elif self.obo_variant in ["learned_obo_per_layer", "learned_obo_per_layer_per_head"]:
                # For per-head, adjust dimensions for broadcasting.
                if self.obo_variant == "learned_obo_per_layer_per_head":
                    obo_param = self.obo_param.unsqueeze(0).unsqueeze(2)  # shape becomes (1, num_heads, 1, 1)
                else:
                    obo_param = self.obo_param
                attn_weights = attn_weights / (obo_param + attn_sum) # attn_sum is (B, H, T, 1)
            else:
                attn_weights = attn_weights / attn_sum
        elif self.attention_variant.startswith("gradual"):
            if self.obo_variant == "obo":
                exp_scores = torch.exp(attn_scores)
                denom = 1.0 + exp_scores.sum(dim=-1, keepdim=True)
                softmax_weights = exp_scores / denom
            elif self.obo_variant in ["learned_obo_per_layer", "learned_obo_per_layer_per_head"]:
                exp_scores = torch.exp(attn_scores)
                attn_sum = exp_scores.sum(dim=-1, keepdim=True)
                if self.obo_variant == "learned_obo_per_layer_per_head":
                    obo_param = self.obo_param.unsqueeze(0).unsqueeze(2)
                else:
                    obo_param = self.obo_param
                denom = obo_param + attn_sum
                softmax_weights = exp_scores / denom
            else:
                softmax_weights = F.softmax(attn_scores, dim=-1)
            if self.attention_variant == "gradual_relu":
                variant_weights = F.relu(attn_scores)
            elif self.attention_variant == "gradual_softplus":
                variant_weights = F.softplus(attn_scores)
            elif self.attention_variant == "gradual_sigmoid":
                variant_weights = torch.sigmoid(attn_scores)
            if self.training:
                attn_weights = (1 - transition_percentage) * softmax_weights + transition_percentage * variant_weights
            else:
                attn_weights = variant_weights
        elif self.attention_variant == "softplus":
            attn_weights = F.softplus(attn_scores)
        elif self.attention_variant == "softplus_norm":
            attn_weights = F.softplus(attn_scores)
            # Normalize so weights sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            if self.obo_variant == "obo":
                attn_weights = attn_weights / (1.0 + attn_sum)
            elif self.obo_variant in ["learned_obo_per_layer", "learned_obo_per_layer_per_head"]:
                # For per-head, adjust dimensions for broadcasting.
                if self.obo_variant == "learned_obo_per_layer_per_head":
                    obo_param = self.obo_param.unsqueeze(0).unsqueeze(2)  # shape becomes (1, num_heads, 1, 1)
                else:
                    obo_param = self.obo_param
                attn_weights = attn_weights / (obo_param + attn_sum) # attn_sum is (B, H, T, 1)
            else:
                attn_weights = attn_weights / attn_sum
        elif self.attention_variant == "sigmoid":
            attn_weights = torch.sigmoid(attn_scores)
        elif self.attention_variant == "sigmoid_norm":
            attn_weights = torch.sigmoid(attn_scores)
            # Normalize so weights sum to 1 along the key dimension.
            attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-9
            if self.obo_variant == "obo":
                attn_weights = attn_weights / (1.0 + attn_sum)
            elif self.obo_variant in ["learned_obo_per_layer", "learned_obo_per_layer_per_head"]:
                # For per-head, adjust dimensions for broadcasting.
                if self.obo_variant == "learned_obo_per_layer_per_head":
                    obo_param = self.obo_param.unsqueeze(0).unsqueeze(2)  # shape becomes (1, num_heads, 1, 1)
                else:
                    obo_param = self.obo_param
                attn_weights = attn_weights / (obo_param + attn_sum) # attn_sum is (B, H, T, 1)
            else:
                attn_weights = attn_weights / attn_sum
        elif self.attention_variant == "softmax":
            if self.obo_variant == "obo":
                exp_scores = torch.exp(attn_scores)
                denom = 1.0 + exp_scores.sum(dim=-1, keepdim=True)
                attn_weights = exp_scores / denom
            elif self.obo_variant in ["learned_obo_per_layer", "learned_obo_per_layer_per_head"]:
                exp_scores = torch.exp(attn_scores)
                attn_sum = exp_scores.sum(dim=-1, keepdim=True)
                # For per-head, adjust dimensions for broadcasting.
                if self.obo_variant == "learned_obo_per_layer_per_head":
                    obo_param = self.obo_param.unsqueeze(0).unsqueeze(2)  # shape becomes (1, num_heads, 1, 1)
                else:
                    obo_param = self.obo_param
                denom = obo_param + attn_sum
                attn_weights = exp_scores / denom
            else:
                attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            raise ValueError("attention_variant isn't one of the valid possible options")
        if self.clip:
            # apply clipping
            attn_weights = torch.clamp((self.zeta - self.gamma) * attn_weights + self.gamma, min=0.0, max=1.0)

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_exp)

        # If using gated attention, compute gating values and apply them.
        if self.gate:
            # Compute gate scores per head and token:
            # query_states has shape (B, nheads, T, d_head)
            gate_scores = torch.einsum('bhtd,hd->bht', query_states, self.gate_weight) + self.gate_bias.view(1, self.num_heads, 1)
            gate_values = torch.sigmoid(gate_scores)  # shape: (B, nheads, T)
            gate_values = gate_values.unsqueeze(-1)   # shape: (B, nheads, T, 1)
            attn_output = attn_output * gate_values

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def apply_custom_attention(model, args, clip, gate, obo_variant, attention_variant, zeta=1.0, gamma=0.0):
    """
    Replaces all decoder layer attention modules in the model with OlmoCustomAttention.
    """
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        original_attn = decoder_layer.self_attn
        custom_attn = OlmoCustomAttention(
            model.config,
            max_steps=args.max_steps,
            layer_idx=layer_idx,
            clip=clip,
            gate=gate,
            obo_variant=obo_variant,
            attention_variant=attention_variant,
            zeta=zeta,
            gamma=gamma
        )
        custom_attn.load_state_dict(original_attn.state_dict(), strict=False)
        decoder_layer.self_attn = custom_attn
    return model

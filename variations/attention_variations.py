# variations/attention_variations.py

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from quantization.quant_utils import create_activation_buffers, set_variant
from quantization.quantize import fake_quantize_act
from variations.linear_variations import linear_dictionary
from variations.position_encoding_variations import (
    FIRE, RotaryEmbedding, SymmetricalOverlapAngularPositions)
from variations.softmax_variations import softmax_dictionary
from variations.triadic_modulation_variations import mod_fn_dict
# Mamba related imports
# if torch.cuda.is_available():
#     from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class CausalSelfAttention(nn.Module):
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval
        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        if (config.n_kv_group is None):
            config.n_kv_group = config.n_head
        else:
            assert config.n_embd % config.n_kv_group == 0

        self.quantization_attn_dict = {}
        self.quantization_attn_dict["activations_quant_method"] = config.activations_quant_method
        for arg, val in vars(config).items():
            # Set each attention Activation precision and method
            if arg.startswith("quantize_") and "attn_act" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act_bits)
            elif arg.startswith("quantize_") and "attn_act" in arg:
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_attn_act)
                if config.store_activations and arg != "quantize_attn_act" and self.quantization_attn_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set each attention Linear precision and method
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_bits"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_attn" in arg and arg.endswith("_method"):
                self.quantization_attn_dict[arg] = set_variant(val, config.quantize_linear_method)

        self.linear_variant_q = linear_dictionary[set_variant(config.linear_variant_q, config.linear_variant_attn)]
        self.linear_variant_k = linear_dictionary[set_variant(config.linear_variant_k, config.linear_variant_attn)]
        self.linear_variant_v = linear_dictionary[set_variant(config.linear_variant_v, config.linear_variant_attn)]
        self.linear_variant_attn_proj = linear_dictionary[set_variant(config.linear_variant_attn_proj, config.linear_variant_attn)]

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = self.linear_variant_q(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_q_method"], self.quantization_attn_dict["quantize_linear_attn_q_bits"], bias=config.bias)

        self.n_head = config.n_head
        if config.n_kv_group is None:
            self.n_kv_group = config.n_head
        else:
            assert config.n_head % config.n_kv_group == 0
            self.n_kv_group = config.n_kv_group

        self.kv_dim = (config.n_embd // config.n_head) * self.n_kv_group
        self.c_attn_k = self.linear_variant_k(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_k_method"], self.quantization_attn_dict["quantize_linear_attn_k_bits"], bias=config.bias)
        self.c_attn_v = self.linear_variant_v(config.n_embd, self.kv_dim, config, self.quantization_attn_dict["quantize_linear_attn_v_method"], self.quantization_attn_dict["quantize_linear_attn_v_bits"], bias=config.bias)
        self.c_proj = self.linear_variant_attn_proj(config.n_embd, config.n_embd, config, self.quantization_attn_dict["quantize_linear_attn_proj_method"], self.quantization_attn_dict["quantize_linear_attn_proj_bits"], bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Embedding
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_embd = config.n_embd
        self.gate = config.gate
        self.use_fire_embeddings = None
        self.disable_flash_attention = config.disable_flash_attention
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)

        # Sliding window size
        self.window_size = config.window_size
        print(f"sliding window size: {self.window_size}")

        # qk_norm
        self.use_qk_norm = config.use_qk_norm
        self.use_qk_norm_scale = config.use_qk_norm_scale

        # Flash Lobo
        self.use_flash_lobo = config.use_flash_lobo
        self.use_flash_lobo_per_head = config.use_flash_lobo_per_head
        if self.use_flash_lobo:
            if config.use_flash_obo_const:
                self.flash_lobo_log_const = config.flash_lobo_log_const # log C (0 -> C = 1)
            elif config.use_flash_lobo_per_head:
                # learnable parameter, one scalar per head
                self.flash_lobo_log_const = nn.Parameter(
                    torch.full( (self.n_head,), config.flash_lobo_log_const)
                )
            else:
                self.flash_lobo_log_const = nn.Parameter(torch.tensor(config.flash_lobo_log_const))  # log C  (0 → C = 1)

        # Using flex attention
        self.use_flex_attn = config.use_flex_attn

        # Gating
        self.gate = config.gate

        # Fire Embeddings
        self.use_fire_embeddings = None
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(config, num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # Note: size is the size of the head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=config.n_embd // self.n_head, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd // self.n_head)
                self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // self.n_head)

        # qk norm factor
        if self.use_qk_norm_scale:
            L = config.block_size
            g0 = math.log2(L*L - L)
            self.qk_norm_factor = nn.Parameter(torch.tensor(g0))

        self.flash = True
        if self.window_size is not None:
            # TODO: look into supporting sliding window attn for flash attn
            self.flash = False
            print("flash attention removed due to windowed attention")

        if self.n_kv_group != self.n_head:
            self.flash = False
            print("flash attention removed due to GQA")

        if self.use_fire_embeddings:
            self.flash = False
            print("flash attention removed due to FIRE")

        # Can't use flash attention if we want to manually quantize most input/output activations in attn
        for key, val in self.quantization_attn_dict.items():
            if key.startswith("quantize_") and val == True:
                self.flash = False
                print("flash attention removed due to Quantization")
                break

        if self.attn_logit_softcapping:
            self.flash = False
            print("flash attention removed due to attn logit softcapping")

        if self.disable_flash_attention:
            self.flash = False

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn == "softmax":
            # Enable flash attention, which is compatible with 'softmax'
            if self.disable_flash_attention or self.flash == False:
                print("setting non-flash softmax attn")
            else:
                self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                print("setting flash attn")
        else:
            # Remove flash attention (only compatible with 'softmax')
            print("flash attention removed due to softmax alternative")
            self.flash = False
            # Set softmax_layer_attn to custom softmax alternative
            self.softmax_layer_attn = softmax_dictionary[config.softmax_variant_attn](config)

        if (not self.flash) and (not self.use_flex_attn):
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    # Flex Attention Related
    def sliding_window_causal(self, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= self.window_size
        return causal_mask & window_mask

    def get_block_mask(self, T, device):
        if T not in self.block_masks:
            block_mask = create_block_mask(
                    self.sliding_window_causal,
                    B=None,
                    H=None,
                    Q_LEN=T,
                    KV_LEN=T,
                    device=device
                    )
            self.block_masks[T] = block_mask
        else:
            block_mask = self.block_masks[T]
        return block_mask
    # End Flex Attention Related

    def forward(self, x, iter_num):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if self.quantization_attn_dict["quantize_attn_act_input"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_input_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            x = fake_quantize_act(self, "attn_act_input", x, num_bits, quant_method, iter_num)

        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        if self.window_size is not None:
            if self.use_flex_attn is not None:
                self.block_masks = {}
            else:
                self.window_mask = torch.ones((1, 1, T, T), device=x.device)
                self.window_mask = torch.triu(self.window_mask, diagonal=-self.window_size)
                self.window_mask = self.bias[:,:,:T,:T] * self.window_mask

        if self.gate:
            if self.n_kv_group == self.n_head:
                Gating = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                gate_ = torch.sigmoid(Gating(x))
                q = q * gate_
                k = k * gate_
                v = v * gate_
            else:
                # TODO: Test more methods to merge Attention Gates with GQA
                # TODO: Evaluate each method's ability to even out parameter sizes
                Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)
                gate_qx = Gating_q(x)
                gate_q = torch.sigmoid(gate_qx)
                gate_kv = torch.sigmoid(Gating_kv(gate_qx))
                q = q * gate_q
                k = k * gate_kv
                v = v * gate_kv

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_h, T, hs)
        k = k.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)

        # rotate q and k before evaluating with the heads
        if (self.rotary_emb_q is not None) and (self.rotary_emb_k is not None):
            q = self.rotary_emb_q(q)
            k = self.rotary_emb_k(k)

        y = None

        if self.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:

            # Flash QK Norm
            if self.use_qk_norm_scale:
                # pre-scale Q so that built-in √dₕ division becomes our g scaling
                head_dim = math.sqrt(k.size(-1))
                qk_scaling_factor = self.qk_norm_factor * math.sqrt(head_dim)
                q = q * qk_scaling_factor

            # Flash Lobo
            attn_bias = None
            if self.use_flash_lobo:
                # 2-a  Make dummy key/value column of zeros
                dummy_k = q.new_zeros(B, self.n_kv_group, 1, q.size(-1))
                dummy_v = q.new_zeros(B, self.n_kv_group, 1, v.size(-1))

                k = torch.cat([dummy_k, k], dim=2)   # prepend → causal mask still valid
                v = torch.cat([dummy_v, v], dim=2)

                # 2-b  Bias only that column with log C
                attn_bias = q.new_zeros(1, self.n_head, 1, k.size(2))
                if self.use_flash_lobo_per_head:
                    attn_bias[..., 0] = self.flash_lobo_log_const.view(1, self.n_head, 1)
                else:
                    attn_bias[..., 0] = self.flash_lobo_log_const    # first column only


            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        elif self.use_flex_attn and self.window_size is not None:
            block_mask = self.get_block_mask(T, x.device)
            y = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=block_mask)
        else:
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_q_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                q = fake_quantize_act(self, "attn_act_qk_mult_q_input", q, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_qk_mult_k_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                k = fake_quantize_act(self, "attn_act_qk_mult_k_input", k, num_bits, quant_method, iter_num)

            att = None
            # manual implementation of attention
            head_dim = math.sqrt(k.size(-1))
            if self.n_head != self.n_kv_group:
                k_repeated = k.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                att = (q @ k_repeated.transpose(-2, -1))
            else:
                att = (q @ k.transpose(-2, -1))

            if self.use_qk_norm_scale:
                att = att * self.qk_norm_factor
            else:
                att = att / head_dim

            # apply logit softcapping after qk but before masking
            if self.attn_logit_softcapping is not None:
                att = att / self.attn_logit_softcapping
                att = torch.tanh(att)
                att = att * self.attn_logit_softcapping

            # apply masks
            if self.window_size is not None:
                # add mask for sliding window attention
                att = att.masked_fill(self.window_mask == 0, float('-inf'))
            else:
                # regular lower triangle attention
                att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))

            # fire position embeddings
            if self.use_fire_embeddings is not None:
                # add learned fire bias
                att = att + self.fire_pos_enc(x)

            if self.quantization_attn_dict["quantize_attn_act_softmax_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_softmax_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_softmax_input", att, num_bits, quant_method, iter_num, causal_mask=True)

            # softmax variation
            if self.softmax_variant_attn != 'softmax':
                att = self.softmax_layer_attn(att)
            else:
                att = F.softmax(att, dim=-1)

            att = self.attn_dropout(att)

            if self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_p_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                att = fake_quantize_act(self, "attn_act_pv_mult_p_input", att, num_bits, quant_method, iter_num)
            if self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input"]:
                num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_v_input_bits"]
                quant_method = self.quantization_attn_dict["activations_quant_method"]
                v = fake_quantize_act(self, "attn_act_pv_mult_v_input", v, num_bits, quant_method, iter_num)

            if self.n_head != self.n_kv_group:
                v_repeated = v.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                y = att @ v_repeated # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            else:
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        if self.quantization_attn_dict["quantize_attn_act_pv_mult_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_pv_mult_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_pv_mult_output", y, num_bits, quant_method, iter_num)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if self.quantization_attn_dict["quantize_attn_act_output"]:
            num_bits = self.quantization_attn_dict["quantize_attn_act_output_bits"]
            quant_method = self.quantization_attn_dict["activations_quant_method"]
            y = fake_quantize_act(self, "attn_act_output", y, num_bits, quant_method, iter_num)

        return y

class LinearAttention(nn.Module):
    """ Implements Linear Attention as described in:
    Katharopoulos, A., et al. (2020). Transformers are RNNs:
    Fast Autoregressive Transformers with Linear Attention. ICML.
    https://arxiv.org/abs/2006.16236

    This class replaces the standard softmax attention with a
    kernel-based linear attention mechanism, enabling linear
    time and space complexity with respect to sequence length.
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Combined linear layer for q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.scale = torch.nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_size)))


    def forward(self, x, iter_num=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)

        # NEW: Scale BEFORE the feature map
        q = q * self.scale
        k = k * self.scale

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        kv = k * v
        k_cumsum = k.cumsum(dim=2)
        kv_cumsum = kv.cumsum(dim=2)


        eps = 1e-3  # Increased epsilon
        y = torch.einsum("BHTD,BHTD->BHTD", q, kv_cumsum) / (torch.einsum("BHTD,BHTD->BHT", q, k_cumsum)[..., None].clamp(min=eps))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

# class HymbaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         HymbaRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

# class MambaBlock(nn.Module):
#     """ This function contains code adapted from [Hymba](https://github.com/NVlabs/hymba/)
#     by the NVIDIA team, licensed under the [NVIDIA Open Model License Agreement]
#     (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
#     """
#     def __init__(self, config, fire_pos_enc=None):
#         super().__init__()

#         self.d_model = config.n_embd
#         self.d_inner = int(self.d_model * config.ssm_mamba_expand)
#         self.conv_kernel_size = config.ssm_conv_kernel_size
#         self.dt_rank = config.ssm_dt_rank
#         self.d_state = config.ssm_d_state
#         self.io_bias = config.ssm_io_bias

#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             bias=True,
#             kernel_size=self.conv_kernel_size,
#             groups=self.d_inner,
#             padding=self.conv_kernel_size - 1
#         )

#         num_ssm_param = 1
#         self.in_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_inner * 2, bias=self.io_bias)])
#         self.x_proj = nn.ModuleList([nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(num_ssm_param)])
#         self.dt_proj = nn.ModuleList([nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(num_ssm_param)])
#         self.out_proj = nn.ModuleList([nn.Linear(self.d_inner, self.d_model, bias=self.io_bias)])

#         A = torch.arange(1, self.d_state + 1, dtype=torch.float32)[None, :]
#         A = A.expand(self.d_inner, -1).contiguous()
#         self.A_log = nn.ParameterList([nn.Parameter(torch.log(A)) for _ in range(num_ssm_param)])

#         self.D = nn.ParameterList([nn.Parameter(torch.ones(self.d_inner)) for _ in range(num_ssm_param)])

#         self.dt_layernorm = HymbaRMSNorm(self.dt_rank, eps=1e-06)
#         self.B_layernorm = HymbaRMSNorm(self.d_state, eps=1e-06)
#         self.C_layernorm = HymbaRMSNorm(self.d_state, eps=1e-06)
#         self.scan_outputs_layernorm = HymbaRMSNorm(self.d_inner)

#     def _apply_layernorms(self, dt, B, C):
#         if self.dt_layernorm is not None:
#             dt = self.dt_layernorm(dt)
#         if self.B_layernorm is not None:
#             B = self.B_layernorm(B)
#         if self.C_layernorm is not None:
#             C = self.C_layernorm(C)
#         return dt, B, C

#     def forward(self, x, gate, iter_num=None):
#         '''
#         Parameters:
#             x: (batch_size, seqlen, d_model)
#         Return:
#             scan_outputs: (batch_size, seqlen, d_model)

#         # d_model == n_embd (in attention)
#         # d_inner == d_model * mamba_expand

#         # conv1d.weight: (d_inner, 1, conv_kernel_size)
#         # conv_weights: (d_inner, conv_kernel_size)
#         # hidden_states: (batch_size, d_inner, seqlen)
#         # gate: (batch_size, d_inner, seqlen)
#         # delta: (batch_size, seqlen, dt_rank)
#         # discrete_delta: (batch, d_inner, seqlen)
#         # A: (d_inner, d_state)
#         # B: (batch_size, seqlen, d_state) before transpose(1,2)
#         # C: (batch_size, seqlen, d_state) before transpose(1,2)
#         '''
#         # we only have a single mamba head at this point
#         index = 0

#         projected_states = self.in_proj[index](x).transpose(1,2)

#         hidden_states, gate = projected_states.tensor_split((self.d_inner,), dim=1)

#         conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
#         hidden_states = causal_conv1d_fn(
#             hidden_states, conv_weights, self.conv1d.bias, activation="silu"
#         )

#         ssm_parameters = self.x_proj[index](hidden_states.transpose(1, 2))
#         delta, B, C = torch.split(ssm_parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         delta, B, C = self._apply_layernorms(delta, B, C)
#         dt_proj_bias = self.dt_proj[index].bias
#         self.dt_proj[index].bias = None
#         discrete_delta = self.dt_proj[index](delta).transpose(1, 2)  # (batch_size, d_inner, seqlen)
#         self.dt_proj[index].bias = dt_proj_bias

#         A = -torch.exp(self.A_log[index].float())

#         dt_proj_bias = dt_proj_bias.float() if dt_proj_bias is not None else None

#         # mammba kernel from mamba_ssm
#         outputs = selective_scan_fn(
#             hidden_states,                          # (batch_size, d_inner, seqlen)
#             discrete_delta,
#             A,
#             B.transpose(1, 2).to(torch.float16),    # torch.float32 -> torch.float16 for selective_scan_fn
#             C.transpose(1, 2).to(torch.float16),    # torch.float32 -> torch.float16 for selective_scan_fn
#             self.D[index].float(),
#             z=gate,
#             delta_bias=dt_proj_bias,
#             delta_softplus=True,
#             return_last_state=True,
#         )                                           # (batch_size, d_inner, seqlen)

#         if len(outputs) == 3:
#             scan_outputs, _, _ = outputs            # scan_outputs, ssm_state, last_state
#                                                     # ssm_state are updated inplace
#         else:
#             scan_outputs, _ = outputs

#         scan_outputs = scan_outputs.transpose(1, 2)  # (batch_size, seqlen, d_inner)
#         scan_outputs = self.scan_outputs_layernorm(scan_outputs)

#         output = self.out_proj[index](scan_outputs)
#         return output


class AttnIdentity(nn.Identity):
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

    def forward(self, x, iter_num=None):
        x = super().forward(x)
        return x

class InfiniteHeadAttention(nn.Module):
    """Instead of concatenating heads, utilizing higher capacity, we assume the
    vector features are independent of each other, and simply add the values.

    This removes the constraint of having number_of_heads % embed_dim = 0, resulting in:
      * a) removes the limit on the number of heads (before increasing heads too much leads to reduced emb_dim per head, and head utility)
      * b) from a), this means we can keep adding heads until the model saturates the vector.
      * c) while all heads need to be the same size, we have new param exploration, number of heads and the dimension per head.
      * d) we can potentially even try removing the c_proj, if the embedding dimension chosen is the same as that of the model
      * e) since the MLP/MoE has the majority of parameters, this may benefit
             parameter efficiency by allowing more relations to be encoded into the
             residual per attention layer.
      * f) for smaller models, we can increase the embedding dim per head, to
             match that of high quality 1024 and higher embedding heads, which has
             been noted to be a bottleneck when digesting large trees of information
             in a single layer e.g. multidigit addition.
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        self.n_head = config.n_head

        # group query attention and fallback
        if (config.n_kv_group is None):
            self.n_kv_group = config.n_head
        else:
            assert config.n_embd % config.n_kv_group == 0
            self.n_kv_group = config.n_kv_group

        self.n_embd = config.n_embd
        self.n_qk_head_dim = config.n_qk_head_dim
        self.n_v_head_dim = config.n_v_head_dim


        # Concat Heads
        self.use_concat_heads = config.use_concat_heads
        self.n_cproj         = config.n_cproj

        # QK Norm
        self.use_qk_norm        = config.use_qk_norm
        self.use_qk_norm_scale  = config.use_qk_norm_scale

        # Flash Lobo
        self.use_flash_lobo          = config.use_flash_lobo
        self.use_flash_lobo_per_head = config.use_flash_lobo_per_head
        self.use_flash_obo_const     = config.use_flash_obo_const

        # Set Flash Lobo_const
        if self.use_flash_lobo:
            if self.use_flash_obo_const:
                # constant of this value for all heads in this layer
                self.flash_lobo_log_const = config.flash_lobo_log_const # log C (0 -> C = 1)
            elif self.use_flash_lobo_per_head:
                # learnable parameter, one scalar per head
                self.flash_lobo_log_const = nn.Parameter(
                    torch.full( (self.n_head,), config.flash_lobo_log_const)
                )
            else:
                # single learnable parameter per layer
                self.flash_lobo_log_const = nn.Parameter(torch.tensor(config.flash_lobo_log_const))  # log C  (0 → C = 1)

        # Set nn.Linear Types for Wk, Wq, Wv and c_proj
        self.linear_variant_q = linear_dictionary[config.linear_variant_attn]
        self.linear_variant_k = linear_dictionary[config.linear_variant_attn]
        self.linear_variant_v = linear_dictionary[config.linear_variant_attn]
        self.linear_variant_attn_proj = linear_dictionary[config.linear_variant_attn]

        # TODO: no reason for qk and v to have same dimension
        self.c_attn_q = self.linear_variant_q(self.n_embd, self.n_head * self.n_qk_head_dim, config, bias=config.bias)
        self.c_attn_k = self.linear_variant_k(self.n_embd, self.n_kv_group * self.n_qk_head_dim, config, bias=config.bias)
        self.c_attn_v = self.linear_variant_v(self.n_embd, self.n_kv_group * self.n_v_head_dim, config, bias=config.bias)

        if self.use_concat_heads:
            print("use_concat_heads")
            # Usually c_proj dim are (n_head * n_head dim = n_embd), here we have to provide the factorized version
            self.c_proj = self.linear_variant_attn_proj(
                self.n_head * self.n_v_head_dim, self.n_embd, config, bias=config.bias
            )
        elif self.n_cproj==1:
            print("use n_cproj 1", self.n_v_head_dim, self.n_embd)
            self.c_proj = self.linear_variant_attn_proj(self.n_v_head_dim, self.n_embd, config, bias=config.bias)
        else:
            print("use_cproj_list")
            self.c_proj_list = nn.ModuleList(
                [
                    self.linear_variant_attn_proj(self.n_v_head_dim, self.n_embd, config, bias=config.bias)
                    for _ in range(self.n_cproj)
                ]
            )

        # option to turn off flash attention
        self.disable_flash_attention = config.disable_flash_attention

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Embedding
        self.n_embd = config.n_embd

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None

        if config.use_rotary_embeddings:
            # Note: "size" here is the size of the qk head dimension
            if config.rope_variant == "soap":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=self.n_qk_head_dim, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=self.n_qk_head_dim, num_angles=self.sym_rot_num_angles)
            elif config.rope_variant == "rope":
                self.rotary_emb_q = RotaryEmbedding(config, size=self.n_qk_head_dim)
                self.rotary_emb_k = RotaryEmbedding(config, size=self.n_qk_head_dim)

        # qk norm factor
        if self.use_qk_norm_scale:
            L = config.block_size
            g0 = math.log2(L*L - L)
            self.qk_norm_factor = nn.Parameter(torch.tensor(g0))

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn != 'softmax':
            self.softmax_layer_attn = softmax_dictionary[config.softmax_variant_attn](config)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, iter_num):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        q = q.view(B, T, self.n_head, self.n_qk_head_dim).transpose(1, 2) # (B, n_h, T, hs)
        k = k.view(B, T, self.n_kv_group, self.n_qk_head_dim).transpose(1, 2) # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_group, self.n_v_head_dim).transpose(1, 2) # (B, n_kv, T, hs)

        # Apply Rotary Position Encodings
        if (self.rotary_emb_q is not None) and (self.rotary_emb_k is not None):
            q = self.rotary_emb_q(q)
            k = self.rotary_emb_k(k)

        # Apply QK Norm
        if self.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        y = None
        att = None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if not self.disable_flash_attention:
            # Flash QK Norm
            if self.use_qk_norm_scale:
                # pre-scale Q so that built-in √dₕ division becomes our g scaling
                sqrt_head_dim = math.sqrt(k.size(-1))
                qk_scaling_factor = self.qk_norm_factor * sqrt_head_dim
                q = q * qk_scaling_factor

            # GQA
            if self.n_kv_group != self.n_head:
                repeat = self.n_head // self.n_kv_group

                # (B, n_head, T, d) -> (B, T, n_head, d) -> contiguous
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                # view with n_kv_group then repeat-interleave to n_head
                k = (
                    k.view(B, T, self.n_kv_group, self.n_qk_head_dim)
                    .repeat_interleave(repeat, dim=2)
                    .transpose(1, 2)
                )  # (B, n_head, T, d)
                v = (
                    v.view(B, T, self.n_kv_group, self.n_v_head_dim)
                    .repeat_interleave(repeat, dim=2)
                    .transpose(1, 2)
                )  # (B, n_head, T, d)

            # Flash Lobo
            attn_bias = None
            if self.use_flash_lobo:
                # 2-a  Make dummy key/value column of zeros
                dummy_k = q.new_zeros(B, k.size(1), 1, q.size(-1))
                dummy_v = q.new_zeros(B, v.size(1), 1, v.size(-1))

                k = torch.cat([dummy_k, k], dim=2)   # prepend → causal mask still valid
                v = torch.cat([dummy_v, v], dim=2)

                # 2-b  Bias only that column with log C
                attn_bias = q.new_zeros(1, self.n_head, 1, k.size(2))
                if self.use_flash_lobo_per_head:
                    attn_bias[..., 0] = self.flash_lobo_log_const.view(1, self.n_head, 1)
                else:
                    attn_bias[..., 0] = self.flash_lobo_log_const    # first column only


            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        else:
            # Manual implementation of attention

            # ----- GQA support (repeat k & v when n_kv_group < n_head) -------
            if self.n_kv_group != self.n_head:
                repeat = self.n_head // self.n_kv_group
                k_adjusted = k.repeat_interleave(repeat, dim=1)
                v_adjusted = v.repeat_interleave(repeat, dim=1)
            else:
                k_adjusted = k
                v_adjusted = v

            att = (q @ k_adjusted.transpose(-2, -1))

            if self.use_qk_norm_scale:
                # utilize learned qk_norm_scaling factor
                att = att * self.qk_norm_factor
            else:
                sqrt_head_dim = math.sqrt(k.size(-1))
                # divide by sqrt of head dimension if not
                att = att / sqrt_head_dim

            # apply lower triangle attention mask
            att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))

            # softmax variation
            if self.softmax_variant_attn != 'softmax':
                att = self.softmax_layer_attn(att)
            else:
                att = F.softmax(att, dim=-1)

            att = self.attn_dropout(att)

            y = att @ v_adjusted # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Concat Heads or Inf Concat Heads
        if self.use_concat_heads:
            # (B, nh, T, v_dim) → (B, T, nh*v_dim); avoid extra .contiguous()
            # flatten heads → (B, T, n_head * n_v_head_dim)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.n_v_head_dim)
            y = self.c_proj(y)
        elif self.n_cproj == 1:
            # Sum heads first: (B, nh, T, v_dim) → (B, T, v_dim)
            y = y.sum(dim=1)
            y = self.c_proj(y)
        else:
            # Sum heads first: (B, nh, T, v_dim) → (B, T, v_dim)
            y_sum = y.sum(dim=1)

            # Parallel small projections then fuse; avoids Python-level loop
            y = torch.stack([proj(y_sum) for proj in self.c_proj_list ], dim=0).sum(dim=0)

        # output projection
        y = self.resid_dropout(y)

        return y

##############################################################################
#  Multi-head Latent Attention (MLA) – DeepSeek-V2 implementation
#  - low-rank joint compression of K & V (latent_kv_dim)
#  - optional low-rank compression of Q (we keep full Q for simplicity)
#  - decoupled RoPE branch (latent_rope_dim) shared across heads
#  - dramatic KV-cache reduction while retaining MHA quality
##############################################################################


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) – see §2.1 of the DeepSeek-V2 paper.
    Only the architectural pieces needed for training/inference are included;
    fast-KV-cache tricks (parameter re-folding at deployment) can be added
    later without changing the forward pass.
    """

    def __init__(self, config, fire_pos_enc: nn.Module | None = None):
        super().__init__()

        # ── dimensions ──────────────────────────────────────────────────────
        self.n_head         = config.n_head
        self.d_head         = config.n_embd // config.n_head
        self.d_latent_kv    = getattr(config, "latent_kv_dim", 4 * self.d_head)
        self.d_rope         = getattr(config, "latent_rope_dim", self.d_head // 2)
        self.dropout_p      = config.dropout

        # ── projections ─────────────────────────────────────────────────────
        # Q : normal per-head projection  (B,T,E) → (B,T,H·d_h)
        self.q_proj  = nn.Linear(config.n_embd, self.n_head * self.d_head, bias=config.bias)

        # RoPE query/key branches – a *small* extra dimension that carries
        # positional signal (decoupled strategy in the paper)
        self.q_rope_proj = nn.Linear(config.n_embd, self.n_head * self.d_rope, bias=config.bias)
        self.k_rope_proj = nn.Linear(config.n_embd, self.d_rope,              bias=config.bias)

        # Low-rank joint compression:  down-project to latent and up-project
        self.kv_down_proj = nn.Linear(config.n_embd, self.d_latent_kv, bias=False)
        self.k_up_proj    = nn.Linear(self.d_latent_kv, self.n_head * self.d_head, bias=False)
        self.v_up_proj    = nn.Linear(self.d_latent_kv, self.n_head * self.d_head, bias=False)

        # Output
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Rotary embeddings only on the RoPE branches (tiny, shared)
        self.rope_q = RotaryEmbedding(config, size=self.d_rope)
        self.rope_k = RotaryEmbedding(config, size=self.d_rope)

        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.resid_dropout = nn.Dropout(self.dropout_p)


        # ───────────  Quiet-Attention style "+" denominator  ────────────
        # Learned *per-head* constant C = exp(lobo_log)  (log-space param)
        self.use_lobo = getattr(config, "use_mla_lobo", False)
        if self.use_lobo:
            init = getattr(config, "mla_lobo_init", 0.0)
            self.lobo_log = nn.Parameter(torch.full((self.n_head,), init))
        else:
            self.register_parameter("lobo_log", None)

        # Pre-build causal mask for the worst-case block_size
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    # --------------------------------------------------------------------- #
    def _shape(self, x: torch.Tensor, heads: int, head_dim: int):
        """(B,T,E) → (B,H,T,D)"""
        B, T, _ = x.shape
        return x.view(B, T, heads, head_dim).transpose(1, 2)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, iter_num: int | None = None):
        """
        Args:
            x: (B, T, E)
        Returns:
            y: (B, T, E)
        """
        B, T, _ = x.shape

        # ── fast projections ───────────────────────────────────────────────
        q_c = self._shape(self.q_proj(x), self.n_head, self.d_head)          # (B,H,T,d_h)
        q_r = self._shape(self.q_rope_proj(x), self.n_head, self.d_rope)     # (B,H,T,d_r)

        # latent-KV path
        latent = self.kv_down_proj(x)                                        # (B,T,d_latent)
        k_c = self._shape(self.k_up_proj(latent), self.n_head, self.d_head)  # (B,H,T,d_h)
        v   = self._shape(self.v_up_proj(latent), self.n_head, self.d_head)  # (B,H,T,d_h)

        # shared RoPE key (broadcast to every head)
        k_r_shared = self.k_rope_proj(x)                                     # (B,T,d_r)
        k_r_shared = self.rope_k(k_r_shared)                                 # apply RoPE
        k_r = k_r_shared.unsqueeze(1).expand(-1, self.n_head, -1, -1)        # (B,H,T,d_r)

        # RoPE for queries
        q_r = self.rope_q(q_r)

        # concat latent and RoPE parts
        q = torch.cat([q_c, q_r], dim=-1)                                    # (B,H,T,d_h+d_r)
        k = torch.cat([k_c, k_r], dim=-1)

        # ── scaled dot-product attention (no Flash for clarity) ────────────
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))           # (B,H,T,T)
        causal = self.causal_mask[..., :T, :T]
        scores = scores.masked_fill(causal == 0, float("-inf"))

        if self.use_lobo:
            # ---- Quiet-Attention softmax with learned "+" ------------------------
            scores_max  = scores.detach().max(dim=-1, keepdim=True).values       # stability
            exp_scores  = torch.exp(scores - scores_max)                          # (B,H,T,T)

            denom = exp_scores.sum(-1, keepdim=True)                              # (B,H,T,1)

            C = torch.exp(self.lobo_log).view(1, -1, 1, 1)                   # (1,H,1,1)
            attn = exp_scores / (denom + C)
        else:
            attn = torch.softmax(scores, dim=-1)

        attn = self.attn_dropout(attn)

        y = attn @ v                                                        # (B,H,T,d_h)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)                   # (B,T,E)
        y = self.resid_dropout(self.out_proj(y))
        return y

###############################################################################
#  Co4Attention  —  Triadic Modulation + Latent-to-Token Attention
#  --------------------------------------------------------------
#  Complexity  O(Lq · N)   (Lq = n_latent  ≪  seq_len N)
#  The class deliberately avoids the heavy quantization / Flash / GQA plumbing
#  present in CausalSelfAttention; if those features are needed, extend here.
###############################################################################

class Co4Attention(nn.Module):
    """
    Implementation of the Co4 mechanism (Adeel 2025 §4).
    *   L learnable latent-query tokens
    *   triadic loops between (Q, K, V) with MOD transfer-function
    *   final latent-to-token dot-product attention
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()

        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd   = config.n_embd
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.n_latent      = getattr(config, "n_latent", 4)
        self.triadic_loops = getattr(config, "triadic_loops", 1)
        mod_name           = getattr(config, "mod_fn", "cooperation")
        self.mod_fn        = mod_fn_dict[mod_name]
        print(mod_name)

        # Projections
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Learnable latent queries  (Lq, E)
        self.latent = nn.Parameter(torch.randn(self.n_latent, config.n_embd) / math.sqrt(config.n_embd))

        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ─────────────────────────── helpers ────────────────────────────────
    def _split_heads(self, x):
        # (B,S,E) → (B,H,S,Hd)
        B, S, _ = x.shape
        return x.view(B, S, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        # (B,H,S,Hd) → (B,S,E)
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    # ─────────────────────────── forward ────────────────────────────────
    def forward(self, x, iter_num=None):
        """
        x : (B, N, E)
        """
        B, N, _ = x.shape

        k = self.k_proj(x)                         # (B, N, E)
        v = self.v_proj(x)                         # (B, N, E)
        q = self.latent.unsqueeze(0).expand(B, -1, -1)  # (B, Lq, E)

        # --- triadic modulation --------------------------------------------------
        for _ in range(self.triadic_loops):
            c_q = 0.5 * (k.mean(1, keepdim=True) + v.mean(1, keepdim=True))  # (B,1,E)
            q   = self.mod_fn(q, c_q)                                        # (B,Lq,E)

            c_k = 0.5 * (q.mean(1, keepdim=True).repeat(1, N, 1) + v)
            k   = self.mod_fn(k, c_k)                                        # (B,N,E)

            c_v = 0.5 * (q.mean(1, keepdim=True).repeat(1, N, 1) + k)
            v   = self.mod_fn(v, c_v)                                        # (B,N,E)

        q = self.q_proj(q)                                                   # (B,Lq,E)

        # --- latent-to-token attention ------------------------------------------
        qh = self._split_heads(q)     # (B,H,Lq,Hd)
        kh = self._split_heads(k)     # (B,H,N ,Hd)
        vh = self._split_heads(v)     # (B,H,N ,Hd)

        att = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)         # (B,H,Lq,N)
        att = att.softmax(dim=-1)

        yh  = att @ vh                                                     # (B,H,Lq,Hd)
        yh  = yh.mean(2, keepdim=True)                                      # (B,H,1,Hd)
        y   = self._merge_heads(yh).repeat(1, N, 1)                          # (B,N,E)

        y = self.resid_dropout(self.out_proj(y))
        return y

class LogLinearAttention(nn.Module):
    """
    Implementation of Log-Linear Attention as described in arXiv:2506.04761v1.
    This mechanism balances linear attention's efficiency with softmax attention's
    expressiveness by using a logarithmically growing set of hidden states.
    """
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Number of log-buckets for the hidden states
        self.n_log_buckets = getattr(config, "n_log_buckets", int(math.log2(config.block_size)))
        
        # Linear layers for Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.head_size)))
        
        # Learnable bucket embeddings
        self.bucket_embeddings = nn.Parameter(
            torch.randn(self.n_log_buckets, self.head_size) / math.sqrt(self.head_size)
        )
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _compute_log_bucket_indices(self, seq_length):
        """Compute logarithmic bucket indices for positions."""
        positions = torch.arange(seq_length, device=self.bucket_embeddings.device)
        log_positions = torch.floor(torch.log2(positions + 1)).long()
        return torch.clamp(log_positions, 0, self.n_log_buckets - 1)

    def forward(self, x, iter_num=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Project input to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (B, T, nh, hs)
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        
        # Apply scaling
        q = q * self.scale
        k = k * self.scale
        
        # Compute log bucket indices and get bucket embeddings
        bucket_indices = self._compute_log_bucket_indices(T)
        bucket_emb = self.bucket_embeddings[bucket_indices]  # (T, hs)
        
        # Add bucket embeddings to keys
        k = k + bucket_emb.unsqueeze(0).unsqueeze(2)  # (B, T, nh, hs)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute attention with log-linear hidden states
        # First compute cumulative sums for keys and values
        k_cumsum = k.cumsum(dim=2)  # (B, nh, T, hs)
        kv = k * v
        kv_cumsum = kv.cumsum(dim=2)  # (B, nh, T, hs)
        
        # Compute attention output using cumulative sums
        y = torch.einsum('bhts,bhsd->bhtd', q, kv_cumsum) / (
            torch.einsum('bhts,bhsd->bhtd', q, k_cumsum) + 1e-6
        )
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y

attention_dictionary = {
    "causal": CausalSelfAttention,
    "linear": LinearAttention,
    # "ssm": MambaBlock,
    "identity": AttnIdentity,
    "infinite": InfiniteHeadAttention,
    "mla": MultiHeadLatentAttention,
    "co4": Co4Attention,
    "log_linear": LogLinearAttention,
}

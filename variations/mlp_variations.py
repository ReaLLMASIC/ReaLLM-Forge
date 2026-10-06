# variations/mlp_variations.py

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary, wrap_with_flashnorm
from quantization.quantize import fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers


def _maybe_print_l2_norm(name: str, weight: torch.Tensor, dim: int, enabled: bool, should_print: bool):
    if enabled and should_print:
        print(f"L2-normalizing {name} over dim {dim} (size {weight.size(dim)})")

class OriginalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.mlp_down_projs = config.mlp_down_projs

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # L2 normalization options
        self.l2_norm_mlp_up = config.l2_norm_mlp_up
        self.l2_norm_mlp_down = config.l2_norm_mlp_down
        self.l2_norm_mlp_up_dim = config.l2_norm_mlp_up_dim
        self.l2_norm_mlp_down_dim = config.l2_norm_mlp_down_dim
        self.l2_norm_print_dims = config.l2_norm_print_dims

        # Add learnable or fixed offsets for the activation function
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Fused down projection
        self.c_proj = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd * self.mlp_down_projs,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=use_down_bias,
        )

        up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
        down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1

        _maybe_print_l2_norm("MLP up projection", self.c_fc.weight, up_dim, self.l2_norm_mlp_up, self.l2_norm_print_dims)
        _maybe_print_l2_norm("MLP down projection", self.c_proj.weight, down_dim, self.l2_norm_mlp_down, self.l2_norm_print_dims)

        self.post_act_l2_norm = config.mlp_post_act_l2_norm
        self.cproj_scale = config.mlp_cproj_scale

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight = F.normalize(self.c_fc.weight, p=2, dim=up_dim)
            x = F.linear(x, weight, self.c_fc.bias)
        else:
            x = self.c_fc(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

        # Apply offsets to the activation function
        x = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset

        if self.post_act_l2_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x = x / self.cproj_scale

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

        # Option to keep cdown proj on hypersphere
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight = F.normalize(self.c_proj.weight, p=2, dim=down_dim)
            x = F.linear(x, weight, self.c_proj.bias)
        else:
            x = self.c_proj(x)

        # Apply fused down projection and sum the outputs
        if self.mlp_down_projs > 1:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len, self.mlp_down_projs, -1)
            x = x.sum(dim=2)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class EdgeLLMASICMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.mlp_down_projs = config.mlp_down_projs

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        if config.use_gradual_activation:
            self.use_gradual_activation = True
            self.start_activation = activation_dictionary[config.activation_start](config=config)
            self.end_activation = activation_dictionary[config.activation_end](config=config)
            self.transition_start = config.activation_transition_start_iter
            self.transition_end = config.activation_transition_end_iter if config.activation_transition_end_iter is not None else config.max_iters
        else:
            self.use_gradual_activation = False

        # L2 normalization options
        if config.l2_norm_mlp_up:
            print("Warning: EdgeLLMASICMLP does not support l2_norm_mlp_up.")
        if config.l2_norm_mlp_down:
            print("Warning: EdgeLLMASICMLP does not support l2_norm_mlp_down.")

        # Add learnable or fixed offsets for the activation function
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = wrap_with_flashnorm(linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)], config)
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Fused down projection
        self.c_proj = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd * self.mlp_down_projs,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=use_down_bias,
        )

        self.post_act_l2_norm = config.mlp_post_act_l2_norm
        self.cproj_scale = config.mlp_cproj_scale

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        x = self.c_fc(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

        if self.use_gradual_activation:
            if not self.training:
                x = self.end_activation(x - self.activation_x_offset) - self.activation_y_offset
            else:
                if iter_num == None:
                    raise ValueError("Iter_num was not passed to GPT model")

                transition_percentage = (iter_num - self.transition_start) / float(self.transition_end - self.transition_start)
                transition_percentage = max(0.0, min(1.0, transition_percentage))  # clamp to [0, 1]

                # Compute both activations
                start_out = self.start_activation(x - self.activation_x_offset)
                end_out = self.end_activation(x - self.activation_x_offset)

                # Blend between start and end
                x = (1 - transition_percentage) * start_out + transition_percentage * end_out
                x = x - self.activation_y_offset
        else:
            # Apply offsets to the activation function
            x = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset

        if self.post_act_l2_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x = x / self.cproj_scale

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

        x = self.c_proj(x)

        # Apply fused down projection and sum the outputs
        if self.mlp_down_projs > 1:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len, self.mlp_down_projs, -1)
            x = x.sum(dim=2)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class DualPathMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # L2 normalization options
        self.l2_norm_mlp_up = config.l2_norm_mlp_up
        self.l2_norm_mlp_down = config.l2_norm_mlp_down
        self.l2_norm_mlp_up_dim = config.l2_norm_mlp_up_dim
        self.l2_norm_mlp_down_dim = config.l2_norm_mlp_down_dim

        # Dual path specific parameters
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Instantiate Linear Layers with configurable bias
        self.c_fc = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=config.mlp_up_bias
        )

        # Two separate projection layers for each activation path
        self.c_proj1 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        self.c_proj2 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
        down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1

        _maybe_print_l2_norm("DualPathMLP up projection", self.c_fc.weight, up_dim, self.l2_norm_mlp_up, self.l2_norm_print_dims)
        _maybe_print_l2_norm("DualPathMLP down projection 1", self.c_proj1.weight, down_dim, self.l2_norm_mlp_down, self.l2_norm_print_dims)
        _maybe_print_l2_norm("DualPathMLP down projection 2", self.c_proj2.weight, down_dim, self.l2_norm_mlp_down, self.l2_norm_print_dims)

        self.post_act_l2_norm = config.mlp_post_act_l2_norm
        self.cproj_scale = config.mlp_cproj_scale

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):
        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        # Common upscale projection
        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight = F.normalize(self.c_fc.weight, p=2, dim=up_dim)
            x = F.linear(x, weight, self.c_fc.bias)
        else:
            x = self.c_fc(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

        # First activation path - shifted right
        x1 = self.activation_variant(x - self.activation_x_offset) - self.activation_y_offset

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x1 = x1 / x1.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x1 = x1 / self.cproj_scale

        # normalization
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight1 = F.normalize(self.c_proj1.weight, p=2, dim=down_dim)
            x1 = F.linear(x1, weight1, self.c_proj1.bias)
        else:
            x1 = self.c_proj1(x1)

        # Second activation path - shifted left and negated input
        x2 = -self.activation_variant(-(x + self.activation_x_offset)) - self.activation_y_offset

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x2 = x2 / x2.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x2 = x2 / self.cproj_scale

        # normalization
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight2 = F.normalize(self.c_proj2.weight, p=2, dim=down_dim)
            x2 = F.linear(x2, weight2, self.c_proj2.bias)
        else:
            x2 = self.c_proj2(x2)

        # Combine paths
        x = x1 + x2

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)

        return x

class Swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler
        self.mlp_down_projs = config.mlp_down_projs

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # L2 normalization options
        self.l2_norm_mlp_up = config.l2_norm_mlp_up
        self.l2_norm_mlp_down = config.l2_norm_mlp_down
        self.l2_norm_mlp_up_dim = config.l2_norm_mlp_up_dim
        self.l2_norm_mlp_down_dim = config.l2_norm_mlp_down_dim

        # Add learnable or fixed offsets for the activation function
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc_in1 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        self.c_fc_in2 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Fused down projection
        self.c_fc_out = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd * self.mlp_down_projs,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=use_down_bias,
        )

        self.post_act_l2_norm = config.mlp_post_act_l2_norm
        self.cproj_scale = config.mlp_cproj_scale

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        # normalization - cosine similarity with input
        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight1 = F.normalize(self.c_fc_in1.weight, p=2, dim=up_dim)
            x_in1 = F.linear(x, weight1, self.c_fc_in1.bias)
        else:
            x_in1 = self.c_fc_in1(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

        x_in1 = self.activation_variant(x_in1 - self.activation_x_offset) - self.activation_y_offset

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

        # normalization - cosine similarity with input
        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight2 = F.normalize(self.c_fc_in2.weight, p=2, dim=up_dim)
            x_in2 = F.linear(x, weight2, self.c_fc_in2.bias)
        else:
            x_in2 = self.c_fc_in2(x)

        x_out = x_in1 * x_in2

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x_out = x_out / x_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x_out = x_out / self.cproj_scale

        # optional down projection normalization to keep vectors on hypersphere
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight = F.normalize(self.c_fc_out.weight, p=2, dim=down_dim)
            x = F.linear(x_out, weight, self.c_fc_out.bias)
        else:
            x = self.c_fc_out(x_out)

        # Apply fused down projection and sum the outputs
        if self.mlp_down_projs > 1:
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size, seq_len, self.mlp_down_projs, -1)
            x = x.sum(dim=2)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x

class DualPathSwiglu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

        # L2 normalization options
        self.l2_norm_mlp_up = config.l2_norm_mlp_up
        self.l2_norm_mlp_down = config.l2_norm_mlp_down
        self.l2_norm_mlp_up_dim = config.l2_norm_mlp_up_dim
        self.l2_norm_mlp_down_dim = config.l2_norm_mlp_down_dim

        # Dual path specific parameters
        if config.learn_mlp_x_offset:
            self.activation_x_offset = nn.Parameter(torch.tensor(config.mlp_x_offset))
        else:
            self.register_buffer("activation_x_offset", torch.tensor(config.mlp_x_offset))

        if config.learn_mlp_y_offset:
            self.activation_y_offset = nn.Parameter(torch.tensor(config.mlp_y_offset))
        else:
            self.register_buffer("activation_y_offset", torch.tensor(config.mlp_y_offset))

        # Sets the class of linear for MLP
        self.linear_variant_mlp_up = linear_dictionary[set_variant(config.linear_variant_mlp_up, config.linear_variant_mlp)]
        self.linear_variant_mlp_down = linear_dictionary[set_variant(config.linear_variant_mlp_down, config.linear_variant_mlp)]

        self.quantization_mlp_dict = {}
        self.quantization_mlp_dict["activations_quant_method"] = config.activations_quant_method

        # Set quantization parameters for MLP
        for arg, val in vars(config).items():
            # Set MLP Activation precision and quantization method
            if arg.startswith("quantize_") and "mlp_act" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act_bits)
            elif arg.startswith("quantize_") and "mlp_act" in arg:
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_mlp_act)
                if config.store_activations and arg != "quantize_mlp_act" and self.quantization_mlp_dict[arg]:
                    create_activation_buffers(self, arg)
            # Set MLP Linear Weight precision and quantization method
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_bits"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_bits)
            elif arg.startswith("quantize_") and "linear_mlp" in arg and arg.endswith("_method"):
                self.quantization_mlp_dict[arg] = set_variant(val, config.quantize_linear_method)

        mlp_expansion_size = None
        if config.mlp_size is not None:
            mlp_expansion_size = config.mlp_size
        else:
            mlp_expansion_size = config.mlp_expansion_factor * config.n_embd

        # Determine bias settings - use specific up/down if set, otherwise use global bias
        use_up_bias = config.mlp_up_bias if config.mlp_up_bias is not None else config.bias
        use_down_bias = config.mlp_down_bias if config.mlp_down_bias is not None else config.bias

        # Instantiate Linear Layers with configurable bias
        self.c_fc_in1 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        self.c_fc_in2 = self.linear_variant_mlp_up(
            config.n_embd,
            mlp_expansion_size,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_up_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_up_bits"],
            bias=use_up_bias
        )

        # Two separate projection layers for each activation path
        self.c_proj1 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        self.c_proj2 = self.linear_variant_mlp_down(
            mlp_expansion_size,
            config.n_embd,
            config,
            self.quantization_mlp_dict["quantize_linear_mlp_down_method"],
            self.quantization_mlp_dict["quantize_linear_mlp_down_bits"],
            bias=config.mlp_down_bias
        )

        self.post_act_l2_norm = config.mlp_post_act_l2_norm
        self.cproj_scale = config.mlp_cproj_scale

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):
        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        # normalization - cosine similarity with input
        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight1 = F.normalize(self.c_fc_in1.weight, p=2, dim=up_dim)
            x_in1 = F.linear(x, weight1, self.c_fc_in1.bias)
        else:
            x_in1 = self.c_fc_in1(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

        x_in1 = self.activation_variant(x_in1 - self.activation_x_offset) - self.activation_y_offset

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

        # normalization - cosine similarity with input
        if self.l2_norm_mlp_up:
            up_dim = 1 if self.l2_norm_mlp_up_dim == 'embed' else 0
            weight2 = F.normalize(self.c_fc_in2.weight, p=2, dim=up_dim)
            x_in2 = F.linear(x, weight2, self.c_fc_in2.bias)
        else:
            x_in2 = self.c_fc_in2(x)

        x_out = x_in1 * x_in2

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x_out = x_out / x_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x_out = x_out / self.cproj_scale

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_out = fake_quantize_act(self, "mlp_act_activation_input", x_out, num_bits, quant_method, iter_num)

        # First activation path - shifted right
        x1 = self.activation_variant(x_out - self.activation_x_offset) - self.activation_y_offset

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x1 = x1 / x1.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x1 = x1 / self.cproj_scale

        # normalization
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight1 = F.normalize(self.c_proj1.weight, p=2, dim=down_dim)
            x1 = F.linear(x1, weight1, self.c_proj1.bias)
        else:
            x1 = self.c_proj1(x1)

        # Second activation path - shifted left and negated input
        x2 = -self.activation_variant(-(x_out + self.activation_x_offset)) - self.activation_y_offset

        # Mitigate Cproj Down Spikes
        if self.post_act_l2_norm:
            x2 = x2 / x2.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.cproj_scale is not None and self.cproj_scale != 1.0:
            x2 = x2 / self.cproj_scale

        # normalization
        if self.l2_norm_mlp_down:
            down_dim = 0 if self.l2_norm_mlp_down_dim == 'embed' else 1
            weight2 = F.normalize(self.c_proj2.weight, p=2, dim=down_dim)
            x2 = F.linear(x2, weight2, self.c_proj2.bias)
        else:
            x2 = self.c_proj2(x2)

        # Combine paths
        x = x1 + x2

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)

        return x


class KanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.kan = linear_dictionary["kan"](config.n_embd, config.n_embd, config=config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None):

        x = self.kan(x)
        x = self.dropout(x)

        return x

class MLP_Identity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = nn.Identity()

    def forward(self, x, iter_num=None):
        x = self.activation(x)
        return x


def _normalized_hadamard(size, *, device=None, dtype=None):
    """Construct the orthonormal Walsh-Hadamard matrix (size must be 2^k)."""
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard factor size must be a positive power of two, got {size}")
    result = torch.ones(1, 1, device=device, dtype=dtype)
    while result.size(0) < size:
        result = torch.cat(
            (torch.cat((result, result), dim=1), torch.cat((result, -result), dim=1)),
            dim=0,
        ) / math.sqrt(2.0)
    return result


class HadamardMLP(nn.Module):
    """Parameter-efficient channel mixer based on learned Kronecker factors.

    Channels are zero-padded to a square tile. Each stage applies A^T Z B,
    starting from an exact Walsh-Hadamard transform, with fixed shuffles between
    stages. Learned diagonals and an optional low-rank token-conditioned gain
    surround the SiLU nonlinearity as described by Cactus' Hadamard MLP.
    """

    def __init__(self, config):
        super().__init__()
        self.input_size = config.n_embd
        requested_size = config.hadamard_mlp_factor_size
        if requested_size == 0:
            requested_size = 1 << math.ceil(math.log2(math.ceil(math.sqrt(self.input_size))))
        if requested_size * requested_size < self.input_size:
            raise ValueError("hadamard_mlp_factor_size squared must cover n_embd")
        # Validate before creating parameters and retain the matrix for initialization.
        hadamard = _normalized_hadamard(requested_size)
        self.factor_size = requested_size
        self.padded_size = requested_size * requested_size
        self.num_stages = config.hadamard_mlp_stages
        if self.num_stages < 1:
            raise ValueError("hadamard_mlp_stages must be at least 1")

        self.left_factors = nn.ParameterList()
        self.right_factors = nn.ParameterList()
        for _ in range(self.num_stages):
            self.left_factors.append(nn.Parameter(hadamard.clone()))
            self.right_factors.append(nn.Parameter(hadamard.clone()))

        self.diagonals = nn.ParameterList(
            [nn.Parameter(torch.ones(self.padded_size)) for _ in range(4)]
        )
        nn.init.constant_(self.diagonals[-1], config.hadamard_mlp_output_scale)
        self.bias = nn.Parameter(torch.zeros(self.padded_size))

        self.gain_rank = config.hadamard_mlp_gain_rank
        if self.gain_rank < 0:
            raise ValueError("hadamard_mlp_gain_rank cannot be negative")
        if self.gain_rank:
            self.gain_v = nn.Linear(self.input_size, self.gain_rank, bias=False)
            self.gain_u = nn.Parameter(torch.zeros(self.gain_rank, self.padded_size))

        # randperm follows the run's torch seed and buffers make checkpoint reload exact.
        for stage in range(self.num_stages - 1):
            self.register_buffer(f"permutation_{stage}", torch.randperm(self.padded_size))
        self.dropout = nn.Dropout(config.dropout)

    def _mix(self, x, stage):
        shape = x.shape
        tile = x.reshape(*shape[:-1], self.factor_size, self.factor_size)
        tile = torch.matmul(self.left_factors[stage].t(), tile)
        tile = torch.matmul(tile, self.right_factors[stage])
        return tile.reshape(shape)

    def forward(self, x, iter_num=None):
        del iter_num
        if self.padded_size != self.input_size:
            x_padded = F.pad(x, (0, self.padded_size - self.input_size))
        else:
            x_padded = x

        hidden = self._mix(x_padded * self.diagonals[0], 0)
        if self.num_stages > 1:
            hidden = hidden.index_select(-1, self.permutation_0)
        gain = 1.0
        if self.gain_rank:
            gain = 1.0 + torch.matmul(F.softmax(self.gain_v(x), dim=-1), self.gain_u)
        hidden = F.silu(hidden * self.diagonals[1] * gain + self.bias)

        if self.num_stages == 1:
            hidden = hidden * self.diagonals[2]
        for stage in range(1, self.num_stages):
            if stage == self.num_stages - 1:
                hidden = hidden * self.diagonals[2]
            hidden = self._mix(hidden, stage)
            if stage < self.num_stages - 1:
                hidden = hidden.index_select(-1, getattr(self, f"permutation_{stage}"))
        hidden = hidden * self.diagonals[3]
        return self.dropout(hidden[..., :self.input_size])

mlp_dictionary = {
    "mlp": OriginalMLP,
    "edgellm_asic_mlp": EdgeLLMASICMLP,
    "swiglu": Swiglu,
    "identity": MLP_Identity,
    "kan": KanMLP,
    "dual_path": DualPathMLP,
    "dual_path_swiglu": DualPathSwiglu,
    "hadamard": HadamardMLP,
    }

def get_mlp_instance(config):
    mlp_type = config.mlp_variant
    mlp_class = mlp_dictionary.get(mlp_type)
    if mlp_class is None:
        raise ValueError(f"Unsupported MLP variant: {mlp_type}")
    return mlp_class(config)

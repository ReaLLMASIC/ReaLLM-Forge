# variations/mlp_variations.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from variations.activation_variations import activation_dictionary
from variations.linear_variations import linear_dictionary
from quantization.quantize import fake_quantize_act
from quantization.quant_utils import set_variant, create_activation_buffers
from torch.linalg import matrix_exp
from torch.nn import ModuleList
import math


class OriginalMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        # Select "mlp variant"
        self.mlp_variant = config.mlp_variant
        self.use_mlp_res = config.mlp_res

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

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

        # Instantiate Linear Layers
        if self.mlp_variant == "mlp":
            self.c_fc = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"], bias=config.bias)
            self.c_proj = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"], bias=config.bias)
        elif self.mlp_variant == "swiglu":
            self.c_fc_in1 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
            self.c_fc_in2 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
            self.c_fc_out = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        if self.mlp_variant == "mlp":
            x = self.c_fc(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_input", x, num_bits, quant_method, iter_num)

            x = self.activation_variant(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x = fake_quantize_act(self, "mlp_act_activation_output", x, num_bits, quant_method, iter_num)

            # MLP Residual
            if self.use_mlp_res:
                if mlp_res is None:
                    mlp_res = torch.zeros_like(x)
                mlp_res = x + mlp_res
                x = mlp_res


            x = self.c_proj(x)

        elif self.mlp_variant == "swiglu":
            x_in1 = self.c_fc_in1(x)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

            x_in1 = self.activation_variant(x_in1)

            if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
                num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
                quant_method = self.quantization_mlp_dict["activations_quant_method"]
                x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

            x_in2 = self.c_fc_in2(x)
            x_out = x_in1 * x_in2

            # MLP Residual on the x_out
            if self.use_mlp_res:
                if mlp_res is None:
                    mlp_res = torch.zeros_like(x_out)
                x_out = mlp_res + x_out
                mlp_res = x_out

            x = self.c_fc_out(x_out)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x, mlp_res

class Swiglu(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.full_quant_iteration = config.full_quant_iteration
        self.eval_interval = config.eval_interval

        self.start_quant_level = config.start_quant_level
        self.quant_scheduler = config.quant_scheduler

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant](config=config)

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

        self.c_fc_in1 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
        self.c_fc_in2 = self.linear_variant_mlp_up(config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])
        self.c_fc_out = self.linear_variant_mlp_down(config.mlp_expansion_factor * config.n_embd, config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_down_method"], self.quantization_mlp_dict["quantize_linear_mlp_down_bits"])

        self.mlp_res_gate = self.linear_variant_mlp_up(config.mlp_expansion_factor * config.n_embd, config.mlp_expansion_factor * config.n_embd, config, self.quantization_mlp_dict["quantize_linear_mlp_up_method"], self.quantization_mlp_dict["quantize_linear_mlp_up_bits"])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None):

        if self.quantization_mlp_dict["quantize_mlp_act_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_input", x, num_bits, quant_method, iter_num)

        x_in1 = self.c_fc_in1(x)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_input"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_input_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_input", x_in1, num_bits, quant_method, iter_num)

        x_in1 = self.activation_variant(x_in1)

        if self.quantization_mlp_dict["quantize_mlp_act_activation_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_activation_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x_in1 = fake_quantize_act(self, "mlp_act_activation_output", x_in1, num_bits, quant_method, iter_num)

        x_in2 = self.c_fc_in2(x)
        x_out = x_in1 * x_in2

        # MLP Residual on the x_out
        if mlp_res is None:
            mlp_res = torch.zeros_like(x_out)
        x_out = mlp_res + x_out
        mlp_res = x_out

        x = self.c_fc_out(x_out)

        x = self.dropout(x)

        if self.quantization_mlp_dict["quantize_mlp_act_output"]:
            num_bits = self.quantization_mlp_dict["quantize_mlp_act_output_bits"]
            quant_method = self.quantization_mlp_dict["activations_quant_method"]
            x = fake_quantize_act(self, "mlp_act_output", x, num_bits, quant_method, iter_num)
        return x, mlp_res

class KanMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.kan = linear_dictionary["kan"](config.n_embd, config.n_embd, config=config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None):

        x = self.kan(x)
        x = self.dropout(x)

        return x, None

class LearnedRotationMLP(nn.Module):
    """
    MLP variation using a single learned rotation.
    Inspired by "On Learning Rotations" (Arora, 2009), using the exponential map
    from a skew-symmetric matrix. Simpler structure like KanMLP.
    Ignores quantization internally for simplicity.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # Learnable matrix A. Skew-symmetry is enforced in the forward pass.
        # Initialize near zero for near-identity initial rotation.
        self.A = nn.Parameter(torch.randn(config.n_embd, config.n_embd) * 0.01)

        if config.learned_rotation_scaling:
            self.scaling_factor = nn.Parameter(torch.tensor([1.0]))
        else:
            self.scaling_factor = 1.0

        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, x, iter_num=None, mlp_res=None): # Match signature, ignore mlp_res
        # Enforce skew-symmetry
        S = self.A - self.A.transpose(-1, -2)
        # Compute rotation matrix using matrix exponential
        R = matrix_exp(S) # R is now in SO(n_embd)
        # Apply rotation
        x = x @ R # x is (B, T, C), R is (C, C)
        x = x * self.scaling_factor
        # Apply dropout
        x = self.dropout_layer(x)
        # Return x and None for mlp_res to match expected return signature of other MLPs
        return x, None

class AngleAndPlaceMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_rotations = config.sequence_rotations

        if self.n_embd < 3:
            raise ValueError("n_embd must be at least 3 for rotation around axis 2.")
        if self.num_rotations <= 0:
            raise ValueError("num_rotations must be positive.")

        # --- Learnable Parameters ---
        # Initial scaling factor for 'a' vector
        self.scale_a = nn.Parameter(torch.tensor(config.sequence_vec_a_mag))

        # Final scaling factor for adding 'c' vector
        self.scale_c = nn.Parameter(torch.tensor(config.sequence_vec_b_mag))

        # The 'n' learnable vectors used for dot products. Shape: (num_rotations, n_embd)
        # Transpose it for efficient matmul later: (n_embd, num_rotations)
        # Initialize them randomly, e.g., from a normal distribution.
        self.learnable_vectors = nn.Parameter(torch.randn(self.n_embd, self.num_rotations) * (1.0 / math.sqrt(self.n_embd))) # Kaiming/Lecun init style adjustment
        # --- End Learnable Parameters ---

        # --- Fixed Vectors (Buffers) ---
        a_vec = torch.zeros(self.n_embd)
        a_vec[0] = 1.0
        self.register_buffer('a_vec', a_vec)

        c_vec = torch.zeros(self.n_embd)
        c_vec[2] = 1.0
        self.register_buffer('c_vec', c_vec)
        # --- End Fixed Vectors ---

        self.learned_rotation = LearnedRotationMLP(config)

        # Constants for rotation plane
        self.idx_a = 0 # Index of component 'a'
        self.idx_b = 1 # Index of component 'b'

    def forward(self, x, iter_num = None, mlp_res = None):
        # x shape: (batch_size, sequence_length, n_embd) -> B, T, E
        B, T, E = x.shape

        # Ensure fixed vectors are on the correct device/dtype
        # Use the device/dtype of the input tensor x for consistency
        current_a_vec = self.a_vec.to(device=x.device, dtype=x.dtype)
        current_c_vec = self.c_vec.to(device=x.device, dtype=x.dtype)

        # 1. Initialize the vector to be rotated
        # Start with scaled 'a' vector, same for all positions initially
        # Expand it to match batch and sequence dimensions: B, T, E
        v = self.scale_a * current_a_vec # Shape (E,)
        v = v.unsqueeze(0).unsqueeze(0).expand(B, T, -1) # Shape (B, T, E)

        # 2. Calculate ALL rotation angles based on dot products with input x
        # angles[b, t, i] = dot(x[b, t, :], learnable_vectors[:, i])
        # Use matrix multiplication for efficiency: (B, T, E) @ (E, N) -> (B, T, N)
        # where N = num_rotations
        # Note: self.learnable_vectors is already stored as (E, N)
        all_thetas = torch.matmul(x, self.learnable_vectors) # Shape (B, T, N)

        # 3. Perform n sequential rotations (vectorized over B, T)
        # Loop through each rotation step
        for i in range(self.num_rotations):
            # Get angles for this rotation step for all B, T positions
            theta_i = all_thetas[:, :, i] # Shape (B, T)

            # Calculate cosine and sine for these angles
            cos_theta = torch.cos(theta_i) # Shape (B, T)
            sin_theta = torch.sin(theta_i) # Shape (B, T)

            # Get the components to be rotated from the current vector 'v'
            comp_a = v[:, :, self.idx_a] # Shape (B, T)
            comp_b = v[:, :, self.idx_b] # Shape (B, T)

            # Apply rotation (vectorized)
            # Need to unsqueeze cos/sin to multiply with components if needed,
            # but direct element-wise multiplication works here as shapes match.
            new_comp_a = comp_a * cos_theta - comp_b * sin_theta
            new_comp_b = comp_a * sin_theta + comp_b * cos_theta

            # Update the components in 'v' for the next iteration
            # Create a temporary copy or update carefully if needed
            # Direct assignment should be fine here if v is not used elsewhere between updates
            v = v.clone() # Clone to ensure we don't have issues with view updates if any exist
            v[:, :, self.idx_a] = new_comp_a
            v[:, :, self.idx_b] = new_comp_b
            # Other components v[:, :, 2:] remain unchanged by this rotation

        # 4. Add the scaled 'c' vector
        # Expand c_vec to match B, T dimensions before adding
        final_v = v + (self.scale_c * current_c_vec.unsqueeze(0).unsqueeze(0).expand(B, T, -1))
        final_v, _ = self.learned_rotation(final_v)

        if iter_num is not None and iter_num % 1001 == 0 and iter_num is not None and iter_num != 0:
            print(iter_num)

            print(self.scale_a, self.scale_c)

        return final_v, None

class Identity(nn.Module):
    def __init__(self, config):
        super(Identity, self).__init__()

    def forward(self, x, iter_num=None, mlp_res=None):
        return x, None

mlp_dictionary = {
    "mlp": OriginalMLP,
    "swiglu": Swiglu,
    "kan": KanMLP,
    "learned_rotation": LearnedRotationMLP,
    "axis_rotation_bias": AngleAndPlaceMLP,
    "identity": Identity,
    }

def get_mlp_instance(config):
    mlp_type = config.mlp_variant
    mlp_class = mlp_dictionary.get(mlp_type)
    if mlp_class is None:
        raise ValueError(f"Unsupported MLP variant: {mlp_type}")
    return mlp_class(config)


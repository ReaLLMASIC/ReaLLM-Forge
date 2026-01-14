# norm_variations.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from variations.activation_variations import activation_dictionary

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        bias = config.bias
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class DynamicActivation(nn.Module):
    """ Dynamic Activation Variations, including that of DyT
    from: https://arxiv.org/abs/2503.10622
    https://github.com/jiachenzhu/DyT
    """
    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd

        if config.dact_use_alpha:
            self.alpha = nn.Parameter(torch.ones(1) * config.dact_alpha_init)
        else:
            self.alpha = 1.0

        if config.dact_use_beta:
            self.beta = nn.Parameter(torch.zeros(ndim))
        else:
            self.beta = 0.0

        if config.dact_use_gamma:
            self.gamma = nn.Parameter(torch.ones(ndim))
        else:
            self.gamma = 1.0


        self.activation = activation_dictionary[config.dact_activation](config)

    def forward(self, x):
        return self.gamma * self.activation(self.alpha * x) + self.beta


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / rms * self.gain

class HyperSphereNorm(nn.Module):
    """Normalization to the surface of Hypersphere"""

    def __init__(self, config):
        super().__init__()


        ndim = config.n_embd
        if config.hsnorm_gain:
            self.gain = nn.Parameter(torch.ones(ndim))
        else:
            self.gain = 1.0

        # Determine radius initialization value
        radius_init = None
        if config.hsnorm_radius is not None:
            radius_init = config.hsnorm_radius
        else:
            radius_init = math.sqrt(ndim)

        # constant for loss scaling (default set to 1.0)
        self.const_radius_factor = config.hsnorm_scale

        # Set as constant or learned param
        self.hsnorm_radius_learning = config.hsnorm_radius_learning
        if config.hsnorm_radius_learning:
            # div by const_radius_factor (no effect if is 1.0)
            radius_init = radius_init / self.const_radius_factor
            self.radius_init_factor = nn.Parameter(torch.tensor([radius_init]))
        else:
            self.radius_init_factor = radius_init

    def forward(self, x):
        radius = self.const_radius_factor * self.radius_init_factor
        hypersphere_norm = x.norm(2, dim=-1, keepdim=True)
        return  x / hypersphere_norm * radius * self.gain

class pRMSNorm(nn.Module):
    """Partial RMS Normalization"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim))
        self.p = config.prmsnorm_pct # percent of elements to use

    def forward(self, x):
        # Calculate the number of elements to use for pRMS
        k = math.ceil(x.size(-1) * self.p)

        # Select the first k elements along the last dimension
        x_part = x[..., :k]

        # Calculate pRMS
        prms = x_part.norm(2, dim=-1, keepdim=True) / math.sqrt(k)

        return x / prms * self.gain

class kRMSNorm(nn.Module):
    """First k, Last k, or Random k elements RMS Normalization with optional int8/int16 quantization, no quantization (fp16), and configurable gain"""

    def __init__(self, config):
        super().__init__()
        ndim = config.n_embd
        self.gain = nn.Parameter(torch.ones(ndim)) if config.krmsnorm_enable_gain else None
        self.k = config.krmsnorm_num
        self.quantize_type = config.krmsnorm_quantize_type  # 'int8', 'int16', or 'none'
        self.enable_gain = config.krmsnorm_enable_gain
        self.selection_type = config.krmsnorm_selection_type  # 'first', 'last', or 'random'
        self.recompute_percentage = config.krmsnorm_recompute_percentage
        self.recomputed = False

    def quantize(self, x, dtype):
        if dtype == 'int8':
            qmin, qmax = -128, 127
            scale = (x.max() - x.min()) / (qmax - qmin)
            zero_point = qmin - x.min() / scale
            x_q = (x / scale + zero_point).clamp(qmin, qmax).round().to(torch.int8)
        elif dtype == 'int16':
            qmin, qmax = -32768, 32767
            scale = (x.max() - x.min()) / (qmax - qmin)
            zero_point = qmin - x.min() / scale
            x_q = (x / scale + zero_point).clamp(qmin, qmax).round().to(torch.int16)
        elif dtype == 'none':
            x_q, scale, zero_point = x.half(), 1.0, 0.0
        else:
            raise ValueError("Unsupported quantization type")
        return x_q, scale, zero_point

    def dequantize(self, x_q, scale, zero_point, dtype):
        if dtype in ['int8', 'int16']:
            x = (x_q.to(torch.float32) - zero_point) * scale
        elif dtype == 'none':
            x = x_q.float()
        else:
            raise ValueError("Unsupported quantization type")
        return x

    def forward(self, x):
        # Calculate the number of elements to use for kRMS
        k = min(x.size(-1), self.k)

        # Select elements based on the selection type
        if self.selection_type == 'first':
            x_part = x[..., :k]
        elif self.selection_type == 'last':
            x_part = x[..., -k:]
        elif self.selection_type == 'random':
            indices = torch.randperm(x.size(-1))[:k]
            x_part = x[..., indices]
        else:
            raise ValueError("Unsupported selection type")

        # Quantize x_part
        x_part_q, scale, zero_point = self.quantize(x_part, self.quantize_type)

        # Calculate kRMS on quantized values
        krms = x_part_q.float().norm(2, dim=-1, keepdim=True) / math.sqrt(k)

        # Dequantize the krms
        krms = self.dequantize(krms, scale, zero_point, self.quantize_type)

        # Apply normalization
        if self.recompute_percentage != None:
            rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
            if krms > rms*(1-self.recompute_percentage) and krms < rms*(1+self.recompute_percentage):
                x = x / krms
                self.recomputed = False
            else:
                x = x / rms
                self.recomputed = True
        else:
            x = x / krms

        # Apply gain if enabled
        if self.enable_gain:
            x = x * self.gain

        return x

class IdentityNorm(nn.Module):
    def __init__(self, config=None):  # Accept config for API consistency
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)


norm_dictionary = {
    "layernorm": LayerNorm,
    "rmsnorm": RMSNorm,
    "prmsnorm": pRMSNorm,
    "krmsnorm": kRMSNorm,
    "hyperspherenorm": HyperSphereNorm,
    "dact": DynamicActivation,
    "identity": IdentityNorm,
}

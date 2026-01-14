# variations/softmax_variations.py
import torch
import torch.nn as nn
import math
from variations.activation_variations import activation_dictionary

# Softmax base 2, with option to remove max subtraction
class Softermax(nn.Module):
    """ Base-2 Softmax with option to remove max subtraction"""
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.subtract_max = config.softermax_use_xmax

    def forward(self, x):
        if self.subtract_max:
            max_x = x.max(dim=self.dim, keepdim=True).values
            x = x - max_x
        e_x = torch.pow(2.0, x)
        return e_x / e_x.sum(dim=self.dim, keepdim=True)


# Softmax variation with learnable constant parameters for xmax and denominator
class ConSmax(nn.Module):
    """ Constant learnable parameters for xmax and denominator """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        # learnable 'xmax' - beta
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

        # denominator - gamma
        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))

        # Set the base of the exponent
        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

    def forward(self, x):
        x_adj = x - self.beta
        e_x = torch.pow(self.consmax_base, x_adj)
        result = e_x / self.gamma

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result


# Softmax variation with per-head learnable constant parameters for xmax and denominator
class ConSmaxV2(nn.Module):
    """ Constant learnable parameters for xmax and denominator """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.n_head = config.n_head
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        self.beta_init = config.consmax_initial_beta
        self.gamma_init = config.consmax_initial_gamma
        self.beta_factor = nn.Parameter(torch.ones(self.n_head, 1, 1))
        self.gamma_factor = nn.Parameter(torch.ones(self.n_head, 1, 1))

        # Set beta and gamma as fields for backwards compatibility
        self.beta = self.beta_init * self.beta_factor
        self.gamma = self.beta_init * self.gamma_factor

        # Set optional clamping (on by default)
        self.clamp_inputs = config.consmax_v2_clamping
        self.clamp_value = config.consmax_v2_clamp_value

        # Set the base of the exponent
        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

    def forward(self, x):
        self.beta = self.beta_factor * self.beta_init
        self.gamma = self.gamma_factor * self.gamma_init

        x_adj = x - self.beta
        if self.clamp_inputs:
            x_adj[x_adj > self.clamp_value] = self.clamp_value

        e_x = torch.pow(self.consmax_base, x_adj)

        result = e_x / self.gamma

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

# Constantmax Quantized

## Quantization Methods Utilized for Separate Forward and Backward Passes
def quantize(tensor,scale):
    tensor = tensor.mul(scale)
    tensor = torch.round(tensor)
    return tensor
def dequantize(tensor,scale):
    tensor = tensor.div(scale)
    return tensor

## helper class for Constantmax_quan
class const_quan(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""
    @staticmethod
    def forward(ctx, beta=None, gamma=None):
        #scaling factor for beta and gamma while doing quantization
        scale_beta=100 #scaling factor for quantization, should make it as parameter
        scale_gamma=10
        beta = quantize(beta, scale_beta)
        gamma = quantize(gamma, scale_gamma)
        return dequantize(beta, scale_beta),dequantize(gamma,scale_gamma)

    @staticmethod
    def backward(ctx, grad_gamma, grad_beta):
        return grad_gamma, grad_beta

_const_quan=const_quan.apply

# Softmax variation with quantized xmax and denominator
class ConSmaxQuan(nn.Module):
    """ Quantized version with learnable beta and gamma """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))
        self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))

        self.fake_beta = None
        self.fake_gamma = None

    def forward(self, x):
        if self.training:
            self.fake_beta, self.fake_gamma = _const_quan(self.beta, self.gamma)
            x = x - self.fake_beta
            e_x = torch.exp(x)
            result = e_x / self.fake_gamma
        else:
            scale_beta = 100
            scale_gamma = 10
            x = x - dequantize(quantize(self.beta, scale_beta), scale_beta)
            e_x = torch.exp(x)
            result = e_x / dequantize(quantize(self.gamma, scale_gamma), scale_gamma)

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result


# Like softmax, but parameterized to permit exploration
class Strongermax(nn.Module):
    """ Exploration of Elemental Modifications of Softmax Equation """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.n_head = config.n_head

        # Strongermax Params
        self.strength = config.strongermax_strength
        self.subtract_max = config.strongermax_use_xmax
        self.xmax_guess = config.strongermax_xmax_guess
        self.divisor = config.strongermax_divisor
        self.div_by_seq_len = config.div_by_seq_len

        # Overflow Recompute
        self.overflow_recompute = config.strongermax_overflow_recompute
        self.overflow_recompute_value = config.strongermax_overflow_recompute_value

        # Set optional clamping (off by default)
        self.clamp_inputs = config.strongermax_clamping
        self.clamp_value = config.strongermax_clamp_value

        # Use denominator
        self.div_by_sum_of_terms = config.strongermax_div_by_sum_of_terms

        # Set optional temperature (already divided by sqrt head dimension)
        self.use_learned_temperature_factor = config.strongermax_use_learned_temperature_factor
        if self.use_learned_temperature_factor:
            self.temperature_factor = nn.Parameter(torch.Tensor([config.strongermax_temperature_factor]))
        else:
            self.temperature_factor = config.strongermax_temperature_factor

        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.overflow_recompute:
            assert self.xmax_guess is not None, "For overflow recompute, xmax_guess must be set"

        # Input and Output Logging
        self.softmax_io_logging = config.softmax_io_logging
        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

        # self.obo_offset default is 0.0, https://www.evanmiller.org/attention-is-off-by-one.html
        self.use_learned_obo = config.strongermax_use_learned_obo
        self.use_learned_obo_per_head = config.strongermax_use_learned_obo_per_head

        if self.use_learned_obo_per_head:
            self.obo_offset = nn.Parameter(torch.ones(self.n_head, 1, 1) * config.strongermax_obo)
        else:
            if self.use_learned_obo:
                self.obo_offset = nn.Parameter(torch.Tensor([config.strongermax_obo]))
            else:
                self.obo_offset = config.strongermax_obo

    def forward(self, x):
        x_adj = x

        if self.clamp_inputs:
            x_adj[x > self.clamp_value] = self.clamp_value

        if self.subtract_max:
            # Guessing correctly instead of subtracting real max can save a pass
            # else we use real xmax
            max_x = x_adj.max(dim=self.dim, keepdim=True).values
            if self.overflow_recompute:
                if (torch.max(x_adj - self.xmax_guess)) > self.overflow_recompute_value:
                    x_adj = x_adj - max_x
                else:
                    x_adj = x_adj - self.xmax_guess
            else:
                if self.xmax_guess:
                    x_adj = x_adj - self.xmax_guess
                else:
                    x_adj = x_adj - max_x

        result = torch.pow(self.strength, x_adj / self.temperature_factor)

        if self.div_by_sum_of_terms:
            result = result / (self.obo_offset + result.sum(dim=self.dim, keepdim=True))

        # TODO: Fix to divide by position from first part of context
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        result = result / self.divisor

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

# Using polynomial instead of exponential for Softmax separation non-linearity
class Polymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(0 - self.x_intercept) # aka 'slope', also x intercept !=0

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor
        self.div_by_seq_len = config.div_by_seq_len
        self.softmax_io_logging = config.softmax_io_logging
        self.softmax_io_log_interval = config.softmax_io_log_interval
        self.iter_num = 0

        if self.softmax_io_logging:
            self.inputs = []
            self.outputs = []

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf
        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # Divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        if self.training and self.softmax_io_logging and self.iter_num % self.softmax_io_log_interval == 0:
            self.inputs = x
            self.outputs = result

        if self.training:
            self.iter_num += 1

        return result

class VPolymax(nn.Module):
    """ variation of polymax with a v-shape, and is non-monotonically increasing"""
    def __init__(self, config, dim=-1):
        super().__init__()

        assert(config.polymax_x_intercept < 0) # ensure x_intercept is strictly left of the y-axis
        self.dim = dim
        self.div_by_seq_len = config.div_by_seq_len

        self.x_intercept = config.polymax_x_intercept # where to transition from y=0 to m*x+b
        self.y_intercept = config.polymax_y_intercept # where the graph crosses y-axis
        self.linear_slope = (self.y_intercept - 0)/(self.x_intercept - 0) # vpoly uses reverse slope

        self.power = config.polymax_power
        self.divisor = config.polymax_divisor

    def forward(self, x):
        # Overview:
        # Flat section:       -inf < x < x_intercept
        # Linear section:     x_intercept <= x <= 0
        # Polynomial section: 0 < x < inf

        # Flat section
        flat_piece = torch.where(x < self.x_intercept, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))

        # Linear section
        linear_piece = torch.where((x >= self.x_intercept) & (x <= 0), self.linear_slope * x + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x > 0, x**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + linear_piece + flat_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class SaturatingConSmax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        if config.consmax_learnable_beta:
            # learnable 'xmax' is beta
            self.beta = nn.Parameter(torch.Tensor([config.consmax_initial_beta]))
        else:
            self.beta = config.consmax_initial_beta

        if config.consmax_learnable_gamma:
            # denominator is gamma
            self.gamma = nn.Parameter(torch.Tensor([config.consmax_initial_gamma]))
        else:
            self.gamma = config.consmax_initial_gamma

        if config.consmax_use_euler_base:
            self.consmax_base = math.e
        else:
            self.consmax_base = config.consmax_base

        self.div_by_seq_len = config.div_by_seq_len

        # ConSmax saturation is like ReLU6 but happens where e^x normally would overflow
        # Since we're subtracting x by beta, we only need to guard at "beta + x_sat_value)
        # Note: for e^x this is around 11 for fp16 precision
        self.x_sat = config.consmax_saturation + config.consmax_initial_beta

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < (sat_point)
        # flat section:           (sat_point) <= x < inf

        # Exponential section
        exponential_piece = torch.where(
            (x < (self.x_sat)),
            torch.pow(self.consmax_base, x - self.beta),
            torch.tensor(0.0, device=x.device))

        # flat section
        flat_piece = torch.where(x >= (self.x_sat), torch.tensor(self.x_sat, device=x.device), torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (exponential_piece + flat_piece)/self.gamma

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# Merging of ConSmax body for gradient prop and Polymax head for numerical stability
class ExpPolymax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()

        self.dim = dim

        self.div_by_seq_len = config.div_by_seq_len

        # Base selection
        if config.exppolymax_use_euler_base:
            self.exppolymax_base = math.e
        else:
            self.exppolymax_base = config.exppolymax_base

        self.y_intercept = config.exppolymax_y_intercept # where the graph crosses y-axis
        self.power = config.exppolymax_power
        self.divisor = config.exppolymax_divisor
        # Assumes Euler Base:
        # Shift of x to move poly portion forward to obtain continuous derivative at x=0
        # derivative of poly at 0 should equal a^0
        # d(x^n + y-int) = d(a^x|x=0) = ln(a) * a^0 = ln(a)
        # n * x^(n-1) = ln(a)
        # x = (ln(a) * ( 1 / n )) ** (1/(n-1))
        # Note: if n==1 (straight line) match is already attained, and calculation would nan, so test this case first
        if config.exppolymax_power == 1.0:
            # Note: this only works with y=x and e^x, since we'd have to implement a multiplier or shift teh exponent otherwise.
            self.x_derivative_match_shift = 0
        elif config.exppolymax_use_euler_base:
            # ln(e) = 1
            self.x_derivative_match_shift = (1.0 / config.exppolymax_power)**(1/(config.exppolymax_power - 1))
        else:
            # ln(a) must be calculated, note torch.log is the natural log 'ln'
            self.x_derivative_match_shift = (torch.log(config.exppolymax_base) * (1.0 / config.exppolymax_power))**(1/(config.exppolymax_power - 1))

    def forward(self, x):
        # Overview:
        # exponential section:    -inf < x < 0
        # Polynomial section:     0 < x < inf

        # Exponential section
        exponential_piece = torch.where((x < 0), torch.pow(self.exppolymax_base, x), torch.tensor(0.0, device=x.device))

        # Polynomial section
        poly_piece = torch.where(x >= 0, (x + self.x_derivative_match_shift)**self.power + self.y_intercept, torch.tensor(0.0, device=x.device))

        # Combine sections
        result = (poly_piece + exponential_piece)/self.divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


# SigSoftmax from https://arxiv.org/abs/1805.10829
class SigSoftmax(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim

        # Set the base of the exponent
        if config.sigsoftmax_use_euler_base:
          self.sigsoftmax_base = math.e
        else:
          # custom base
          self.sigsoftmaxmax_base = config.sigsoftmax_base

    def forward(self, inputs):

        # Set exponent
        exp_x = torch.pow(self.sigsoftmax_base, inputs)

        # Similarly set sigmoid approximation
        sig_x = 1 / (1 + torch.pow(self.sigsoftmax_base, -inputs))

        # calculation of numerator and denominator
        numerator = exp_x * sig_x
        denominator = torch.sum(exp_x * sig_x, dim=self.dim, keepdim=True)

        return numerator / denominator

class SigmoidMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()
        self.sigmoidmax_divisor = config.sigmoidmax_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.sigmoid(x) / self.sigmoidmax_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class ReLUMax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.relumax = nn.ReLU()
        self.relumax_divisor = config.relumax_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.relumax(x) / self.relumax_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class ReLU2Max(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.relu2max_divisor = config.relu2max_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = torch.relu(x) ** 2 / self.relu2max_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


class Gelumax(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softshrink = nn.GELU()
        self.softshrink_attn_divisor = config.softshrink_attn_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        result = self.softshrink(x) / self.softshrink_attn_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


class Softshrink(nn.Module):
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softshrink = nn.Softshrink(lambd=config.softshrink_attn_lambda)
        self.softshrink_attn_divisor = config.softshrink_attn_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        result = self.softshrink(x) / self.softshrink_attn_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

class Softplus(nn.Module):
    """ Softmax variant based on arxiv 1805.10829 with added handles for base """
    def __init__(self, config, dim=-1):
        super().__init__()
        self.dim = dim
        self.softplus = nn.Softplus()
        self.softplus_divisor = config.softplus_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):

        result = self.softplus(x) / self.softplus_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result


class Squareplus(nn.Module):
    """Squareplus activation function.
       This is a computation friendly version of softplus
       source: https://arxiv.org/abs/2112.11687
    """

    def __init__(self, config, dim=-1, b=4.0*math.log(2)**2):
        super().__init__()
        self.b = b
        self.squareplus_divisor = config.squareplus_divisor
        self.div_by_seq_len = config.div_by_seq_len

    def forward(self, x):
        result = 0.5 * (x + torch.sqrt(x**2 + self.b)) / self.squareplus_divisor

        # divide by sequence length
        if self.div_by_seq_len:
            seq_len = x.shape[self.dim]
            result = result / seq_len

        return result

# ------------------------------------------------------------------------- #
#  PFLA‑Softmax  –  two interpolation modes (linear vs. quadratic)          #
# ------------------------------------------------------------------------- #
class PFLASoftmax(nn.Module):
    """
    A spline‑based activation whose control points are stored in √y‑space.

    • **linear**   (default)  –  square the knot values **once**, then do
      piece‑wise *linear* interpolation on the squared knots.

    • **quadratic**          –  first linearly interpolate in √y‑space,
      square the interpolated value, **then multiply the original x input**
      by that squared scale.  The extra √•² and × x makes the mapping
      effectively quadratic between knots.

    After either variant we apply the normalisation described earlier
    (classic Σy + OBO  *or*  learned γ).
    """
    def __init__(self, config, dim: int = -1):
        super().__init__()
        self.dim = dim

        # ---------------- user‑selectable interpolation mode ---------------
        self.mode = config.pfla_softmax_mode         # 'linear' | 'quadratic'

        # ---------------- knot generation (unchanged) ----------------------
        n            = config.pfla_softmax_num_points
        self.x_left  = config.pfla_softmax_left_bound
        self.x_right = config.pfla_softmax_right_bound
        learn_x      = config.pfla_softmax_learn_x
        learn_y      = config.pfla_softmax_learn_y
        density      = config.pfla_softmax_density
        act_name     = config.pfla_softmax_init_activation.lower()

        if density == "linear":
            x_init = torch.linspace(self.x_left, self.x_right, n + 2)[1:-1]
        elif density == "quad":
            lin = torch.linspace(-1, 1, n + 2)[1:-1]
            x_init = torch.sign(lin) * (lin.abs() ** 2)
            x_init *= max(abs(self.x_left), self.x_right)
        elif density == "exp":
            lin = torch.linspace(-1, 1, n + 2)[1:-1]
            x_init = torch.sign(lin) * (torch.exp(lin.abs()) - 1) / (math.e - 1)
            x_init *= max(abs(self.x_left), self.x_right)
        else:
            raise ValueError(f"Unknown density '{density}'")

        if learn_x:
            self.x_vals = nn.Parameter(x_init)
        else:
            self.register_buffer("x_vals", x_init)

        # √y initialisation from a reference activation
        if act_name not in activation_dictionary:
            raise ValueError(f"Unknown init activation '{act_name}'")
        ref_act = activation_dictionary[act_name](config)          # GELU etc.
        y_ref   = ref_act(x_init).detach().clamp(min=1e-6)         # ≥0
        y_param_init = torch.sqrt(y_ref)

        if learn_y:
            self.y_vals = nn.Parameter(y_param_init)
        else:
            self.register_buffer("y_vals", y_param_init)

        # -------------- normalisation controls (as before) -----------------
        self.use_learned_divisor = config.pfla_softmax_use_learned_divisor
        self.use_obo             = config.pfla_softmax_use_obo
        self.use_learned_obo     = config.pfla_softmax_use_learned_obo
        self.obo_init_val        = config.pfla_softmax_obo

        if self.use_learned_divisor:
            self._gamma_raw = nn.Parameter(torch.tensor(config.pfla_softmax_gamma_init))
            self._sp_gamma = nn.ReLU()

        if self.use_learned_obo:
            self._obo_raw = nn.Parameter(torch.tensor(self.obo_init_val))
            self._sp_obo = torch.exp()

    # ---------------------------------------------------------------------
    def _linear_interp(self, y_knots: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Safe piece‑wise linear interpolation that never indexes beyond the
        last knot.  Works on any *y_knots* (squared or √y) tensor.
        """
        N   = self.x_vals.numel()                  # number of knots
        idx = torch.searchsorted(self.x_vals, x).clamp(0, N - 1)

        # capped next index – never exceeds N‑1
        idx_next = torch.clamp(idx + 1, max=N - 1)

        x_k   = self.x_vals[idx]
        x_k1  = self.x_vals[idx_next]
        y_k   = y_knots[idx]
        y_k1  = y_knots[idx_next]

        # avoid 0‑division when idx_next == idx  (happens at the last knot)
        denom = torch.clamp(x_k1 - x_k, min=1e-6)
        slope = (y_k1 - y_k) / denom
        return y_k + slope * (x - x_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6                      # small constant for numerical safety

        # ───────── MASK 1: replace ±inf inputs by zero for safe math ─────────
        finite_mask = torch.isfinite(x)
        x_safe      = torch.where(finite_mask, x, torch.zeros_like(x))

        # -------- obtain y_pos depending on interpolation mode ---------------
        if self.mode == "linear":
            y_knots_sq = self.y_vals ** 2
            y_pos = self._linear_interp(y_knots_sq, x_safe)

        elif self.mode == "quadratic":
            y_sqrt  = self._linear_interp(self.y_vals, x_safe)
            y_scale = y_sqrt ** 2
            y_pos   = x_safe * y_scale
        else:
            raise ValueError(f"Unknown pfla_softmax_mode '{self.mode}'")

        # ───────── MASK 2: hard‑zero the originally masked positions ─────────
        y_pos = torch.where(finite_mask, y_pos, torch.zeros_like(y_pos))

        # ------------------------ normalisation ------------------------------
        if self.use_learned_divisor:
            gamma  = self._sp_gamma(self._gamma_raw) + eps        # ← EPS
            out = y_pos / gamma
        else:
            denom = y_pos.sum(dim=self.dim, keepdim=True) + eps   # ← EPS
            if self.use_obo:
                if self.use_learned_obo:
                    obo = self._sp_obo(self._obo_raw)
                else:
                    obo = self._sp_obo(self.obo_init_val)
                out = y_pos / (denom + obo)
            else:
                out = y_pos / (denom)

        return out




# Note: we use the built in library for regular softmax
softmax_dictionary = {
    "consmax": ConSmax,
    "consmax_v2": ConSmaxV2,
    "consmax_quan": ConSmaxQuan,
    "saturatingconsmax": SaturatingConSmax,
    "vpolymax": VPolymax,
    "polymax": Polymax,
    "exppolymax": ExpPolymax,
    "softermax": Softermax,
    "strongermax": Strongermax,
    "sigsoftmax": SigSoftmax,
    "relumax": ReLUMax,
    "relu2max": ReLU2Max,
    "sigmoidmax": SigmoidMax,
    "softshrink": Softshrink,
    "gelumax": Gelumax,
    "softplus": Softplus,
    "squareplus": Squareplus,
    "pfla_softmax": PFLASoftmax,
}

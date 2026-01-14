# variations/activation_variations.py
import torch
import torch.nn as nn

# Custom Activation Variations
class SquaredReLU(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)

class SquaredGELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return torch.sign(x)*self.gelu(x)*self.gelu(x)

class GELUShifted(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.learnable_shift = config.shifted_gelu_learnable_shift
        self.initial_shift = config.shifted_gelu_initial_shift

        # Initialize shift parameter, either fixed or learnable
        if self.learnable_shift:
            self.shift = nn.Parameter(torch.tensor(self.initial_shift))
        else:
            self.register_buffer("shift", torch.tensor(self.initial_shift))

        self.gelu = nn.GELU()

    def forward(self, x):
        # Apply the shifted GELU activation
        return self.gelu(x - self.shift)


class PiecewiseLearnableActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_of_points = config.pla_num_points
        self.left_bound = config.pla_left_bound
        self.right_bound = config.pla_right_bound

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(torch.linspace(self.left_bound, self.right_bound, self.num_of_points + 2)[1:-1])  # Exclude -2 and 2

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Create a piecewise linear function
        result = torch.zeros_like(x)

        # Leftmost segment (-2 <= x < x_vals[0])
        result = torch.where(x < self.x_vals[0], 0, result)  # x = -2 -> y = 0

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where((x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result)

        # Segment before the last (x_vals[-1] <= x < 2)
        slope = (self.right_bound - self.y_vals[-1]) / (self.right_bound - self.x_vals[-1])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where((x >= self.x_vals[-1]) & (x < self.right_bound), segment, result)

        # Rightmost segment (x >= 2)
        result = torch.where(x >= self.right_bound, x, result)  # x = y for x >= 2

        return result

class PiecewiseFullyLearnableActivation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_of_points = config.pfla_num_points
        self.left_bound = config.pfla_left_bound
        self.right_bound = config.pfla_right_bound

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(torch.linspace(self.left_bound, self.right_bound, self.num_of_points + 2)[1:-1])  # Exclude left_bound and right_bound

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Create a piecewise linear function
        result = torch.zeros_like(x)

        # Leftmost segment (left_bound <= x < x_vals[0])
        result = torch.where(x < self.x_vals[0], 0, result)  # x = left_bound -> y = 0

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where((x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result)

        # Segment before the last (x_vals[-1] <= x < right_bound)
        slope = (self.right_bound - self.y_vals[-1]) / (self.right_bound - self.x_vals[-1])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where((x >= self.x_vals[-1]) & (x < self.right_bound), segment, result)

        # Rightmost segment (x >= right_bound)
        result = torch.where(x >= self.right_bound, x, result)  # x = y for x >= right_bound

        return result


class PiecewiseFullyLearnableActivationLearnedEnds(nn.Module):
    def __init__(self, config, left_bound=-10, right_bound=10, num_of_points=30):
        super().__init__()
        self.num_of_points = config.pfla_num_points
        self.left_bound = config.pfla_left_bound
        self.right_bound = config.pfla_right_bound

        # Initialize learnable parameters for x and y values of intermediate points
        self.x_vals = nn.Parameter(
            torch.linspace(self.left_bound, self.right_bound, self.num_of_points + 2)[1:-1]
        )  # Exclude left_bound and right_bound

        # Initialize y_vals using GELU output for corresponding x_vals
        gelu = nn.GELU()
        self.y_vals = nn.Parameter(gelu(self.x_vals))

    def forward(self, x):
        # Initialize result with the scalar y-value at the leftmost x
        result = torch.full_like(x, self.y_vals[0].item())

        # Intermediate segments (x_vals[i] <= x < x_vals[i+1])
        for i in range(self.num_of_points - 1):
            slope = (self.y_vals[i + 1] - self.y_vals[i]) / (self.x_vals[i + 1] - self.x_vals[i])
            intercept = self.y_vals[i] - slope * self.x_vals[i]
            segment = slope * x + intercept
            result = torch.where(
                (x >= self.x_vals[i]) & (x < self.x_vals[i + 1]), segment, result
            )

        # Extrapolate using the last segment's slope for x >= x_vals[-1]
        slope = (self.y_vals[-1] - self.y_vals[-2]) / (self.x_vals[-1] - self.x_vals[-2])
        intercept = self.y_vals[-1] - slope * self.x_vals[-1]
        segment = slope * x + intercept
        result = torch.where(x >= self.x_vals[-1], segment, result)

        return result


class LearnedSplineActivation(nn.Module):

    def __init__(self, config, num_knots=10, init_x_range=(-5, 5)):
        super().__init__()
        self.num_knots = config.lsa_num_knots

        # Initialize learnable x_vals and y_vals
        x_init = torch.linspace(init_x_range[0], init_x_range[1], num_knots)

        # Create an instance of GELU
        gelu = nn.GELU()
        y_init = gelu(x_init)  # Use the GELU instance here

        self.x_vals = nn.Parameter(x_init)
        self.y_vals = nn.Parameter(y_init)

    def forward(self, x):
        # Compute spline coefficients
        coeffs = self._compute_spline_coefficients(self.x_vals, self.y_vals)

        # Evaluate spline at input x
        result = self._evaluate_spline(x, self.x_vals, coeffs)
        return result

    def _compute_spline_coefficients(self, x_vals, y_vals):
        """
        Compute the coefficients for cubic spline interpolation.
        """
        n = x_vals.size(0)
        h = x_vals[1:] - x_vals[:-1]  # Intervals between x knots

        # Set up the system of equations
        # Compute the differences in y_vals
        delta = (y_vals[1:] - y_vals[:-1]) / h

        # Construct the tridiagonal matrix
        A = torch.zeros(n, n, device=x_vals.device)
        rhs = torch.zeros(n, device=x_vals.device)

        A[0, 0] = 1  # Natural spline boundary condition
        A[-1, -1] = 1  # Natural spline boundary condition

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3 * (delta[i] - delta[i - 1])

        # Solve for the second derivatives (M)
        M = torch.linalg.solve(A, rhs)

        # Compute spline coefficients for each interval
        coeffs = []
        for i in range(n - 1):
            h_i = h[i]
            a = y_vals[i]
            b = delta[i] - h_i * (2 * M[i] + M[i + 1]) / 3
            c = M[i] / 2
            d = (M[i + 1] - M[i]) / (6 * h_i)
            coeffs.append((a, b, c, d))

        # Stack coefficients for vectorized computation
        coeffs = tuple(torch.stack(c) for c in zip(*coeffs))
        return coeffs

    def _evaluate_spline(self, x, x_vals, coeffs):
        """
        Evaluate the spline at the input x using the computed coefficients.
        """
        a, b, c, d = coeffs  # Unpack coefficients
        n = x_vals.size(0)

        # Find the interval each x belongs to
        indices = torch.searchsorted(x_vals, x) - 1
        indices = torch.clamp(indices, 0, n - 2)

        x_k = x_vals[indices]
        dx = x - x_k

        # Compute the spline value
        result = a[indices] + b[indices] * dx + c[indices] * dx**2 + d[indices] * dx**3
        return result


class ActivationWrapper(nn.Module):
    """Base wrapper class for PyTorch activation functions"""
    def __init__(self, activation_class, config=None):
        super().__init__()
        self.activation = activation_class()

    def forward(self, x):
        return self.activation(x)

# _Config classes for native PyTorch activations
class CELU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.CELU, config)

class ELU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.ELU, config)

class GELU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.GELU, config)

class GLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.GLU, config)

class LeakyReLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.LeakyReLU, config)

class Mish_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Mish, config)

class PReLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.PReLU, config)

class ReLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.ReLU, config)

class ReLU6_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.ReLU6, config)

class RReLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.RReLU, config)

class SELU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.SELU, config)

class Sigmoid_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Sigmoid, config)

class SiLU_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.SiLU, config)

class Softplus_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Softplus, config)

class Softsign_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Softsign, config)

class Softshrink_Config(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        lambd = getattr(config, "softshrink_lambda", 0.5) if config is not None else 0.5
        self.activation = nn.Softshrink(lambd=lambd)

    def forward(self, x):
        return self.activation(x)

class Tanh_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Tanh, config)

class Identity_Config(ActivationWrapper):
    def __init__(self, config=None):
        super().__init__(nn.Identity, config)


activation_dictionary = {
    "celu": CELU_Config,
    "elu": ELU_Config,
    "gelu": GELU_Config,
    "gelu_shifted": GELUShifted,
    "glu": GLU_Config,
    "leaky_relu": LeakyReLU_Config,
    "mish": Mish_Config,
    "piecewise": PiecewiseLearnableActivation,
    "pfla": PiecewiseFullyLearnableActivation,
    "pfla_le": PiecewiseFullyLearnableActivationLearnedEnds,
    "learned_spline": LearnedSplineActivation,
    "prelu": PReLU_Config,
    "relu": ReLU_Config,
    "relu6": ReLU6_Config,
    "rrelu": RReLU_Config,
    "selu": SELU_Config,
    "sigmoid": Sigmoid_Config,
    "silu": SiLU_Config,
    "softplus": Softplus_Config,
    "softsign": Softsign_Config,
    "softshrink": Softshrink_Config,
    "squared_relu": SquaredReLU,
    "squared_gelu": SquaredGELU,
    "tanh": Tanh_Config,
    "identity": Identity_Config,
}


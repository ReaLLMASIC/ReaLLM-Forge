"""Independent tables and linked SLERP curves; one module per numeric column."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import blend_scores, lookup, weighted_lookup


ANGLE_EPS = 1e-3


def quantize(values, minimum, maximum, levels):
    """Nearest-level quantization, with explicit saturation outside the range."""
    unit = ((values - minimum) / (maximum - minimum)).clamp(0, 1)
    return (unit * (levels - 1)).round().long()


def dequantize(indices, minimum, maximum, levels):
    return indices.float() * ((maximum - minimum) / (levels - 1)) + minimum


def segment_coefficients(unit, angles):
    """Map normalized positions to shared adjacent knots and SLERP weights."""
    count = angles.shape[0] + 1
    position = unit.clamp(0, 1) * (count - 1)
    left = position.floor().long().clamp(0, count - 2)
    local = position - left
    theta = angles[left]
    denominator = theta.sin()
    a = ((1 - local) * theta).sin() / denominator
    b = (local * theta).sin() / denominator
    return left, left + 1, a, b


class ColumnGeometry(nn.Module):
    """A column's input geometry and optional independent classifier geometry.

    Tied SLERP uses a normalized curve on a common learned radius. Untied SLERP
    copies it once, then learns unconstrained output knots and independent
    coefficient angles. That relaxed head is not necessarily spherical.
    """

    def __init__(self, spec, width):
        super().__init__()
        self.spec = dict(spec)
        self.kind = spec["kind"]
        self.tied = spec["tied"]
        self.backend = spec["backend"]
        self.levels = spec["levels"]
        self.minimum, self.maximum = spec["minimum"], spec["maximum"]
        self.width = width
        if self.kind == "independent":
            self.input_table = nn.Parameter(torch.randn(self.levels, width) * 0.3)
            if not self.tied:
                self.output_table = nn.Parameter(self.input_table.detach().clone())
        else:
            self.anchors = nn.Parameter(torch.randn(spec["knots"], width))
            self.log_radius = nn.Parameter(torch.tensor(0.5 * math.log(width)))
            if not self.tied:
                knots, angles = self.input_geometry()
                self.output_knots = nn.Parameter(knots.detach().clone())
                self.raw_angles = nn.Parameter(self._inverse_angles(angles.detach()))
        self.register_buffer("level_units", torch.linspace(0, 1, self.levels), persistent=False)

    @staticmethod
    def _inverse_angles(angles):
        fraction = ((angles - ANGLE_EPS) / (math.pi - 2 * ANGLE_EPS)).clamp(1e-6, 1 - 1e-6)
        return torch.logit(fraction)

    def input_geometry(self):
        unit = F.normalize(self.anchors, dim=-1)
        cosine = (unit[:-1] * unit[1:]).sum(-1).clamp(-1, 1)
        tangent = unit[1:] - cosine[:, None] * unit[:-1]
        angles = torch.atan2(tangent.norm(dim=-1), cosine).clamp(ANGLE_EPS, math.pi - ANGLE_EPS)
        return unit * self.log_radius.exp(), angles

    def output_geometry(self):
        if self.tied:
            return self.input_geometry()
        angles = ANGLE_EPS + (math.pi - 2 * ANGLE_EPS) * self.raw_angles.sigmoid()
        return self.output_knots, angles

    @torch.no_grad()
    def reset_output_from_input(self):
        """Call AFTER loading input weights when converting tied to untied."""
        if self.tied:
            raise ValueError("reset_output_from_input requires an untied column")
        if self.kind == "independent":
            self.output_table.copy_(self.input_table)
        else:
            knots, angles = self.input_geometry()
            self.output_knots.copy_(knots)
            self.raw_angles.copy_(self._inverse_angles(angles))

    def encode(self, values):
        if self.kind == "independent":
            ids = quantize(values, self.minimum, self.maximum, self.levels)
            return lookup(self.input_table, ids, backend=self.backend)
        unit = (values - self.minimum) / (self.maximum - self.minimum)
        knots, angles = self.input_geometry()
        left, right, a, b = segment_coefficients(unit, angles)
        return weighted_lookup(knots, left, right, a, b, backend=self.backend)

    def logits(self, hidden):
        # Head tensors remain FP32 under autocast; the caller's CUDA TF32
        # matmul policy still applies (this does not promise strict IEEE math).
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            if self.kind == "independent":
                table = self.input_table if self.tied else self.output_table
                return F.linear(hidden, table.float())
            knots, angles = self.output_geometry()
            left, right, a, b = segment_coefficients(self.level_units.float(), angles.float())
            scores = F.linear(hidden, knots.float())
            return blend_scores(scores, left, right, a, b, backend=self.backend)

    def dense_output_table(self):
        """Reference/debug materialization; inference projects only onto K knots."""
        if self.kind == "independent":
            return self.input_table if self.tied else self.output_table
        knots, angles = self.output_geometry()
        left, right, a, b = segment_coefficients(self.level_units, angles)
        return a[:, None] * knots[left] + b[:, None] * knots[right]

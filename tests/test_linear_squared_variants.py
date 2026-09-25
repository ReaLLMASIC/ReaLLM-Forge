from types import SimpleNamespace

import pytest
import torch

from variations.activation_variations import SquaredReLULinear
from variations.softmax_variations import ReLU2MaxLinear


def test_squared_relu_linear_matches_square_and_tangent():
    activation = SquaredReLULinear(
        SimpleNamespace(squared_relu_linear_cutoff=2.0)
    )
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0], requires_grad=True)

    output = activation(x)

    torch.testing.assert_close(output, torch.tensor([0.0, 0.0, 1.0, 4.0, 8.0]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.0, 2.0, 4.0, 4.0]))


def test_relu2max_linear_applies_divisors():
    activation = ReLU2MaxLinear(
        SimpleNamespace(
            relu2max_divisor=2.0,
            relu2max_linear_cutoff=2.0,
            div_by_seq_len=True,
        )
    )

    output = activation(torch.tensor([[1.0, 2.0, 3.0, -1.0]]))

    torch.testing.assert_close(output, torch.tensor([[0.125, 0.5, 1.0, 0.0]]))


@pytest.mark.parametrize(
    ("variant", "attribute"),
    [
        (SquaredReLULinear, "squared_relu_linear_cutoff"),
        (ReLU2MaxLinear, "relu2max_linear_cutoff"),
    ],
)
def test_linear_squared_variants_reject_nonpositive_cutoffs(variant, attribute):
    config = {
        attribute: 0.0,
        "relu2max_divisor": 1.0,
        "div_by_seq_len": False,
    }
    with pytest.raises(ValueError, match="must be greater than zero"):
        variant(SimpleNamespace(**config))

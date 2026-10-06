import pytest
import torch

from gpt_conf import GPTConfig
from variations.mlp_variations import HadamardMLP, _normalized_hadamard, get_mlp_instance


def test_hadamard_matrix_is_orthonormal():
    matrix = _normalized_hadamard(8)
    torch.testing.assert_close(matrix @ matrix.t(), torch.eye(8))


def test_hadamard_mlp_forward_backward_and_dispatch():
    torch.manual_seed(7)
    config = GPTConfig(
        n_embd=10,
        mlp_variant="hadamard",
        hadamard_mlp_factor_size=4,
        hadamard_mlp_stages=3,
        hadamard_mlp_gain_rank=3,
        dropout=0.0,
    )
    mlp = get_mlp_instance(config)
    assert isinstance(mlp, HadamardMLP)
    assert mlp.padded_size == 16

    inputs = torch.randn(2, 5, 10, requires_grad=True)
    output = mlp(inputs)
    assert output.shape == inputs.shape
    output.square().mean().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in mlp.parameters())


def test_hadamard_mlp_initialization_and_validation():
    config = GPTConfig(n_embd=8, hadamard_mlp_factor_size=4)
    mlp = HadamardMLP(config)
    torch.testing.assert_close(mlp.gain_u, torch.zeros_like(mlp.gain_u))
    torch.testing.assert_close(mlp.diagonals[-1], torch.full((16,), 0.02))

    config.hadamard_mlp_factor_size = 3
    with pytest.raises(ValueError, match="power of two"):
        HadamardMLP(config)

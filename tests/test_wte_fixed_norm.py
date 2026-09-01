import math

import pytest

torch = pytest.importorskip("torch")

from gpt_conf import GPTConfig
from model import GPT


def test_fixed_norm_initialization_and_reprojection():
    config = GPTConfig(
        block_size=4,
        vocab_size=14,
        n_layer=1,
        n_head=1,
        n_kv_group=1,
        n_embd=3,
        wte_fixed_norm=True,
    )
    model = GPT(config)
    expected = torch.full((config.vocab_size,), math.sqrt(config.n_embd))

    assert torch.allclose(model.transformer.wte.weight.norm(dim=-1), expected)

    with torch.no_grad():
        model.transformer.wte.weight[0].mul_(0.25)
        model.transformer.wte.weight[1].mul_(3.0)
    model.reproject_token_embeddings()

    assert torch.allclose(model.transformer.wte.weight.norm(dim=-1), expected)


def test_fixed_norm_custom_radius():
    config = GPTConfig(
        block_size=4,
        vocab_size=4,
        n_layer=1,
        n_head=1,
        n_kv_group=1,
        n_embd=3,
        wte_fixed_norm=True,
        wte_fixed_norm_value=2.5,
    )
    model = GPT(config)

    assert torch.allclose(model.transformer.wte.weight.norm(dim=-1), torch.full((4,), 2.5))

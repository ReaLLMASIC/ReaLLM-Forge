import torch

from gpt_conf import GPTConfig
from model import GPT
from variations.attention_residual_variations import FullAttentionResidual


def test_zero_queries_start_as_equal_weight_average():
    mixer = FullAttentionResidual(n_destinations=1, n_embd=2)
    sources = [
        torch.tensor([[[1.0, 3.0]]]),
        torch.tensor([[[5.0, 7.0]]]),
    ]

    result = mixer(sources, destination=0)

    torch.testing.assert_close(result, torch.tensor([[[3.0, 5.0]]]))


def test_full_attention_residual_model_forward_and_backward():
    config = GPTConfig(
        block_size=4,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_group=2,
        n_embd=8,
        dropout=0.0,
        attention_residual_variant="full",
    )
    model = GPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(tokens, targets)
    loss.backward()

    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert model.attention_residual.queries.shape == (2 * config.n_layer + 1, config.n_embd)
    assert model.attention_residual.queries.grad is not None

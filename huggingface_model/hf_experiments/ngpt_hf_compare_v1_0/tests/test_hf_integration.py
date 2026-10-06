"""These tests need Transformers, but do NOT require a network connection."""
import pytest
import torch
pytest.importorskip("transformers")
from transformers import AutoConfig, AutoModelForCausalLM
from hf_model import NGPTConfig


@pytest.mark.parametrize("variant", ["gpt", "ngpt"])
def test_hf_save_load_and_standard_label_shift(tmp_path, variant):
    c = NGPTConfig(vocab_size=32, width=16, layers=1, heads=2, context=16, variant=variant)
    m = AutoModelForCausalLM.from_config(c).eval()
    x = torch.randint(32, (2, 10))
    output = m(input_ids=x, labels=x)
    torch.testing.assert_close(output.loss, m.next_token_loss(x))
    path = tmp_path/variant
    m.save_pretrained(path, safe_serialization=True)
    loaded = AutoModelForCausalLM.from_pretrained(path).eval()
    torch.testing.assert_close(output.logits, loaded(x).logits, atol=1e-6, rtol=1e-6)
    assert AutoConfig.from_pretrained(path).variant == variant
    assert loaded.get_input_embeddings().weight is not loaded.get_output_embeddings().weight
    assert loaded.net.c.context == 16

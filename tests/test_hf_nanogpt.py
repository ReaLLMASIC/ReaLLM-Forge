import math
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hf_model import NanoGPTConfig, NanoGPTForCausalLM
from hf_model.modeling_nanogpt import _apply_rope
from hf_model.triton_relu2max import TRITON_AVAILABLE, triton_relu2max
from scripts.compare_hf_relu2max import evaluation_strategy_argument
from scripts.compare_hf_relu2max import tokenize_dataset


def test_direct_comparison_script_can_import_local_model():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "compare_hf_relu2max.py"), "--help"],
        cwd=repo_root.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ReLU2Max/softmax" in result.stdout


def test_training_arguments_accept_selected_evaluation_keyword(tmp_path):
    from transformers import TrainingArguments

    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        report_to=[],
        **evaluation_strategy_argument(),
    )
    strategy = getattr(arguments, "eval_strategy", None)
    if strategy is None:
        strategy = arguments.evaluation_strategy
    assert strategy.value == "steps"


def test_relu2max_auto_falls_back_on_cpu():
    model = NanoGPTForCausalLM(tiny_config(relu2max_accelerator="auto")).eval()
    ids = torch.randint(0, 31, (1, 4))
    with torch.no_grad():
        output = model(ids)
    assert torch.isfinite(output.logits).all()


def test_pretraining_packing_appends_eos(monkeypatch):
    from types import SimpleNamespace

    class TinyTokenizer:
        eos_token_id = 9

        def __call__(self, texts, **kwargs):
            del kwargs
            return {"input_ids": [[int(text)] for text in texts], "attention_mask": [[1] for _ in texts]}

    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_dict({"text": ["1", "2", "3", "4"]})
    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
    monkeypatch.setattr("scripts.compare_hf_relu2max.load_dataset", lambda *args: dataset_dict)
    args = SimpleNamespace(dataset="unused", dataset_config=None, train_samples=0, eval_samples=0,
                           text_column="text", pack_text=True, block_size=4, append_eos=True, seed=1)
    train, _ = tokenize_dataset(args, TinyTokenizer())
    assert train[0]["input_ids"] == [1, 9, 2, 9]


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_relu2max_matches_torch_forward_and_backward(dtype):
    values = torch.randn(4099, device="cuda", dtype=dtype, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(values)
    actual = triton_relu2max(values, 256.0)
    expected = torch.relu(reference_values).square() / 256.0
    tolerance = 2e-3 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(values.grad, reference_values.grad, atol=tolerance, rtol=tolerance)


def tiny_config(**kwargs):
    values = dict(vocab_size=31, max_position_embeddings=16, hidden_size=16,
                  intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=4, rope_length=4)
    values.update(kwargs)
    return NanoGPTConfig(**values)


@pytest.mark.parametrize("normalizer", ["relu2max", "softmax"])
def test_causal_lm_forward_backward_and_roundtrip(tmp_path, normalizer):
    model = NanoGPTForCausalLM(tiny_config(attention_normalizer=normalizer))
    input_ids = torch.randint(0, 31, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 8, 31)
    assert torch.isfinite(output.loss)
    output.loss.backward()

    model.save_pretrained(tmp_path)
    loaded = NanoGPTForCausalLM.from_pretrained(tmp_path).eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits)


def test_qk_scale_and_partial_interleaved_rope():
    config = tiny_config()
    assert config.qk_norm_scale_init == pytest.approx(math.log2(16 * 16 - 16))
    x = torch.tensor([[[[1.0, 0.0, 3.0, 4.0, 9.0, 10.0]]]])
    result = _apply_rope(x, torch.tensor([[1]]), 4, 10000.0)
    torch.testing.assert_close(result[..., 4:], x[..., 4:])
    torch.testing.assert_close(result[..., 0], torch.cos(torch.tensor(1.0)))
    torch.testing.assert_close(result[..., 1], torch.sin(torch.tensor(1.0)))


def test_generation_cache_matches_full_forward():
    model = NanoGPTForCausalLM(tiny_config(attention_normalizer="relu2max")).eval()
    ids = torch.randint(0, 31, (1, 6))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)

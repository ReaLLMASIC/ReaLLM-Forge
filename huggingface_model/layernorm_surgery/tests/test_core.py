import torch

from huggingface_model.layernorm_surgery.core import effective_gain, merge_gain_into_head, set_effective_gain, symmetric_quantize, threshold_gain
from huggingface_model.layernorm_surgery.sweep import float_range


def test_threshold_is_magnitude_based_and_strict():
	gain = torch.tensor([-0.2, -0.09, 0.1, 0.4])
	assert threshold_gain(gain, 0.1).tolist() == [-0.2, 0.0, 0.1, 0.4]


def test_merge_groups_zero_channels_and_reports_dot_product_savings():
	head = torch.arange(12, dtype=torch.float32).reshape(3, 4)
	stats = merge_gain_into_head(head, torch.tensor([0.01, 2.0, -3.0, 0.02]), 0.1)
	assert stats.permutation.tolist() == [1, 2, 0, 3]
	assert stats.merged_nonzero.shape == (3, 2)
	assert stats.shaved_parameters == 6
	assert stats.shaved_fraction == 0.5
	assert torch.equal(stats.merged_nonzero, head[:, [1, 2]] * torch.tensor([2.0, -3.0]))


def test_symmetric_quantization_and_ranges():
	values = torch.tensor([-1.0, -0.2, 0.3, 1.0])
	quantized = symmetric_quantize(values, 2)
	assert quantized.tolist() == [-1.0, 0.0, 0.0, 1.0]
	assert float_range(0, 0.3, 0.1) == [0, 0.1, 0.2, 0.3]


def test_gemma_rmsnorm_offset_is_converted_to_effective_gain():
	class GemmaRMSNorm:
		weight = torch.tensor([0.0, 0.5])
	module = GemmaRMSNorm()
	assert effective_gain(module).tolist() == [1.0, 1.5]
	set_effective_gain(module, torch.tensor([0.0, 2.0]))
	assert module.weight.tolist() == [-1.0, 1.0]

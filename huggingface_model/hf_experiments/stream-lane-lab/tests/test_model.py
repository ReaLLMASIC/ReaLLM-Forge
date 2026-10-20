"""Semantic checks: geometry, causal targets, sharing, and checkpoint contracts."""
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from lane_lab import LaneConfig, LaneModel
from lane_lab.geometry import ColumnGeometry, quantize, dequantize, segment_coefficients


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)


def tiny(columns=None, **kwargs):
    return LaneModel(LaneConfig(columns=columns, d_model=12, n_heads=2,
        n_layers=1, d_ff=20, context_length=8, **kwargs))


def spec(kind="slerp", knots=5, tied=True, **kwargs):
    return LaneConfig(columns=[dict(kind=kind, knots=knots, tied=tied, **kwargs)]).columns[0]


def test_mixed_vocabularies_ranges_and_equal_column_loss():
    columns = [dict(name="lookup", kind="independent", levels=9, minimum=-3., maximum=2.),
               dict(name="curve", kind="slerp", knots=3, tied=False, levels=17, minimum=10., maximum=20.)]
    model = tiny(columns)
    x = torch.stack((torch.rand(2, 5) * 5 - 3, torch.rand(2, 5) * 10 + 10), -1)
    labels = model.quantize(x.roll(-1, 1))
    labels[0, :3, 0] = -100
    output = model(x, labels=labels)
    assert [list(t.shape) for t in output.logits] == [[2, 5, 9], [2, 5, 17]]
    expected = torch.stack([F.cross_entropy(t.reshape(-1, t.shape[-1]), labels[..., i].reshape(-1))
                            for i, t in enumerate(output.logits)]).mean()
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert model.columns[0].input_table.grad.norm() > 0
    assert model.columns[1].anchors.grad.norm() > 0
    assert model.columns[1].output_knots.grad.norm() > 0
    assert model.columns[1].raw_angles.grad.norm() > 0


def test_causal_outputs_do_not_depend_on_future_inputs():
    model = tiny().eval()
    x = torch.rand(2, 8, 1) * 2 - 1
    changed = x.clone()
    changed[:, 4:] = torch.rand_like(changed[:, 4:]) * 2 - 1
    with torch.no_grad():
        a, b = model(x).logits[0], model(changed).logits[0]
    torch.testing.assert_close(a[:, :4], b[:, :4], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_forward_does_not_shift_aligned_labels_again():
    model = tiny([dict(kind="independent", levels=8)])
    stream = torch.tensor([[[-1.], [-.7], [.2], [.9]]])
    inputs, targets = stream[:, :-1], model.quantize(stream[:, 1:])
    result = model(inputs, labels=targets)
    manual = F.cross_entropy(result.logits[0].reshape(-1, 8), targets.reshape(-1))
    torch.testing.assert_close(result.loss, manual)


@pytest.mark.parametrize("knots", [2, 3, 5, 9])
def test_linked_slerp_exact_knots_continuity_and_radius(knots):
    geometry = ColumnGeometry(spec(knots=knots), width=16)
    learned, _ = geometry.input_geometry()
    knot_values = torch.linspace(-1, 1, knots)
    torch.testing.assert_close(geometry.encode(knot_values), learned, atol=2e-6, rtol=2e-6)
    curve = geometry.encode(torch.linspace(-1, 1, 257))
    torch.testing.assert_close(curve.norm(dim=-1), geometry.log_radius.exp().expand(257),
                               atol=2e-6, rtol=2e-6)
    if knots > 2:
        joins = knot_values[1:-1]
        torch.testing.assert_close(geometry.encode(joins - 1e-6), geometry.encode(joins + 1e-6),
                                   atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("knots", [2, 3, 5, 9])
@pytest.mark.parametrize("tied", [True, False])
def test_factorized_output_equals_dense_and_has_rank_at_most_knots(knots, tied):
    geometry = ColumnGeometry(spec(knots=knots, tied=tied, levels=31), width=16)
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    table = geometry.dense_output_table()
    factorized = geometry.logits(hidden)
    dense = F.linear(hidden, table)
    torch.testing.assert_close(factorized, dense, atol=4e-6, rtol=4e-6)
    assert int(torch.linalg.matrix_rank(table, atol=1e-4, rtol=1e-4)) <= knots
    torch.testing.assert_close(torch.autograd.grad(factorized.sum(), hidden, retain_graph=True)[0],
                               torch.autograd.grad(dense.sum(), hidden)[0], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("kind", ["independent", "slerp"])
def test_untied_warm_start_copies_without_alias_and_stays_trainable(kind):
    geometry = ColumnGeometry(spec(kind=kind, tied=False), width=16)
    if kind == "independent":
        assert geometry.input_table.data_ptr() != geometry.output_table.data_ptr()
        torch.testing.assert_close(geometry.input_table, geometry.output_table)
        with torch.no_grad(): geometry.input_table.add_(.1)
        assert not torch.allclose(geometry.input_table, geometry.output_table)
        geometry.reset_output_from_input()
        torch.testing.assert_close(geometry.input_table, geometry.output_table)
    else:
        original_knots, original_angles = geometry.input_geometry()
        output_knots, output_angles = geometry.output_geometry()
        assert geometry.anchors.data_ptr() != output_knots.data_ptr()
        torch.testing.assert_close(original_knots, output_knots)
        torch.testing.assert_close(original_angles, output_angles)
        with torch.no_grad(): geometry.anchors.add_(.2)
        assert not torch.allclose(geometry.input_geometry()[0], output_knots)
        geometry.reset_output_from_input()
        torch.testing.assert_close(geometry.input_geometry()[0], geometry.output_geometry()[0])
        torch.testing.assert_close(geometry.input_geometry()[1], geometry.output_geometry()[1])


def test_slerp_input_has_continuous_value_gradients():
    geometry = ColumnGeometry(spec(knots=5), width=16)
    values = torch.tensor([-.83, -.11, .22, .87], requires_grad=True)
    geometry.encode(values).square().mul(torch.arange(1, 17)).sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().min() > 1e-5
    assert torch.isfinite(geometry.anchors.grad).all()


def test_last_quantized_level_is_last_knot_not_out_of_bounds():
    left, right, a, b = segment_coefficients(torch.tensor([0., 1.]), torch.tensor([.8, 1.2, 1.5, 2.1]))
    assert left.tolist() == [0, 3]
    assert right.tolist() == [1, 4]
    torch.testing.assert_close(a, torch.tensor([1., 0.]))
    torch.testing.assert_close(b, torch.tensor([0., 1.]))


def test_quantization_saturates_and_reconstructs_level_centers():
    ids = quantize(torch.tensor([-100., -2., 0., 2., 100.]), -2., 2., 9)
    assert ids.tolist() == [0, 0, 4, 8, 8]
    levels = torch.arange(9)
    assert torch.equal(quantize(dequantize(levels, -2., 2., 9), -2., 2., 9), levels)


@pytest.mark.parametrize("mode", ["mean", "greedy", "sample"])
def test_forecast_stays_in_column_ranges_preserves_context_and_training_state(mode):
    model = tiny([dict(name="a", kind="independent", levels=9, minimum=-4., maximum=-2.),
                  dict(name="b", kind="slerp", tied=False, minimum=3., maximum=5.)])
    context = torch.tensor([[[-3., 4.]]]).expand(2, 12, 2).clone()
    original = context.clone()
    model.train()
    generator = torch.Generator().manual_seed(33)
    predicted = model.forecast(context, 4, mode=mode, generator=generator)
    assert predicted.shape == (2, 4, 2)
    assert torch.all((predicted[..., 0] >= -4.) & (predicted[..., 0] <= -2.))
    assert torch.all((predicted[..., 1] >= 3.) & (predicted[..., 1] <= 5.))
    assert model.training
    assert torch.equal(context, original)
    if mode == "sample":
        repeated = model.forecast(context, 4, mode=mode, generator=torch.Generator().manual_seed(33))
        assert torch.equal(predicted, repeated)


def test_huggingface_auto_save_load_reproduces_mixed_forecast(tmp_path):
    model = tiny([dict(name="a", kind="independent", tied=False, levels=13),
                  dict(name="b", kind="slerp", tied=True, knots=3, levels=17),
                  dict(name="c", kind="slerp", tied=False, knots=5, levels=19)]).eval()
    x = torch.rand(1, 8, 3) * 2 - 1
    prediction = model.forecast(x, 3)
    model.save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    restored = AutoModel.from_pretrained(tmp_path)
    assert config.columns == model.config.columns
    assert restored.parameter_count == model.parameter_count
    torch.testing.assert_close(restored.forecast(x, 3), prediction, atol=0, rtol=0)


def test_explicit_triton_does_not_silently_run_on_cpu():
    model = tiny([dict(kind="slerp", backend="triton")])
    with pytest.raises(RuntimeError, match="CUDA"):
        model(torch.zeros(1, 3, 1))


def test_ignore_only_column_and_auxiliary_loss():
    model = tiny([dict(name="a"), dict(name="b")], auxiliary_mse_weight=2.)
    x = torch.rand(1, 5, 2) * 2 - 1
    labels = model.quantize(x)
    labels[..., 1] = -100
    out = model(x, labels=labels, target_values=x)
    expected_mse = (model.expected_values(out.logits)[..., 0] - x[..., 0]).square().mean()
    torch.testing.assert_close(out.auxiliary_mse, expected_mse)
    torch.testing.assert_close(out.loss, out.cross_entropy + 2 * expected_mse)
    with pytest.raises(ValueError, match="valid"):
        model(x, labels=torch.full_like(labels, -100))


def test_invalid_context_shape_and_length_are_rejected():
    model = tiny()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="length"):
        model(torch.zeros(1, 9, 1))
    with pytest.raises(ValueError, match="length"):
        model(torch.zeros(1, 0, 1))


def test_matched_initialization_is_independent_of_geometry_size():
    from lane_lab.experiment import initialize_matched_model
    settings = dict(d_model=12, n_heads=2, n_layers=1, d_ff=20, context_length=8)
    lookup_config = LaneConfig(columns=[dict(kind="independent", levels=129)], **settings)
    curve_config = LaneConfig(columns=[dict(kind="slerp", knots=3, tied=False, levels=17)], **settings)
    before = torch.random.get_rng_state().clone()
    lookup_model = initialize_matched_model(lookup_config, 777)
    curve_model = initialize_matched_model(curve_config, 777)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert lookup_model.parameter_count != curve_model.parameter_count
    lookup_state, curve_state = lookup_model.state_dict(), curve_model.state_dict()
    backbone_names = [name for name in lookup_state if not name.startswith("columns.")]
    assert backbone_names
    for name in backbone_names:
        torch.testing.assert_close(lookup_state[name], curve_state[name], atol=0, rtol=0)
    # Independent output remains a one-time copy of its own input geometry.
    geometry = curve_model.columns[0]
    torch.testing.assert_close(geometry.input_geometry()[0], geometry.output_geometry()[0])


def test_training_rng_streams_ignore_model_and_noise_rng_consumption():
    from lane_lab.experiment import initialize_matched_model, training_generators
    a_batch, a_noise = training_generators(18)
    first_batch = torch.randint(512, (32,), generator=a_batch)
    first_noise = torch.randn(2, 8, 3, generator=a_noise)
    initialize_matched_model(LaneConfig(columns=[dict(kind="independent", levels=300)]), 18)
    torch.randn(1000)  # unrelated global RNG activity must not change the streams
    b_batch, b_noise = training_generators(18)
    assert torch.equal(first_batch, torch.randint(512, (32,), generator=b_batch))
    assert torch.equal(first_noise, torch.randn(2, 8, 3, generator=b_noise))
    torch.randn(91, generator=a_noise)  # even different noise consumption cannot alter batches
    assert torch.equal(torch.randint(512, (32,), generator=a_batch),
                       torch.randint(512, (32,), generator=b_batch))
def test_linear_numeric_targets_have_exact_mean():
    from lane_lab.experiment import categorical_loss
    config = LaneConfig(columns=[dict(name="x", kind="slerp", knots=3, levels=5)])
    model = LaneModel(config)
    targets = torch.tensor([[[-1.0], [-0.75], [0.0], [0.7], [1.0]]])
    logits = torch.zeros(1, 5, 5, requires_grad=True)
    loss = categorical_loss(model, (logits,), targets, "linear")
    gradient, = torch.autograd.grad(loss, logits)
    target_probability = torch.full_like(logits, .2) - gradient * 5
    grid = torch.linspace(-1, 1, 5)
    torch.testing.assert_close((target_probability * grid).sum(-1), targets[..., 0])
    torch.testing.assert_close(target_probability.sum(-1), torch.ones(1, 5))


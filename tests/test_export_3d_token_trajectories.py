import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE = runpy.run_path(Path(__file__).parents[1] / "analysis/export_3d_token_trajectories.py")
project_to_3d = MODULE["project_to_3d"]
finite_metric = MODULE["finite_metric"]


def test_native_three_dimensions_are_not_projected():
    frames = [torch.randn(5, 3), torch.randn(5, 3)]

    projected, metadata = project_to_3d(frames)

    assert projected is frames
    assert metadata == {"method": "native", "input_dimensions": 3}


def test_native_two_dimensions_are_padded_for_threejs():
    frames = [torch.randn(5, 2), torch.randn(5, 2)]

    projected, metadata = project_to_3d(frames)

    assert [tuple(frame.shape) for frame in projected] == [(5, 3), (5, 3)]
    assert all(torch.equal(actual[:, :2], source) for actual, source in zip(projected, frames))
    assert all(torch.count_nonzero(actual[:, 2]) == 0 for actual in projected)
    assert metadata == {"method": "native_2d", "input_dimensions": 2}


@pytest.mark.parametrize("embedding_dim", [8, 16, 64])
def test_global_pca_projects_higher_dimensions(embedding_dim):
    generator = torch.Generator().manual_seed(1234)
    frames = [torch.randn(14, embedding_dim, generator=generator) for _ in range(4)]

    projected, metadata = project_to_3d(frames)

    assert [tuple(frame.shape) for frame in projected] == [(14, 3)] * 4
    assert metadata["method"] == "pca"
    assert metadata["input_dimensions"] == embedding_dim
    assert len(metadata["explained_variance_ratio"]) == 3
    assert 0 < sum(metadata["explained_variance_ratio"]) <= 1

    # The same input yields the same sign-fixed global coordinates.
    repeated, _ = project_to_3d(frames)
    for actual, expected in zip(repeated, projected):
        assert torch.allclose(actual, expected)


def test_checkpoint_metrics_are_json_safe():
    assert finite_metric(torch.tensor(1.25)) == 1.25
    assert finite_metric(None) is None
    assert finite_metric(float("nan")) is None
    assert finite_metric(float("inf")) is None

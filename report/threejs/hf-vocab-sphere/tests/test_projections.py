from __future__ import annotations

import numpy as np
import pytest

from app.projections import nearest_neighbor_edges, project_vectors, projection_catalog


@pytest.fixture()
def vectors() -> np.ndarray:
    return np.random.default_rng(7).normal(size=(48, 96))


@pytest.mark.parametrize(
    "method",
    ["auto", "spherical_pca", "tangent_pca", "cosine_kernel", "angular_mds", "spherical_stress", "isomap", "random"],
)
def test_projection_is_unit_sphere(vectors: np.ndarray, method: str) -> None:
    output = project_vectors(vectors, method=method, anchor_index=0, seed=11)
    assert output.coordinates.shape == (48, 3)
    np.testing.assert_allclose(np.linalg.norm(output.coordinates, axis=1), 1.0, atol=1e-5)
    assert output.metrics["token_count"] == 48
    assert output.metrics["pair_sample_count"] > 0


def test_anchor_is_aligned_to_north(vectors: np.ndarray) -> None:
    output = project_vectors(vectors, method="spherical_pca", anchor_index=4, align_anchor=True)
    np.testing.assert_allclose(output.coordinates[4], np.array([0.0, 1.0, 0.0]), atol=1e-5)


def test_edges_are_unique(vectors: np.ndarray) -> None:
    edges = nearest_neighbor_edges(vectors, k=3)
    pairs = {(row["source_index"], row["target_index"]) for row in edges}
    assert len(pairs) == len(edges)
    assert all(source < target for source, target in pairs)


def test_catalog_includes_fidelity_methods() -> None:
    keys = {row["key"] for row in projection_catalog()}
    assert {"spherical_pca", "tangent_pca", "cosine_kernel", "angular_mds", "spherical_stress", "umap", "tsne"} <= keys


def test_projection_can_remain_in_free_3d(vectors: np.ndarray) -> None:
    output = project_vectors(vectors, method="spherical_pca", anchor_index=0, seed=11, geometry_mode="euclidean")
    assert output.coordinates.shape == (48, 3)
    radii = np.linalg.norm(output.coordinates, axis=1)
    assert np.std(radii) > 1e-3
    assert output.metrics["radial_normalization"] is False
    assert output.metrics["metric_geometry"] == "euclidean"

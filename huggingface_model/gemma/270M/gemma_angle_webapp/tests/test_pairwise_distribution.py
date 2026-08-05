from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.PreTrainedTokenizerBase = object
    sys.modules["transformers"] = fake_transformers

from app.model_service import TokenInfo, common_close_tokens, linear_transform_neighbors, minimum_angular_distances, pairwise_angle_distribution, vector_magnitudes  # noqa: E402


@dataclass
class TinyAssets:
    weight: torch.Tensor
    model_name: str = "tiny"
    requested_device: str = "cpu"
    effective_device: str = "cpu"

    def __post_init__(self) -> None:
        self.magnitudes = vector_magnitudes(self.weight)
        self.token_infos = [TokenInfo(i, str(i), str(i), str(i)) for i in range(self.weight.shape[0])]

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.weight.shape[1])

    def token(self, token_id: int) -> TokenInfo:
        return self.token_infos[token_id]


def test_pairwise_distribution_counts_unique_unordered_acute_angles() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = pairwise_angle_distribution(assets, block_size=2, compute_device="cpu", include_self=False)

    assert data["total_pairs"] == 6
    counts = {row["label"]: row["count"] for row in data["bins"]}
    assert counts["0–5°"] == 1
    assert counts["45–50°"] == 3
    assert counts["85–90°"] == 2
    assert sum(counts.values()) == 6
    assert data["bins"][0]["label"] == "0–5°"
    assert data["bins"][0]["rank"] == 1
    assert data["bins"][9]["label"] == "45–50°"
    assert data["bins"][9]["rank"] == 10
    assert data["bins"][17]["label"] == "85–90°"
    assert data["bins"][17]["rank"] == 18


def test_pairwise_distribution_includes_diagonal_when_requested() -> None:
    assets = TinyAssets(torch.eye(3))

    data = pairwise_angle_distribution(assets, block_size=2, compute_device="cpu", include_self=True)

    assert data["total_pairs"] == 6
    counts = {row["label"]: row["count"] for row in data["bins"]}
    assert counts["0–5°"] == 3
    assert counts["85–90°"] == 3
    assert sum(counts.values()) == 6


def test_pairwise_distribution_saves_unique_token_membership_by_bin() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = pairwise_angle_distribution(assets, block_size=2, compute_device="cpu", include_self=False)

    counts = {row["label"]: row["token_count"] for row in data["bins"]}
    assert counts["0–5°"] == 2
    assert counts["45–50°"] == 4
    assert counts["85–90°"] == 3

    from app.model_service import pairwise_angle_bin_tokens

    zero_degree_tokens = pairwise_angle_bin_tokens(assets, 0)
    assert zero_degree_tokens["label"] == "0–5°"
    assert [row["token_id"] for row in zero_degree_tokens["tokens"]] == [0, 3]

    forty_five_degree_tokens = pairwise_angle_bin_tokens(assets, 9)
    assert forty_five_degree_tokens["label"] == "45–50°"
    assert [row["token_id"] for row in forty_five_degree_tokens["tokens"]] == [0, 1, 2, 3]


def test_common_close_tokens_requires_both_angles_under_threshold_and_sorts_jointly() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    rows = common_close_tokens(assets, 0, 2, threshold_deg=46)

    assert [row["token_id"] for row in rows] == [0, 2]
    assert rows[0]["angle_to_token_a_deg"] == pytest.approx(0.0)
    assert rows[0]["angle_to_token_b_deg"] == pytest.approx(45.0)
    assert rows[1]["angle_to_token_a_deg"] == pytest.approx(45.0)
    assert rows[1]["angle_to_token_b_deg"] == pytest.approx(0.0)
    assert rows[1]["magnitude"] == pytest.approx(torch.sqrt(torch.tensor(2.0)).item())


def test_minimum_angular_distances_excludes_self_and_tracks_nearest_other_token() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = minimum_angular_distances(assets, block_size=2, compute_device="cpu")

    assert data["total_pairs"] == 6
    rows = data["rows"]
    assert [row["token_id"] for row in rows] == [0, 1, 2, 3]
    assert [row["other_token_id"] for row in rows] == [2, 2, 0, 1]
    assert [row["min_angle_rank"] for row in rows] == [1, 2, 3, 4]
    assert [row["min_angle_deg"] for row in rows] == pytest.approx([45.0, 45.0, 45.0, 90.0])
    assert rows[0]["magnitude"] == pytest.approx(1.0)
    assert rows[0]["other_magnitude"] == pytest.approx(torch.sqrt(torch.tensor(2.0)).item())


def test_linear_transform_neighbors_rank_one_maps_source_to_target() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(assets, 0, 2, 0, limit=4, transform_type="rank_one")

    assert data["transform_type"] == "rank_one"
    assert data["coefficient"] == pytest.approx(1.0)
    assert data["source_to_target_angle_deg"] == pytest.approx(45.0)
    assert data["transformed_vector_magnitude"] == pytest.approx(torch.sqrt(torch.tensor(2.0)).item())
    assert [row["token_id"] for row in data["rows"]] == [2, 0, 1, 3]
    assert data["rows"][0]["angle_deg"] == pytest.approx(0.0)


def test_linear_transform_neighbors_closest_identity_mode_is_available() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(assets, 0, 2, 0, limit=2, transform_type="closest-to-identity")

    assert data["transform_type"] == "closest_identity"
    assert data["coefficient"] == pytest.approx(1.0)
    assert data["transform_parameter_label"] == "Projection coefficient"
    assert data["rows"][0]["token_id"] == 2


def test_linear_transform_neighbors_closest_identity_alias_is_available() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(assets, 0, 2, 0, limit=2, transform_type="closest_identity")

    assert data["transform_type"] == "closest_identity"
    assert data["coefficient"] == pytest.approx(1.0)
    assert data["transform_parameter_label"] == "Projection coefficient"
    assert [row["token_id"] for row in data["rows"]] == [2, 0]
    assert data["rows"][0]["angle_deg"] == pytest.approx(0.0)


def test_linear_transform_neighbors_orthogonal_mode_rotates_direction_preserves_length() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(assets, 0, 1, 0, limit=4, transform_type="orthonormal")

    assert data["transform_type"] == "orthogonal"
    assert data["coefficient"] == pytest.approx(90.0)
    assert data["transform_parameter_label"] == "Direction rotation/reflection angle °"
    assert data["transformed_vector_magnitude"] == pytest.approx(1.0)
    assert [row["token_id"] for row in data["rows"]] == [1, 2, 0, 3]
    assert data["rows"][0]["angle_deg"] == pytest.approx(0.0)


def test_linear_transform_scale_zero_keeps_original_input_vector() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(
        assets,
        0,
        2,
        0,
        limit=3,
        transform_type="closest_identity",
        transform_scale=0.0,
    )

    assert data["transform_scale"] == pytest.approx(0.0)
    assert data["input_to_transformed_angle_deg"] == pytest.approx(0.0)
    assert data["transformed_vector_magnitude"] == pytest.approx(1.0)
    assert [row["token_id"] for row in data["rows"]] == [0, 2, 1]


def test_linear_transform_scale_extrapolates_fitted_effect() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(
        assets,
        0,
        2,
        0,
        limit=2,
        transform_type="closest_identity",
        transform_scale=2.0,
    )

    assert data["transform_scale"] == pytest.approx(2.0)
    assert data["transformed_vector_magnitude"] == pytest.approx(torch.sqrt(torch.tensor(5.0)).item())
    assert data["input_to_transformed_angle_deg"] == pytest.approx(63.434948, abs=1e-5)


def test_linear_transform_neighbors_offset_mode_is_available() -> None:
    assets = TinyAssets(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]))

    data = linear_transform_neighbors(assets, 0, 2, 1, limit=2, transform_type="offset")

    assert data["transform_type"] == "offset"
    assert data["coefficient"] == pytest.approx(1.0)
    assert [row["token_id"] for row in data["rows"]] == [1, 2]
    assert data["rows"][0]["angle_deg"] == pytest.approx(0.0)

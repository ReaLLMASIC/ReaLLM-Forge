from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from fastapi.testclient import TestClient


# The lightweight test environment may not have transformers installed. The app
# only needs these names at import time; model loading is monkeypatched below.
if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.PreTrainedTokenizerBase = object
    sys.modules["transformers"] = fake_transformers

from app import main, model_service  # noqa: E402
from app.model_service import LocalModelInfo, TokenInfo  # noqa: E402


@dataclass
class FakeAssets:
    model_name: str = "fake-model"
    requested_device: str = "cpu"
    effective_device: str = "cpu"

    def __post_init__(self) -> None:
        self.token_infos = [
            TokenInfo(token_id=0, raw="<bos>", display="<bos>", normalized="<bos> bos"),
            TokenInfo(token_id=1, raw="▁Hello", display=" Hello", normalized="▁hello hello"),
            TokenInfo(token_id=2, raw="world", display="world", normalized="world world"),
            TokenInfo(token_id=3, raw="<0xF9>", display="<0xF9>", normalized="<0xf9> 0xf9 f9"),
        ]
        self.weight = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.magnitudes = torch.linalg.vector_norm(self.weight, ord=2, dim=1)

    @property
    def vocab_size(self) -> int:
        return len(self.token_infos)

    @property
    def hidden_dim(self) -> int:
        return int(self.weight.shape[1])

    def token(self, token_id: int) -> TokenInfo:
        if token_id < 0 or token_id >= len(self.token_infos):
            raise IndexError(f"token_id must be in [0, {len(self.token_infos) - 1}], got {token_id}")
        return self.token_infos[token_id]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_assets = FakeAssets()
    monkeypatch.setattr(main, "get_current_assets", lambda: fake_assets)
    monkeypatch.setattr(main, "load_active_model", lambda *args, **kwargs: fake_assets)
    return TestClient(main.app)


def test_homepage_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "LM-Head Angle Explorer" in response.text


def test_status_route_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called():
        raise AssertionError("status should not load model assets")

    monkeypatch.setattr(main, "get_current_assets", fail_if_called)
    monkeypatch.setattr(
        main,
        "get_model_status",
        lambda: {
            "loaded": False,
            "model_name": "configured/default",
            "requested_device": "cpu",
            "effective_device": "cpu",
            "vocab_size": 0,
            "hidden_dim": 0,
        },
    )
    response = TestClient(main.app).get("/api/status")
    assert response.status_code == 200
    assert response.json() == {
        "loaded": False,
        "model_name": "configured/default",
        "requested_device": "cpu",
        "effective_device": "cpu",
        "vocab_size": 0,
        "hidden_dim": 0,
    }


def test_search_before_model_load_returns_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "get_current_assets",
        lambda: (_ for _ in ()).throw(ValueError("No model is loaded. Enter a Hugging Face model ID and click Load model first.")),
    )
    response = TestClient(main.app).get("/api/tokens/search", params={"q": "hello"})
    assert response.status_code == 400
    assert "click Load model" in response.json()["detail"]


def test_search_route_is_not_captured_by_token_id_route(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "hello"
    assert [row["token_id"] for row in data["results"]] == [1]


def test_search_route_supports_blank_query_without_result_limit(client: TestClient) -> None:
    # Stale clients may still send limit, but the endpoint should ignore it and
    # return the complete vocabulary.
    response = client.get("/api/tokens/search", params={"q": "", "limit": 2})
    assert response.status_code == 200
    assert [row["token_id"] for row in response.json()["results"]] == [0, 1, 2, 3]


def test_search_route_has_no_limit_validation(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "", "limit": -1})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 4


def test_search_is_literal_and_not_pattern_matching(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "^F9$"})
    assert response.status_code == 200
    data = response.json()
    assert "mode" not in data
    assert data["results"] == []


def test_literal_search_still_matches_byte_alias_contents(client: TestClient) -> None:
    response = client.get("/api/tokens/search", params={"q": "F9"})
    assert response.status_code == 200
    assert [row["token_id"] for row in response.json()["results"]] == [3]


def test_explicit_token_lookup_route(client: TestClient) -> None:
    response = client.get("/api/tokens/id/2")
    assert response.status_code == 200
    assert response.json()["raw"] == "world"


def test_legacy_token_lookup_route_still_works(client: TestClient) -> None:
    response = client.get("/api/tokens/2")
    assert response.status_code == 200
    assert response.json()["raw"] == "world"


def test_common_close_tokens_route_finds_tokens_under_threshold_from_both(client: TestClient) -> None:
    response = client.get(
        "/api/common-close-tokens",
        params={"token_a": 0, "token_b": 2, "max_angle_deg": 46},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["token_a_id"] == 0
    assert data["token_b_id"] == 2
    assert data["threshold_deg"] == 46
    assert data["match_count"] == 2
    assert [row["token_id"] for row in data["rows"]] == [0, 2]
    assert data["rows"][0]["angle_to_token_a_deg"] == pytest.approx(0.0)
    assert data["rows"][0]["angle_to_token_b_deg"] == pytest.approx(45.0)
    assert data["rows"][0]["magnitude"] == pytest.approx(1.0)


def test_common_close_tokens_route_uses_default_35_degree_threshold(client: TestClient) -> None:
    response = client.get(
        "/api/common-close-tokens",
        params={"token_a": 0, "token_b": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["threshold_deg"] == 35.0
    assert data["match_count"] == 0
    assert data["rows"] == []


def test_common_close_tokens_route_rejects_invalid_threshold(client: TestClient) -> None:
    response = client.get(
        "/api/common-close-tokens",
        params={"token_a": 0, "token_b": 2, "max_angle_deg": -1},
    )

    assert response.status_code == 422


def test_model_load_endpoint_accepts_huggingface_designation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(model_name: str, requested_device: str, **kwargs):
        calls.append((model_name, requested_device, kwargs))
        return FakeAssets(model_name=model_name, requested_device=requested_device, effective_device=requested_device)

    monkeypatch.setattr(main, "load_active_model", fake_load)
    client = TestClient(main.app)

    response = client.post("/api/model/load", json={"model_name": "Qwen/Qwen3.5-0.8B-Base", "device": "cpu"})

    assert response.status_code == 200
    data = response.json()
    assert data["loaded"] is True
    assert data["model_name"] == "Qwen/Qwen3.5-0.8B-Base"
    assert data["requested_device"] == "cpu"
    assert calls == [("Qwen/Qwen3.5-0.8B-Base", "cpu", {"force_reload": False, "allow_download": True})]


def test_model_load_endpoint_rejects_blank_model_name() -> None:
    client = TestClient(main.app)
    response = client.post("/api/model/load", json={"model_name": "   "})
    assert response.status_code == 400
    assert "blank" in response.json()["detail"].lower() or "empty" in response.json()["detail"].lower()



def test_model_load_endpoint_can_disable_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(model_name: str, requested_device: str, **kwargs):
        calls.append((model_name, requested_device, kwargs))
        return FakeAssets(model_name=model_name, requested_device=requested_device, effective_device=requested_device)

    monkeypatch.setattr(main, "load_active_model", fake_load)
    client = TestClient(main.app)

    response = client.post(
        "/api/model/load",
        json={"model_name": "Qwen/Qwen3.5-0.8B-Base", "device": "cpu", "allow_download": False},
    )

    assert response.status_code == 200
    assert calls == [("Qwen/Qwen3.5-0.8B-Base", "cpu", {"force_reload": False, "allow_download": False})]


def test_available_models_route_returns_local_cache_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "list_local_models",
        lambda: [
            LocalModelInfo(
                model_name="Qwen/Qwen3.5-0.8B-Base",
                cache_path="/tmp/hf/models--Qwen--Qwen3.5-0.8B-Base",
                size_bytes=1234,
                last_modified=100.0,
            )
        ],
    )
    client = TestClient(main.app)

    response = client.get("/api/models/available")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "model_name": "Qwen/Qwen3.5-0.8B-Base",
                "cache_path": "/tmp/hf/models--Qwen--Qwen3.5-0.8B-Base",
                "size_bytes": 1234,
                "last_modified": 100.0,
            }
        ]
    }



def test_manual_local_cache_scanner_finds_huggingface_model_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "hub"
    (cache_dir / "models--Qwen--Qwen3.5-0.8B-Base").mkdir(parents=True)
    (cache_dir / "models--google--gemma-3-270m").mkdir(parents=True)
    (cache_dir / "datasets--not--a-model").mkdir(parents=True)

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(model_service, "_scan_cache_with_hub", lambda records: None)

    models = model_service.list_local_models()

    assert [item.model_name for item in models] == [
        "google/gemma-3-270m",
        "Qwen/Qwen3.5-0.8B-Base",
    ]


def test_pairwise_angle_bins_route_returns_angle_ordered_5_degree_counts(client: TestClient) -> None:
    response = client.get(
        "/api/pairwise-angle-bins",
        params={"block_size": 2, "compute_device": "cpu", "include_self": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "fake-model"
    assert data["total_pairs"] == 6
    assert data["bin_degrees"] == 5.0
    assert data["angle_min_deg"] == 0.0
    assert data["angle_max_deg"] == 90.0
    assert data["acute_angle"] is True
    assert data["compute_device"] == "cpu"
    assert data["block_size"] == 2
    assert len(data["bins"]) == 18

    assert [(row["rank"], row["label"], row["count"]) for row in data["bins"][:3]] == [
        (1, "0–5°", 1),
        (2, "5–10°", 0),
        (3, "10–15°", 0),
    ]
    assert data["bins"][9]["rank"] == 10
    assert data["bins"][9]["label"] == "45–50°"
    assert data["bins"][9]["count"] == 3
    assert data["bins"][17]["rank"] == 18
    assert data["bins"][17]["label"] == "85–90°"
    assert data["bins"][17]["count"] == 2
    assert sum(row["count"] for row in data["bins"]) == data["total_pairs"]


def test_pairwise_angle_bins_route_can_include_self_pairs(client: TestClient) -> None:
    response = client.get(
        "/api/pairwise-angle-bins",
        params={"block_size": 2, "compute_device": "cpu", "include_self": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["include_self"] is True
    assert data["total_pairs"] == 10
    assert sum(row["count"] for row in data["bins"]) == 10
    zero_degree_bin = next(row for row in data["bins"] if row["label"] == "0–5°")
    assert zero_degree_bin["count"] == 5


def test_weight_tensor_name_prefers_lm_head_over_input_embedding() -> None:
    chosen = model_service._choose_weight_tensor_name(
        [
            "model.layers.0.mlp.up_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
    )
    assert chosen == "lm_head.weight"


def test_load_active_model_uses_weight_only_matrix_without_importing_model_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {"<bos>": 0, "hello": 1}

    weight = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)
    calls: list[tuple[str, bool, str]] = []

    monkeypatch.setattr(model_service, "_ASSETS", None)
    monkeypatch.setattr(model_service, "_load_tokenizer", lambda model_name, allow_download=True: FakeTokenizer())

    def fake_weight_matrix(model_name: str, *, allow_download: bool, effective_device: str) -> torch.Tensor:
        calls.append((model_name, allow_download, effective_device))
        return weight

    monkeypatch.setattr(model_service, "_load_output_weight_matrix", fake_weight_matrix)
    monkeypatch.setattr(
        model_service,
        "_load_causal_lm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full model class should not be imported")),
    )

    assets = model_service.load_active_model("google/gemma-3-270m", "cpu", force_reload=True, allow_download=True)

    assert calls == [("google/gemma-3-270m", True, "cpu")]
    assert assets.vocab_size == 2
    assert assets.hidden_dim == 2
    assert assets.token(1).raw == "hello"
    assert assets.magnitudes.detach().cpu().tolist() == [5.0, 13.0]


def test_model_load_error_formatter_adds_gemma_and_cpp_extension_hints() -> None:
    message = model_service._format_model_load_error(
        RuntimeError("Could not import module 'Gemma3ForCausalLM'. Skipping import of cpp extensions due to incompatible torch version.")
    )
    assert "safetensors-only" in message
    assert "compiled extension" in message


def test_weight_only_loader_uses_safetensors_index_for_single_needed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "fake-indexed-model"
    model_dir.mkdir()
    (model_dir / "model-00002-of-00003.safetensors").write_bytes(b"fake")
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map": {'
        '"model.layers.0.mlp.up_proj.weight": "model-00001-of-00003.safetensors", '
        '"model.embed_tokens.weight": "model-00002-of-00003.safetensors"'
        '}}',
        encoding="utf-8",
    )

    calls = []

    def fake_safe_open(path: Path, tensor_name: str) -> torch.Tensor:
        calls.append((path.name, tensor_name))
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    monkeypatch.setattr(model_service, "_safe_open_tensor", fake_safe_open)

    weight = model_service._load_output_weight_from_safetensors(str(model_dir), allow_download=False)

    assert weight.shape == (2, 2)
    assert calls == [("model-00002-of-00003.safetensors", "model.embed_tokens.weight")]


def test_pairwise_angle_bin_tokens_route_returns_last_clicked_bin_tokens(client: TestClient) -> None:
    model_service.clear_pairwise_bin_token_cache()
    bins_response = client.get(
        "/api/pairwise-angle-bins",
        params={"block_size": 2, "compute_device": "cpu", "include_self": False},
    )
    assert bins_response.status_code == 200

    response = client.get("/api/pairwise-angle-bins/9/tokens")

    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "45–50°"
    assert data["token_count"] == 4
    assert [row["token_id"] for row in data["tokens"]] == [0, 1, 2, 3]
    assert [row["raw"] for row in data["tokens"]] == ["<bos>", "▁Hello", "world", "<0xF9>"]


def test_pairwise_angle_bin_tokens_route_requires_prior_bin_compute(client: TestClient) -> None:
    model_service.clear_pairwise_bin_token_cache()

    response = client.get("/api/pairwise-angle-bins/0/tokens")

    assert response.status_code == 400
    assert "Compute pairwise bins first" in response.json()["detail"]


def test_min_angular_distances_route_returns_closest_non_self_tokens(client: TestClient) -> None:
    response = client.get(
        "/api/min-angular-distances",
        params={"block_size": 2, "compute_device": "cpu"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "fake-model"
    assert data["total_pairs"] == 6
    assert data["compute_device"] == "cpu"
    assert data["block_size"] == 2

    rows = data["rows"]
    assert [row["token_id"] for row in rows] == [0, 1, 2, 3]
    assert [row["other_token_id"] for row in rows] == [2, 2, 0, 1]
    assert [row["min_angle_rank"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["min_angle_deg"] == pytest.approx(45.0)
    assert rows[1]["min_angle_deg"] == pytest.approx(45.0)
    assert rows[2]["min_angle_deg"] == pytest.approx(45.0)
    assert rows[3]["min_angle_deg"] == pytest.approx(90.0)
    assert rows[0]["token_raw"] == "<bos>"
    assert rows[0]["other_token_raw"] == "world"
    assert rows[0]["magnitude"] == pytest.approx(1.0)
    assert rows[0]["other_magnitude"] == pytest.approx(2**0.5)

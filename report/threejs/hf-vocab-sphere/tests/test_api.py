from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_and_metadata_routes() -> None:
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}
    status = client.get("/api/status")
    assert status.status_code == 200
    assert "loaded" in status.json()
    methods = client.get("/api/projection/methods")
    assert methods.status_code == 200
    assert len(methods.json()["methods"]) >= 8


def test_model_required_route_is_explicit() -> None:
    response = client.get("/api/tokens/0")
    assert response.status_code == 409
    assert "No model is loaded" in response.json()["detail"]


def test_projection_endpoint_includes_vector_arithmetic(monkeypatch) -> None:
    from app import main as main_module
    from app.model_store import ModelAssets, TokenInfo

    weight = torch.eye(4, dtype=torch.float32)
    token_infos = [
        TokenInfo(
            token_id=index,
            raw=f"tok{index}",
            display=f"tok{index}",
            search_blob=f"tok{index}",
            special=False,
        )
        for index in range(4)
    ]
    assets = ModelAssets(
        model_name="test/model",
        revision="main",
        matrix_source="input",
        tensor_name="embed.weight",
        tokenizer=None,
        token_infos=token_infos,
        weight=weight,
        magnitudes=torch.linalg.vector_norm(weight, dim=1),
        compute_device="cpu",
        load_strategy="test",
    )
    monkeypatch.setattr(main_module, "_assets_or_http", lambda: assets)

    response = client.post(
        "/api/projection",
        json={
            "token_ids": [0, 1, 2],
            "anchor_id": 0,
            "method": "spherical_pca",
            "geometry_mode": "sphere",
            "arithmetic_expressions": [
                {"expression": "(A - B) + C", "label": "analogy"}
            ],
            "edge_k": 1,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["geometry_mode"] == "sphere"
    assert len(payload["points"]) == 4
    resultant = payload["points"][-1]
    assert resultant["kind"] == "resultant"
    assert resultant["token_id"] is None
    assert resultant["alias"] == "R1"
    assert resultant["label"] == "analogy"
    assert resultant["referenced_aliases"] == ["A", "B", "C"]
    assert resultant["magnitude"] == pytest.approx(3 ** 0.5)
    assert payload["edges"]
    assert all("angle_deg" in edge for edge in payload["edges"])


def test_tokenize_text_endpoint_preserves_occurrences(monkeypatch) -> None:
    from app import main as main_module
    from app.model_store import ModelAssets, TokenInfo

    class DummyTokenizer:
        all_special_ids = [2]

        def encode(self, text, add_special_tokens=False):
            ids = [0, 1, 0, 5]
            return ([2] + ids) if add_special_tokens else ids

        def convert_ids_to_tokens(self, token_id):
            return f"tok{token_id}"

        def decode(self, token_ids, **_kwargs):
            return f"piece-{token_ids[0]}"

    weight = torch.eye(4, dtype=torch.float32)
    token_infos = [
        TokenInfo(
            token_id=index,
            raw=f"tok{index}",
            display=f"tok{index}",
            search_blob=f"tok{index}",
            special=index == 2,
        )
        for index in range(4)
    ]
    assets = ModelAssets(
        model_name="test/model",
        revision="main",
        matrix_source="input",
        tensor_name="embed.weight",
        tokenizer=DummyTokenizer(),
        token_infos=token_infos,
        weight=weight,
        magnitudes=torch.linalg.vector_norm(weight, dim=1),
        compute_device="cpu",
        load_strategy="test",
    )
    monkeypatch.setattr(main_module, "_assets_or_http", lambda: assets)

    response = client.post(
        "/api/tokens/tokenize",
        json={"text": "hello", "add_special_tokens": True, "max_tokens": 20},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["token_count"] == 5
    assert payload["returned_count"] == 5
    assert payload["unique_token_count"] == 4
    assert payload["projectable_token_count"] == 3
    assert [row["token_id"] for row in payload["tokens"]] == [2, 0, 1, 0, 5]
    assert payload["tokens"][3]["sequence_index"] == 3
    assert payload["tokens"][3]["decoded"] == "piece-0"
    assert payload["tokens"][-1]["projectable"] is False

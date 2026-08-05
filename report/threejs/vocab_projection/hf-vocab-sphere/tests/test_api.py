from __future__ import annotations

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

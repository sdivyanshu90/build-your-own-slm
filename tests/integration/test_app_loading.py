"""Tests for the app's model-loading lifespan behaviour."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from slm.api.app import create_app
from slm.config import Settings

pytestmark = pytest.mark.integration


def test_app_loads_model_from_disk(trained_model_dir):
    settings = Settings(
        _env_file=None, model_dir=trained_model_dir, device="cpu", api_keys="", log_level="WARNING"
    )
    app = create_app(settings)  # load_model defaults to True, no engine injected
    with TestClient(app) as client:
        assert client.get("/readyz").json()["model_loaded"] is True
        resp = client.post("/v1/completions", json={"prompt": "the", "max_tokens": 3})
        assert resp.status_code == 200


def test_app_handles_missing_model_dir(tmp_path):
    settings = Settings(
        _env_file=None, model_dir=tmp_path / "absent", api_keys="", log_level="WARNING"
    )
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/healthz").json()["model_loaded"] is False

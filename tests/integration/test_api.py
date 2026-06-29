"""Integration tests for the HTTP API."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from slm.api.app import create_app
from slm.config import Settings

pytestmark = pytest.mark.integration


# ---- Health & observability -----------------------------------------------
def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_readyz_ready(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_metrics_exposes_prometheus(client, auth_header):
    client.get("/v1/models", headers=auth_header)  # generate some traffic
    text = client.get("/metrics").text
    assert "slm_http_requests_total" in text
    assert "slm_http_request_duration_seconds" in text


# ---- Auth -----------------------------------------------------------------
def test_completion_requires_api_key(client):
    resp = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "http_error"


def test_completion_rejects_bad_key(client):
    resp = client.post(
        "/v1/completions", headers={"X-API-Key": "wrong"}, json={"prompt": "hi", "max_tokens": 3}
    )
    assert resp.status_code == 401


def test_bearer_token_accepted(client):
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": "Bearer test-key"},
        json={"prompt": "hi", "max_tokens": 2},
    )
    assert resp.status_code == 200


# ---- Inference ------------------------------------------------------------
def test_list_models(client, auth_header):
    body = client.get("/v1/models", headers=auth_header).json()
    assert body["data"][0]["id"] == "byo-slm"
    assert body["data"][0]["parameters"] > 0


def test_completion_buffered(client, auth_header):
    resp = client.post(
        "/v1/completions",
        headers=auth_header,
        json={"prompt": "the quick", "max_tokens": 8, "seed": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_completion_streaming(client, auth_header):
    with client.stream(
        "POST",
        "/v1/completions",
        headers=auth_header,
        json={"prompt": "the dog", "max_tokens": 6, "stream": True, "seed": 2},
    ) as stream:
        events = [line for line in stream.iter_lines() if line.startswith("data:")]
    assert any("[DONE]" in e for e in events)
    # The penultimate event carries the terminal finish_reason.
    payloads = [json.loads(e[5:]) for e in events if "[DONE]" not in e]
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


# ---- Validation -----------------------------------------------------------
def test_validation_rejects_zero_max_tokens(client, auth_header):
    resp = client.post(
        "/v1/completions", headers=auth_header, json={"prompt": "x", "max_tokens": 0}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_validation_rejects_unknown_field(client, auth_header):
    resp = client.post("/v1/completions", headers=auth_header, json={"prompt": "x", "nope": 1})
    assert resp.status_code == 422


# ---- Security headers & request id ----------------------------------------
def test_security_headers_present(client):
    headers = client.get("/healthz").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_request_id_is_echoed(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["X-Request-ID"] == "trace-123"


def test_request_id_generated_when_absent(client):
    assert client.get("/healthz").headers.get("X-Request-ID")


# ---- Failure modes (dedicated apps) ---------------------------------------
def test_readiness_503_without_model(inference_engine):
    settings = Settings(api_keys="", log_level="WARNING")
    app = create_app(settings, engine=None, load_model=False)
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 503
        assert c.get("/healthz").status_code == 200
        assert c.post("/v1/completions", json={"prompt": "x", "max_tokens": 1}).status_code == 503


def test_prompt_too_long_returns_400(trained_model_dir):
    from slm.inference.engine import InferenceEngine

    engine = InferenceEngine.from_pretrained(trained_model_dir, device="cpu", max_prompt_tokens=2)
    app = create_app(Settings(api_keys="", log_level="WARNING"), engine=engine, load_model=False)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/completions",
            json={"prompt": "this is definitely more than two tokens long", "max_tokens": 4},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request"


def test_unhandled_error_returns_500(inference_engine, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(inference_engine, "generate", boom)
    app = create_app(
        Settings(api_keys="", log_level="WARNING"), engine=inference_engine, load_model=False
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.post("/v1/completions", json={"prompt": "x", "max_tokens": 2})
        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "internal_error"


def test_rate_limit_429(inference_engine):
    settings = Settings(
        api_keys="k", rate_limit_per_minute=60, rate_limit_burst=2, log_level="WARNING"
    )
    app = create_app(settings, engine=inference_engine, load_model=False)
    with TestClient(app) as c:
        codes = [
            c.post(
                "/v1/completions", headers={"X-API-Key": "k"}, json={"prompt": "a", "max_tokens": 1}
            ).status_code
            for _ in range(4)
        ]
    assert codes.count(200) == 2
    assert codes.count(429) == 2


def test_auth_disabled_allows_anonymous(inference_engine):
    app = create_app(
        Settings(api_keys="", log_level="WARNING"), engine=inference_engine, load_model=False
    )
    with TestClient(app) as c:
        assert c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2}).status_code == 200

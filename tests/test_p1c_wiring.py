"""Phase 1c wiring: vLLM backend routing + a uniform API error envelope.

- build_backend must route "vllm" to the vLLM client, and gate on the server's
  health BEFORE loading the segmenter — so a down server fails fast with an
  actionable message and (on a CPU box) never even tries to import torch.
- API errors should come back as a uniform {error, message} object instead of
  FastAPI's default {detail}.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_build_backend_routes_vllm_and_gates_on_health(monkeypatch):
    # Point at a closed port: the health gate must raise the actionable error
    # before the segmenter (torch) is ever touched.
    monkeypatch.setenv("ANOTMED_BACKEND", "vllm")
    monkeypatch.setenv("ANOTMED_VLLM_URL", "http://127.0.0.1:9/v1")
    from anotmed.backends import build_backend
    from anotmed.config import Config

    with pytest.raises(RuntimeError) as exc:
        build_backend(Config())
    assert "serve_vllm" in str(exc.value).lower()


def test_api_errors_use_error_message_envelope():
    from anotmed.api import app

    client = TestClient(app)
    r = client.get("/api/studies/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "message" in body and body["message"]

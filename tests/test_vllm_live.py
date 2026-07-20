"""Real-HTTP integration test for the vLLM client.

The unit tests inject a fake http object; here we stand up an actual HTTP server
that speaks the vLLM OpenAI shape and drive VllmLocalizer over a real socket.
This catches anything the mock can't: httpx request serialization, the data-URL
image surviving the wire, /health path resolution. Still zero GPU.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from anotmed.backends.vllm_medgemma import VllmLocalizer, _VllmClient
from anotmed.config import Config
from anotmed.schema import ImageMeta

_REQUESTS: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    RESPONSE = ('{"findings": [{"box_2d": [100, 100, 200, 200], '
                '"label": "nodule", "confidence": 0.7}]}')

    def log_message(self, *args):
        pass  # keep test output clean

    def do_GET(self):
        self.send_response(200 if self.path.endswith("/health") else 404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _REQUESTS.append(json.loads(self.rfile.read(n) or b"{}"))
        data = json.dumps({"choices": [{"message": {"content": self.RESPONSE}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def vllm_url():
    _REQUESTS.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


def _meta() -> ImageMeta:
    return ImageMeta(study_id="s", rows=256, cols=256, modality="CT",
                     window_center=40, window_width=400)


def test_localizer_over_real_http_parses_and_sends_wire_format(vllm_url):
    cfg = Config(backend="vllm", vllm_url=vllm_url, max_findings=8)
    dets = VllmLocalizer(_VllmClient(cfg), cfg).propose(
        np.zeros((256, 256), np.float32), _meta())

    assert len(dets) == 1 and dets[0].label == "nodule"
    assert dets[0].score == pytest.approx(0.7)

    body = _REQUESTS[-1]
    assert body["temperature"] == 0
    assert body["guided_json"]["properties"]["findings"]["maxItems"] == 8
    image_part = next(p for p in body["messages"][0]["content"] if p.get("type") == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_health_over_real_http_succeeds(vllm_url):
    _VllmClient(Config(backend="vllm", vllm_url=vllm_url)).health()  # must not raise

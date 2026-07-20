"""vLLM MedGemma backend — exercised entirely GPU-free by mocking one httpx call.

The backend is a thin client to a separate vLLM OpenAI server, so the whole
contract can be tested by injecting a fake http client: what request goes out
(temperature 0, a data-URL image, the guided schema) and how each kind of
response is turned into Detections (guided JSON, free-text fallback, garbage,
connection refused).
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from anotmed.backends.vllm_medgemma import VllmLocalizer, VllmReporter, _VllmClient
from anotmed.backends.base import Detection
from anotmed.config import Config
from anotmed.schema import BBox, ImageMeta


# ---- fakes ------------------------------------------------------------------

class _Resp:
    def __init__(self, content: str | None = None, status: int = 200):
        self._content = content
        self.status_code = status

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeHttp:
    """Records the outbound request; replays a canned response or raises."""

    def __init__(self, content: str | None = None, raise_exc: Exception | None = None,
                 health_ok: bool = True):
        self.content = content
        self.raise_exc = raise_exc
        self.health_ok = health_ok
        self.last_post = None

    def post(self, url, json=None):
        self.last_post = (url, json)
        if self.raise_exc:
            raise self.raise_exc
        return _Resp(self.content)

    def get(self, url):
        if self.raise_exc:
            raise self.raise_exc
        return _Resp(status=200 if self.health_ok else 503)

    def close(self):
        pass


def _cfg() -> Config:
    return Config(backend="vllm", max_findings=8)


def _meta() -> ImageMeta:
    return ImageMeta(study_id="s", rows=256, cols=256, modality="CT",
                     window_center=40, window_width=400)


def _image() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.float32)


def _localizer(http) -> VllmLocalizer:
    return VllmLocalizer(_VllmClient(_cfg(), http=http), _cfg())


# ---- response handling ------------------------------------------------------

def test_guided_json_response_becomes_detections():
    content = ('{"findings": [{"box_2d": [100, 100, 200, 200], '
               '"label": "nodule", "confidence": 0.8}]}')
    dets = _localizer(_FakeHttp(content)).propose(_image(), _meta())
    assert len(dets) == 1
    assert dets[0].label == "nodule"
    assert dets[0].score == pytest.approx(0.8)
    b = dets[0].box
    assert b.x + b.w <= 256 and b.y + b.h <= 256


def test_free_text_with_boxes_falls_back_to_the_parser():
    # Not the guided {"findings": ...} object — a raw array in prose. The
    # FindingList validation misses; parsing.py must still recover the box.
    content = 'Findings: [{"label": "mass", "box_2d": [10, 10, 20, 20]}]'
    dets = _localizer(_FakeHttp(content)).propose(_image(), _meta())
    assert [d.label for d in dets] == ["mass"]
    assert dets[0].score == 1.0  # no confidence given -> default


def test_garbage_response_returns_empty_and_does_not_raise():
    dets = _localizer(_FakeHttp("The study is unremarkable.")).propose(_image(), _meta())
    assert dets == []


def test_empty_content_returns_empty():
    dets = _localizer(_FakeHttp(None)).propose(_image(), _meta())
    assert dets == []


# ---- what goes out over the wire --------------------------------------------

def test_request_carries_temperature_zero_data_url_and_guided_schema():
    http = _FakeHttp('{"findings": []}')
    _localizer(http).propose(_image(), _meta())
    url, body = http.last_post
    assert url.endswith("/v1/chat/completions")
    assert body["temperature"] == 0
    assert body["guided_json"]["properties"]["findings"]["maxItems"] == 8
    parts = body["messages"][0]["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_guided_json_omitted_when_disabled():
    http = _FakeHttp('{"findings": []}')
    cfg = Config(backend="vllm", guided_json=False)
    VllmLocalizer(_VllmClient(cfg, http=http), cfg).propose(_image(), _meta())
    _, body = http.last_post
    assert "guided_json" not in body


# ---- health readiness -------------------------------------------------------

def test_health_check_raises_actionable_error_on_connection_refused():
    client = _VllmClient(_cfg(), http=_FakeHttp(raise_exc=httpx.ConnectError("refused")))
    with pytest.raises(RuntimeError) as exc:
        client.health()
    msg = str(exc.value)
    assert "vllm" in msg.lower() and "serve_vllm" in msg.lower()


def test_health_check_passes_when_server_ready():
    client = _VllmClient(_cfg(), http=_FakeHttp(health_ok=True))
    client.health()  # must not raise


# ---- sleep / wake (VRAM offload between phases) ------------------------------

class _AdminHttp:
    def __init__(self, sleeping=False):
        self.sleeping = sleeping
        self.posts = []

    def post(self, url, json=None, params=None):
        self.posts.append((url, params))
        return _Resp()

    def get(self, url):
        class _R:
            status_code = 200

            def json(_self):
                return {"is_sleeping": self.sleeping}

            def raise_for_status(_self):
                pass

        return _R()


def test_sleep_posts_to_server_root_with_level():
    http = _AdminHttp()
    _VllmClient(_cfg(), http=http).sleep(level=1)
    url, params = http.posts[-1]
    assert url.endswith("/sleep") and "/v1" not in url  # admin endpoint at root
    assert params == {"level": 1}


def test_wake_up_posts_to_wake_endpoint():
    http = _AdminHttp()
    _VllmClient(_cfg(), http=http).wake_up()
    assert http.posts[-1][0].endswith("/wake_up")


def test_is_sleeping_reads_the_json_flag():
    assert _VllmClient(_cfg(), http=_AdminHttp(sleeping=True)).is_sleeping() is True
    assert _VllmClient(_cfg(), http=_AdminHttp(sleeping=False)).is_sleeping() is False


# ---- reporter ---------------------------------------------------------------

def test_reporter_returns_draft_with_verify_disclaimer():
    http = _FakeHttp("Rounded soft-tissue density in the right lung base.")
    reporter = VllmReporter(_VllmClient(_cfg(), http=http), _cfg())
    det = Detection(box=BBox(x=50, y=50, w=40, h=40), label="nodule", score=0.8)
    text = reporter.describe(_image(), det, np.zeros((256, 256), bool), _meta())
    assert "verify" in text.lower()
    assert "right lung base" in text

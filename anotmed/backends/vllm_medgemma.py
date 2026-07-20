"""MedGemma behind a vLLM OpenAI server — a thin httpx client, no torch here.

The VLM runs in a separate vLLM process (see scripts/serve_vllm.sh); this module
just talks to it over HTTP. That is what lets the whole backend be tested on CPU
by mocking one request (tests/test_vllm_backend.py) — the app imports no torch or
transformers on the VLM path.

Localization uses vLLM guided decoding: we hand the server a JSON Schema
(schemas.guided_schema) so generation is grammar-constrained to well-formed
boxes. When guided decoding is off or the model answers in prose, we fall back to
the tolerant free-text parser from parsing.py. Nothing here raises on bad model
output — a garbage response yields zero findings, never a crashed upload.
"""

from __future__ import annotations

import base64
import logging

import httpx
import numpy as np
from pydantic import ValidationError

from ..config import Config
from ..io_dicom import image_png
from ..schema import BBox, ImageMeta
from .base import Detection
from .parsing import parse_detections, to_pixel_bbox
from .schemas import FindingList, guided_schema

log = logging.getLogger("anotmed.backends.vllm")


_LOCALIZE_PROMPT = (
    "You are assisting a radiologist by flagging regions to review. This is not a "
    'diagnosis. Return a JSON object {"findings": [ ... ]} where each finding is '
    '{"box_2d": [ymin, xmin, ymax, xmax], "label": <short finding name>, '
    '"confidence": <0-1>} with coordinates normalized to 0-1000. Use an empty list '
    "if nothing warrants review."
)

_REPORT_PROMPT = (
    "Describe the highlighted region (possible {label}) concisely for a radiologist "
    "to verify. One or two sentences. State observations only, not a diagnosis."
)


def _mime_of(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def _data_url(data: bytes) -> str:
    return f"data:{_mime_of(data)};base64,{base64.b64encode(data).decode('ascii')}"


class _VllmClient:
    """One shared connection to the vLLM OpenAI server. `http` is injectable for tests."""

    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self.model = cfg.medgemma_model
        base = cfg.vllm_url.rstrip("/")
        self._chat_url = f"{base}/chat/completions"
        root = base[:-3] if base.endswith("/v1") else base  # /health lives at server root
        self._health_url = f"{root}/health"
        self._http = http or httpx.Client(timeout=cfg.vllm_timeout_s)

    def chat(self, data_url: str, prompt: str, guided_schema: dict | None = None,
             max_tokens: int = 512) -> str:
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if guided_schema is not None:
            body["guided_json"] = guided_schema
        resp = self._http.post(self._chat_url, json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def health(self) -> None:
        """Raise an actionable RuntimeError if the vLLM server is not ready."""
        try:
            resp = self._http.get(self._health_url)
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"vLLM not reachable at {self.cfg.vllm_url} — run scripts/serve_vllm.sh ({e})"
            ) from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM at {self.cfg.vllm_url} not ready (HTTP {resp.status_code}) — "
                "run scripts/serve_vllm.sh"
            )

    def close(self) -> None:
        self._http.close()


def _detections_from_findings(fl: FindingList, rows: int, cols: int) -> list[Detection]:
    return [
        Detection(
            box=to_pixel_bbox(*f.box_2d, rows=rows, cols=cols),
            label=f.label or "finding",
            score=float(f.confidence),
        )
        for f in fl.findings
    ]


class VllmLocalizer:
    def __init__(self, client: _VllmClient, cfg: Config):
        self.client = client
        self.cfg = cfg
        self.max_findings = cfg.max_findings
        self.name = f"vllm-localizer:{cfg.medgemma_model}"

    def propose(self, image: np.ndarray, meta: ImageMeta) -> list[Detection]:
        data_url = _data_url(image_png(image, meta))
        schema = guided_schema(self.max_findings) if self.cfg.guided_json else None
        text = self.client.chat(data_url, _LOCALIZE_PROMPT, guided_schema=schema)
        return self._parse(text or "", meta.rows, meta.cols)[: self.max_findings]

    def _parse(self, text: str, rows: int, cols: int) -> list[Detection]:
        # Trust a validated guided-JSON object; otherwise recover boxes from prose.
        try:
            fl = FindingList.model_validate_json(text)
            return _detections_from_findings(fl, rows, cols)
        except (ValidationError, ValueError, TypeError):
            pass
        dets = parse_detections(text, rows, cols)
        if not dets and text.strip():
            log.warning("vLLM localizer: could not parse response: %.200r", text)
        return dets


class VllmReporter:
    def __init__(self, client: _VllmClient, cfg: Config):
        self.client = client
        self.cfg = cfg
        self.name = f"vllm-reporter:{cfg.medgemma_model}"

    def describe(self, image: np.ndarray, det: Detection, mask: np.ndarray,
                 meta: ImageMeta) -> str:
        h, w = image.shape
        pad = 0.15  # crop with margin so the model attends to the finding in context
        b = BBox(
            x=det.box.x - det.box.w * pad,
            y=det.box.y - det.box.h * pad,
            w=det.box.w * (1 + 2 * pad),
            h=det.box.h * (1 + 2 * pad),
        ).clamp(w, h)
        rs, cs = b.as_slice(w, h)
        data_url = _data_url(image_png(image[rs, cs], meta))
        text = self.client.chat(
            data_url, _REPORT_PROMPT.format(label=det.label), max_tokens=160
        ).strip()
        return f"{text}\n\n[AI-generated draft — radiologist must verify.]"


def build(cfg: Config) -> tuple[VllmLocalizer, VllmReporter]:
    client = _VllmClient(cfg)
    return VllmLocalizer(client, cfg), VllmReporter(client, cfg)

"""Guided-JSON schema + coord scaling for the vLLM MedGemma backend.

Two pure surfaces, both GPU-free:

  * `schemas.guided_schema(n)` — the JSON Schema handed to vLLM so the decoder is
    grammar-constrained to emit well-formed boxes (4 ints in 0-1000), capped at n
    findings. `FindingList`/`FindingBox` validate the response we get back.
  * `parsing.to_pixel_bbox` — shared 0-1000 → pixel scaling with clamping, reused
    by both the guided path and the free-text fallback so they can't drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anotmed.backends.parsing import to_pixel_bbox
from anotmed.backends.schemas import FindingBox, FindingList, guided_schema


# ---- guided_schema: what vLLM's decoder is constrained to -------------------

def test_guided_schema_caps_number_of_findings():
    assert guided_schema(3)["properties"]["findings"]["maxItems"] == 3


def test_guided_schema_box_is_four_ints_bounded_0_1000():
    items = guided_schema(8)["properties"]["findings"]["items"]
    box = items["properties"]["box_2d"]
    assert box["minItems"] == 4 and box["maxItems"] == 4
    assert box["items"]["minimum"] == 0 and box["items"]["maximum"] == 1000
    assert "box_2d" in items["required"] and "label" in items["required"]


def test_guided_schema_is_self_contained_no_refs():
    # vLLM guided decoding is most portable without $ref/$defs indirection.
    import json
    assert "$ref" not in json.dumps(guided_schema(8))


# ---- FindingList / FindingBox: validating the model's response --------------

def test_findinglist_accepts_a_well_formed_response():
    fl = FindingList.model_validate(
        {"findings": [{"box_2d": [1, 2, 3, 4], "label": "nodule", "confidence": 0.5}]}
    )
    assert fl.findings[0].confidence == 0.5


def test_findingbox_rejects_out_of_range_coordinate():
    with pytest.raises(ValidationError):
        FindingBox.model_validate({"box_2d": [0, 0, 0, 2000], "label": "x"})


def test_findingbox_rejects_wrong_length_box():
    with pytest.raises(ValidationError):
        FindingBox.model_validate({"box_2d": [0, 0, 0], "label": "x"})


def test_findingbox_confidence_defaults_to_one():
    fb = FindingBox.model_validate({"box_2d": [0, 0, 10, 10], "label": "x"})
    assert fb.confidence == 1.0


# ---- to_pixel_bbox: shared scaling, used by both parse paths -----------------

def test_to_pixel_bbox_scales_normalized_to_pixels():
    b = to_pixel_bbox(0, 0, 1000, 1000, rows=100, cols=200)
    assert (b.x, b.y) == (0.0, 0.0)
    assert b.w == 200 and b.h == 100


def test_to_pixel_bbox_clamps_out_of_range_into_image():
    b = to_pixel_bbox(-100, -100, 3000, 3000, rows=50, cols=50)
    assert b.x >= 0 and b.y >= 0
    assert b.x + b.w <= 50 and b.y + b.h <= 50


# ---- config: the new vLLM knobs have sane defaults --------------------------

def test_config_exposes_vllm_defaults(monkeypatch):
    for var in ("ANOTMED_VLLM_URL", "ANOTMED_VLLM_TIMEOUT_S", "ANOTMED_GUIDED_JSON"):
        monkeypatch.delenv(var, raising=False)
    from anotmed.config import Config

    cfg = Config()
    assert cfg.vllm_url.startswith("http")
    assert cfg.vllm_timeout_s > 0
    assert cfg.guided_json is True

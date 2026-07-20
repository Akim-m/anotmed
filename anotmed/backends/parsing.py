"""Turn raw MedGemma text into Detections — defensively.

Model output is adversarial input. It can arrive wrapped in prose or code
fences, carry brackets inside a label string, list non-numeric coordinates, or
give values outside the normalized 0-1000 range. None of that may crash a study
upload, so every function here follows one rule: **skip the bad element, keep
the good ones, and on total failure return an empty list — never raise.**

Kept free of torch/transformers so it is unit-testable on CPU (see
tests/test_parsing.py). The MedGemma adapter imports these; it does not
re-implement them.
"""

from __future__ import annotations

import json

from ..schema import BBox
from .base import Detection

# Gemma-family boxes are normalized to this range; clamp raw coords here before
# scaling to pixels so a hallucinated 3000 can't produce an off-image box.
_COORD_MAX = 1000.0


def extract_json_array(text: str) -> list:
    """Return the first JSON array in `text`, or [] if there isn't a valid one.

    String-aware: uses json's own decoder (which respects quotes) rather than
    counting brackets, so a ']' inside a label does not truncate the array. If
    the first '[' does not start valid JSON (e.g. it's prose like "[see note]"),
    scan to the next '[' and try again.
    """
    if not text:
        return []
    text = text.strip()
    for fence in ("```json", "```"):
        text = text.removeprefix(fence)
    text = text.removesuffix("```").strip()

    decoder = json.JSONDecoder()
    idx = text.find("[")
    while idx != -1:
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("[", idx + 1)
            continue
        if isinstance(value, list):
            return value
        idx = text.find("[", idx + 1)
    return []


def _coord(value: float) -> float:
    return max(0.0, min(_COORD_MAX, float(value)))


def to_pixel_bbox(ymin, xmin, ymax, xmax, rows: int, cols: int) -> BBox:
    """Scale a Gemma-family normalized box (0-1000) to a clamped, in-image BBox.

    Shared by the free-text fallback parser and the vLLM guided-JSON path so the
    two can never disagree on how a box maps to pixels.
    """
    ymin, xmin, ymax, xmax = (_coord(v) for v in (ymin, xmin, ymax, xmax))
    y0, y1 = sorted((ymin / _COORD_MAX * rows, ymax / _COORD_MAX * rows))
    x0, x1 = sorted((xmin / _COORD_MAX * cols, xmax / _COORD_MAX * cols))
    return BBox(x=x0, y=y0, w=max(1.0, x1 - x0), h=max(1.0, y1 - y0)).clamp(cols, rows)


def parse_detections(
    text: str, rows: int, cols: int, default_score: float = 1.0
) -> list[Detection]:
    """Parse `text` into clamped, in-image Detections. Never raises.

    Expects Gemma-family items: {"label", "box_2d": [ymin, xmin, ymax, xmax]}
    normalized to 0-1000, optional "score". `default_score` is 1.0 because
    MedGemma emits no confidence — a 0.0 default would be filtered out the moment
    an owner sets a positive ANOTMED_MIN_SCORE (see pipeline.py).
    """
    dets: list[Detection] = []
    for item in extract_json_array(text):
        if not isinstance(item, dict):
            continue
        box = item.get("box_2d")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        try:
            bbox = to_pixel_bbox(*box, rows=rows, cols=cols)
        except (TypeError, ValueError):
            continue  # a non-numeric coordinate drops just this box
        try:
            score = float(item.get("score", default_score))
        except (TypeError, ValueError):
            score = default_score
        dets.append(Detection(box=bbox, label=str(item.get("label", "finding")), score=score))
    return dets

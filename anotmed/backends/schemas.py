"""Structured output contract for the vLLM MedGemma backend.

Two halves of the same contract:

  * `guided_schema(n)` — a self-contained JSON Schema handed to vLLM's guided
    decoder so generation is grammar-constrained to well-formed boxes (four ints
    in 0-1000), capped at n findings. Kept free of `$ref`/`$defs` because guided
    decoding is most portable that way.
  * `FindingBox` / `FindingList` — pydantic models that *validate* the response
    we get back (the decoder can be disabled, so we never trust it blindly).

`confidence` is captured as advisory provenance only. LLM self-reported
confidence is uncalibrated and must not gate which findings a radiologist sees.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

_Coord = Annotated[int, Field(ge=0, le=1000)]


class FindingBox(BaseModel):
    box_2d: list[_Coord] = Field(min_length=4, max_length=4)
    label: str = "finding"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FindingList(BaseModel):
    findings: list[FindingBox] = Field(default_factory=list)


def guided_schema(max_findings: int) -> dict:
    """JSON Schema for vLLM `guided_json`, capped at `max_findings` items."""
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": max_findings,
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "label": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["box_2d", "label"],
                },
            }
        },
        "required": ["findings"],
    }

"""Domain types shared by every layer.

These models are the contract between the pipeline, the store, the API, and the
UI. Geometry is kept in two spaces on purpose: pixel coordinates for drawing on
the image, physical millimetres for anything a clinician reads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReviewStatus(str, Enum):
    """A suggestion is PENDING until a human decides. Only ACCEPTED leaves the system."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates. (x, y) is the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    def clamp(self, width: int, height: int) -> "BBox":
        x0 = max(0.0, min(self.x, width - 1))
        y0 = max(0.0, min(self.y, height - 1))
        x1 = max(0.0, min(self.x + self.w, width))
        y1 = max(0.0, min(self.y + self.h, height))
        return BBox(x=x0, y=y0, w=max(0.0, x1 - x0), h=max(0.0, y1 - y0))

    def as_slice(self, width: int, height: int) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for indexing a numpy image."""
        b = self.clamp(width, height)
        r0, c0 = int(round(b.y)), int(round(b.x))
        r1, c1 = int(round(b.y + b.h)), int(round(b.x + b.w))
        return slice(r0, max(r0 + 1, r1)), slice(c0, max(c0 + 1, c1))


class Measurement(BaseModel):
    """Physical measurements derived deterministically from a mask.

    Naming is deliberately literal about what each number is — none of these is a
    certified RECIST value. `max_diameter_mm` is the maximum caliper (Feret)
    diameter; `perpendicular_diameter_mm` is the extent perpendicular to that axis.
    """

    n_pixels: int
    area_mm2: float
    area_cm2: float
    max_diameter_mm: float
    perpendicular_diameter_mm: float
    equivalent_diameter_mm: float
    convex_perimeter_mm: float
    centroid_px: tuple[float, float]  # (row, col)
    bbox_px: BBox
    pixel_spacing_mm: tuple[float, float]  # (row, col) = (dy, dx)


class Provenance(BaseModel):
    """Where a suggestion came from. Every artifact records the model that made it."""

    backend: str
    localizer: str = "unknown"
    segmenter: str = "unknown"
    reporter: str = "unknown"
    score: float | None = None
    created_at: datetime = Field(default_factory=_now)


class Annotation(BaseModel):
    id: str
    study_id: str
    slice_index: int = 0
    label: str
    box: BBox
    mask_path: str | None = None
    measurement: Measurement | None = None
    report_text: str = ""
    status: ReviewStatus = ReviewStatus.PENDING
    edited: bool = False
    reviewer_note: str = ""
    provenance: Provenance


class ImageMeta(BaseModel):
    """Per-image metadata carried alongside the pixel array."""

    study_id: str
    slice_index: int = 0
    rows: int
    cols: int
    pixel_spacing_mm: tuple[float, float] = (1.0, 1.0)  # (dy, dx)
    modality: str = "OT"  # DICOM "other" if unknown
    window_center: float | None = None
    window_width: float | None = None
    spacing_is_default: bool = False  # True when DICOM lacked PixelSpacing


class Study(BaseModel):
    id: str
    source_name: str
    source_path: str | None = None
    meta: ImageMeta
    deidentified: bool = False
    created_at: datetime = Field(default_factory=_now)

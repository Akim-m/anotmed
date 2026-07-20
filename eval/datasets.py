"""Labeled cases for the validation harness.

A `Case` is an image plus ground truth: per-lesion boxes and boolean masks. Two
sources:
  * `synthetic_cases()` — fully synthetic, derived from the same Gaussian lesions
    as examples/make_sample_dicom.py, so the harness runs today with no data.
  * `load_dir()` — the owner's real labeled set (Phase 3b): images + GT masks.

The synthetic ground truth is a filled disk at each lesion centre. It is a
plumbing fixture, not a clinical reference — the stub is not clinical, and neither
is this GT. Its only job is to exercise metrics, floors, and exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anotmed.schema import BBox, ImageMeta

# (cy, cx, r) — MUST match examples/make_sample_dicom.py::synth_image so the stub
# sees the same blobs the ground truth describes.
_BASE_LESIONS = [(90, 100, 14, 900), (170, 165, 9, 820)]


@dataclass
class Case:
    image: np.ndarray
    meta: ImageMeta
    gt_boxes: list[BBox]
    gt_masks: list[np.ndarray]
    labels: list[str]


def _render(lesions, size: int, seed: int) -> np.ndarray:
    """Mirror synth_image: Gaussian blobs + fixed-seed noise → uint16."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.full((size, size), 40.0)
    for cy, cx, r, peak in lesions:
        img += peak * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    img += np.random.default_rng(seed).normal(0, 12, img.shape)
    return np.clip(img, 0, 4095).astype(np.uint16)


def _disk(cy: int, cx: int, r: float, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _mask_bbox(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask)
    return BBox(x=float(xs.min()), y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1), h=float(ys.max() - ys.min() + 1))


def synthetic_cases(n: int = 6, size: int = 256) -> list[Case]:
    """A small deterministic distribution of synthetic cases (n images)."""
    cases: list[Case] = []
    for i in range(n):
        dy, dx = (i % 3 - 1) * 6, (i // 3 - 1) * 5  # deterministic jitter
        lesions = [(cy + dy, cx + dx, r, peak) for cy, cx, r, peak in _BASE_LESIONS]
        image = _render(lesions, size, seed=i).astype(np.float32)
        meta = ImageMeta(study_id=f"synth-{i}", rows=size, cols=size,
                         pixel_spacing_mm=(0.7, 0.7), modality="CT",
                         window_center=400, window_width=1200)
        # GT lesion extent ≈ the Gaussian's full-width-half-maximum radius (1.177·r).
        gt_masks = [_disk(cy, cx, 1.177 * r, size) for cy, cx, r, _ in lesions]
        gt_boxes = [_mask_bbox(m) for m in gt_masks]
        cases.append(Case(image=image, meta=meta, gt_boxes=gt_boxes,
                          gt_masks=gt_masks, labels=["lesion"] * len(lesions)))
    return cases


def load_dir(path: str | Path) -> list[Case]:
    """Load a real labeled set (Phase 3b). Placeholder until the owner wires a set."""
    raise NotImplementedError(
        "Real labeled-set loading is Phase 3b — needs the owner's modality + data layout. "
        "See PLAN.md §4. Use synthetic_cases() for the P3a machinery check."
    )

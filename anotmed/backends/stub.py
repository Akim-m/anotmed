"""Deterministic, CPU-only backend.

Classical computer vision standing in for the neural models: Otsu thresholding
for localization and segmentation, a template for the report. It produces
plausible boxes/masks on real images so the whole pipeline — DICOM I/O,
geometry, review workflow, export — runs and is testable without GPU weights.

It is NOT clinically meaningful. Its only jobs are to exercise the plumbing and
to give the tests something reproducible to assert against.
"""

from __future__ import annotations

import numpy as np

from ..schema import BBox, ImageMeta
from .base import Detection


def _normalize(image: np.ndarray, meta: ImageMeta | None = None) -> np.ndarray:
    """Scale to [0, 1] using the DICOM window if present, else min-max."""
    img = image.astype(np.float64)
    if meta is not None and meta.window_center is not None and meta.window_width:
        lo = meta.window_center - meta.window_width / 2.0
        hi = meta.window_center + meta.window_width / 2.0
    else:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def _otsu(values: np.ndarray) -> float:
    """Otsu's threshold on values already scaled to [0, 1]."""
    hist, edges = np.histogram(values.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    cum_mean = np.cumsum(hist * centers)
    mean_total = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = np.where(w0 > 0, cum_mean / w0, 0.0)
        m1 = np.where(w1 > 0, (mean_total - cum_mean) / w1, 0.0)
        between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.nanargmax(between))])


def _label(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """4-connectivity connected components via iterative flood fill.

    Pure numpy/Python so the stub has no scipy dependency. Returns a label image
    and a {label: pixel_count} map.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes: dict[int, int] = {}
    cur = 0
    h, w = mask.shape
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or labels[sr, sc]:
                continue
            cur += 1
            count = 0
            stack = [(sr, sc)]
            labels[sr, sc] = cur
            while stack:
                r, c = stack.pop()
                count += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = cur
                        stack.append((nr, nc))
            sizes[cur] = count
    return labels, sizes


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, sizes = _label(mask)
    if not sizes:
        return np.zeros_like(mask, dtype=bool)
    biggest = max(sizes, key=sizes.get)
    return labels == biggest


class StubLocalizer:
    name = "stub-otsu-localizer"

    def __init__(self, max_findings: int = 8, min_area_frac: float = 0.0008):
        self.max_findings = max_findings
        self.min_area_frac = min_area_frac

    def propose(self, image: np.ndarray, meta: ImageMeta) -> list[Detection]:
        norm = _normalize(image, meta)
        fg = norm > _otsu(norm)
        labels, sizes = _label(fg)
        h, w = image.shape
        img_area = h * w
        global_mean = float(norm.mean())

        dets: list[Detection] = []
        for lab, size in sizes.items():
            if size < self.min_area_frac * img_area or size > 0.5 * img_area:
                continue  # skip specks and body-sized regions
            ys, xs = np.nonzero(labels == lab)
            # Skip components that hug the image border (usually background).
            if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
                continue
            contrast = float(norm[ys, xs].mean()) - global_mean
            score = max(0.0, min(1.0, contrast)) * min(1.0, size / (0.05 * img_area))
            box = BBox(
                x=float(xs.min()),
                y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1),
                h=float(ys.max() - ys.min() + 1),
            )
            dets.append(Detection(box=box, label="suggested finding", score=round(score, 4)))

        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.max_findings]


class StubSegmenter:
    name = "stub-otsu-segmenter"

    def segment(self, image: np.ndarray, box: BBox, meta: ImageMeta) -> np.ndarray:
        h, w = image.shape
        rs, cs = box.as_slice(w, h)
        crop = _normalize(image, meta)[rs, cs]
        full = np.zeros((h, w), dtype=bool)
        if crop.size == 0:
            return full
        local = crop > _otsu(crop)
        if not local.any():
            local = crop > crop.mean()
        if not local.any():
            local = np.ones_like(crop, dtype=bool)  # fall back to the whole box
        else:
            local = _largest_component(local)
        full[rs, cs] = local
        return full


class StubReporter:
    name = "stub-template-reporter"

    def describe(
        self, image: np.ndarray, det: Detection, mask: np.ndarray, meta: ImageMeta
    ) -> str:
        return (
            f"AI-suggested {det.label} on slice {meta.slice_index} "
            f"(confidence {det.score:.2f}). Draft annotation for a radiologist "
            f"to review and verify — not a diagnosis."
        )

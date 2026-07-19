"""Deterministic geometry engine.

Turns a binary mask plus DICOM pixel spacing into physical measurements. No
machine learning touches this file — the numbers a clinician reads must be
reproducible and auditable, so they come from geometry alone.

All physical lengths are computed in millimetre space to stay correct under
anisotropic spacing (dy != dx), which is common in CT/MR.
"""

from __future__ import annotations

import math

import numpy as np

from .schema import BBox, Measurement

Spacing = tuple[float, float]  # (dy, dx) in mm/pixel


def _boundary_pixels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows/cols of foreground pixels touching the background or the image edge.

    Reducing to the boundary before the convex hull keeps the hull cheap even
    for large regions without changing the result (the hull of a filled shape is
    the hull of its border).
    """
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    boundary = mask & ~interior
    return np.nonzero(boundary)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. `points` is Nx2 (x, y); returns hull vertices CCW."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _feret(hull: np.ndarray) -> tuple[float, float, float]:
    """Return (max_diameter, perpendicular_diameter, perimeter) for hull vertices."""
    n = len(hull)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, 0.0, 0.0

    # Maximum caliper diameter: farthest pair of hull vertices.
    diffs = hull[:, None, :] - hull[None, :, :]
    d2 = (diffs**2).sum(-1)
    i, j = np.unravel_index(int(d2.argmax()), d2.shape)
    max_d = math.sqrt(float(d2[i, j]))

    # Extent perpendicular to the max-diameter axis.
    if max_d > 0:
        axis = (hull[j] - hull[i]) / max_d
        perp = np.array([-axis[1], axis[0]])
        proj = hull @ perp
        perp_d = float(proj.max() - proj.min())
    else:
        perp_d = 0.0

    closed = np.vstack([hull, hull[:1]])
    perimeter = float(np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(-1)).sum())
    return max_d, perp_d, perimeter


def measure_mask(mask: np.ndarray, spacing: Spacing = (1.0, 1.0)) -> Measurement:
    """Measure a boolean mask. Spacing is (dy, dx) in mm/pixel.

    Returns zero-valued measurements for an empty mask rather than raising, so a
    segmenter that finds nothing degrades cleanly.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    dy, dx = float(spacing[0]), float(spacing[1])

    rows, cols = np.nonzero(mask)
    n = int(rows.size)
    if n == 0:
        return Measurement(
            n_pixels=0,
            area_mm2=0.0,
            area_cm2=0.0,
            max_diameter_mm=0.0,
            perpendicular_diameter_mm=0.0,
            equivalent_diameter_mm=0.0,
            convex_perimeter_mm=0.0,
            centroid_px=(0.0, 0.0),
            bbox_px=BBox(x=0, y=0, w=0, h=0),
            pixel_spacing_mm=(dy, dx),
        )

    area_mm2 = n * dy * dx
    centroid = (float(rows.mean()), float(cols.mean()))
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    bbox = BBox(x=float(c0), y=float(r0), w=float(c1 - c0 + 1), h=float(r1 - r0 + 1))

    br, bc = _boundary_pixels(mask)
    # Physical coordinates (x = col*dx, y = row*dy).
    pts = np.column_stack([bc * dx, br * dy])
    hull = _convex_hull(pts)
    max_d, perp_d, perim = _feret(hull)

    return Measurement(
        n_pixels=n,
        area_mm2=area_mm2,
        area_cm2=area_mm2 / 100.0,
        max_diameter_mm=max_d,
        perpendicular_diameter_mm=perp_d,
        equivalent_diameter_mm=2.0 * math.sqrt(area_mm2 / math.pi),
        convex_perimeter_mm=perim,
        centroid_px=centroid,
        bbox_px=bbox,
        pixel_spacing_mm=(dy, dx),
    )


def measure_box(box: BBox, spacing: Spacing = (1.0, 1.0)) -> Measurement:
    """Fallback measurement from a box alone (a filled rectangle).

    Used when no segmenter is available; every number is then a coarse
    over-estimate and should be treated as such.
    """
    dy, dx = float(spacing[0]), float(spacing[1])
    w = max(1, int(round(box.w)))
    h = max(1, int(round(box.h)))
    mask = np.ones((h, w), dtype=bool)
    m = measure_mask(mask, spacing)
    # Re-anchor the bbox/centroid to the box's real position in the image.
    m.bbox_px = BBox(x=box.x, y=box.y, w=float(w), h=float(h))
    m.centroid_px = (box.y + h / 2.0, box.x + w / 2.0)
    return m

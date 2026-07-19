"""The measurement engine against shapes with known geometry.

If these pass, the numbers a clinician reads are trustworthy under both
isotropic and anisotropic spacing.
"""

import math

import numpy as np

from anotmed.measure import measure_box, measure_mask
from anotmed.schema import BBox


def _disk(size, r, cy, cx):
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_circle_area_and_diameter():
    r = 30
    m = measure_mask(_disk(128, r, 64, 64), (1.0, 1.0))
    true_area = math.pi * r * r
    assert abs(m.area_mm2 - true_area) / true_area < 0.03
    assert abs(m.max_diameter_mm - 2 * r) < 2.0
    assert abs(m.equivalent_diameter_mm - 2 * r) < 1.5


def test_anisotropic_area_is_exact():
    mask = np.zeros((64, 64), bool)
    mask[5:25, 10:20] = True  # 20 rows x 10 cols = 200 px
    dy, dx = 2.0, 0.5
    m = measure_mask(mask, (dy, dx))
    assert m.n_pixels == 200
    assert abs(m.area_mm2 - 200 * dy * dx) < 1e-6  # 200.0 mm^2
    expected_diag = math.hypot((20 - 1) * dy, (10 - 1) * dx)
    assert abs(m.max_diameter_mm - expected_diag) < 1.0


def test_empty_mask_is_zeroed_not_raised():
    m = measure_mask(np.zeros((10, 10), bool), (1.0, 1.0))
    assert m.n_pixels == 0
    assert m.area_mm2 == 0.0
    assert m.max_diameter_mm == 0.0


def test_single_pixel():
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    m = measure_mask(mask, (0.5, 0.5))
    assert m.n_pixels == 1
    assert abs(m.area_mm2 - 0.25) < 1e-9
    assert m.max_diameter_mm == 0.0


def test_measure_box_is_anchored_to_image_position():
    m = measure_box(BBox(x=10, y=20, w=8, h=6), (1.0, 1.0))
    assert m.n_pixels == 48
    assert m.bbox_px.x == 10 and m.bbox_px.y == 20

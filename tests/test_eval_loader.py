"""Loading a real labeled set into the validation harness (Phase 3b hook).

Layout (modality-agnostic, 2D):
    <dir>/images/<stem>.{png,npy}
    <dir>/masks/<stem>.{png,npy}   # 0 = background
        - binary mask  -> each connected foreground region is one lesion
        - label map    -> each distinct nonzero value is one lesion
Each lesion yields a GT mask + its bounding box, so eval/run.py can score
segmentation (GT-box-prompted) and localization on real data.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eval.datasets import Case, load_dir


def _make(root, img, mask, stem="c1", mask_ext="png"):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(root / "images" / f"{stem}.png")
    if mask_ext == "png":
        Image.fromarray(mask.astype(np.uint8)).save(root / "masks" / f"{stem}.png")
    else:
        np.save(root / "masks" / f"{stem}.npy", mask)


def _two_blob(rows=32, cols=32):
    img = np.zeros((rows, cols), np.uint8)
    img[5:10, 5:10] = 200
    img[20:26, 20:26] = 180
    return img


def test_binary_mask_splits_into_connected_lesions(tmp_path):
    img = _two_blob()
    mask = np.zeros((32, 32), np.uint8)
    mask[5:10, 5:10] = 1
    mask[20:26, 20:26] = 1  # two separate blobs, both value 1
    _make(tmp_path, img, mask)

    cases = load_dir(tmp_path)
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, Case)
    assert len(c.gt_masks) == 2 and len(c.gt_boxes) == 2
    for m in c.gt_masks:
        assert m.dtype == np.bool_ and m.shape == (32, 32) and m.any()
    # the two boxes are disjoint (different lesions)
    b0, b1 = sorted(c.gt_boxes, key=lambda b: b.x)
    assert b0.x + b0.w <= b1.x + 1


def test_label_map_uses_distinct_values_as_instances(tmp_path):
    img = _two_blob()
    mask = np.zeros((32, 32), np.uint8)
    mask[5:10, 5:10] = 1
    mask[20:26, 20:26] = 2  # explicit instance ids
    _make(tmp_path, img, mask)

    c = load_dir(tmp_path)[0]
    assert len(c.gt_masks) == 2


def test_spacing_is_applied_to_meta(tmp_path):
    _make(tmp_path, _two_blob(), (_two_blob() > 0).astype(np.uint8))
    c = load_dir(tmp_path, spacing=(0.7, 0.5))[0]
    assert c.meta.pixel_spacing_mm == (0.7, 0.5)
    assert c.meta.rows == 32 and c.meta.cols == 32


def test_npy_mask_is_supported(tmp_path):
    img = _two_blob()
    mask = np.zeros((32, 32), np.uint8)
    mask[5:10, 5:10] = 1
    _make(tmp_path, img, mask, mask_ext="npy")
    c = load_dir(tmp_path)[0]
    assert len(c.gt_masks) == 1


def test_empty_mask_yields_a_case_with_no_lesions(tmp_path):
    _make(tmp_path, _two_blob(), np.zeros((32, 32), np.uint8))
    c = load_dir(tmp_path)[0]
    assert c.gt_masks == [] and c.gt_boxes == []


def test_missing_mask_raises_actionable_error(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "masks").mkdir(parents=True)
    Image.fromarray(_two_blob()).save(tmp_path / "images" / "lonely.png")
    with pytest.raises(FileNotFoundError) as exc:
        load_dir(tmp_path)
    assert "lonely" in str(exc.value)

"""Validation metrics — exact values on known shapes.

These are the numbers the safety gate is built on, so they are pinned to hand-
computed ground truth (identical/disjoint/half-overlap), not golden snapshots.
Pure numpy, no GPU, no model.
"""

from __future__ import annotations

import numpy as np

from anotmed.schema import BBox
from eval.metrics import box_iou, dice, iou, localization_scores, valid_mask


def _mask(rows_true: range, size: int = 10) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[list(rows_true), :] = True
    return m


# ---- Dice / IoU -------------------------------------------------------------

def test_dice_identical_is_one():
    m = _mask(range(0, 10))
    assert dice(m, m) == 1.0


def test_dice_disjoint_is_zero():
    a = _mask(range(0, 5))
    b = _mask(range(5, 10))
    assert dice(a, b) == 0.0


def test_dice_half_overlap_is_one_half():
    a = _mask(range(0, 10))   # 100 px
    b = _mask(range(5, 15), size=15)  # different size on purpose? keep same grid
    # keep both 10x10 for a clean number:
    a = _mask(range(0, 10))
    b = _mask(range(5, 10))
    b2 = np.zeros((10, 10), bool)
    b2[5:15 - 5, :] = True  # rows 5..9 -> 50 px
    inter = 50.0
    assert dice(a, b2) == 2 * inter / (100 + 50)


def test_iou_half_overlap():
    a = _mask(range(0, 10))       # 100 px
    b = np.zeros((10, 10), bool)
    b[5:10, :] = True             # 50 px, all inside a
    # intersection 50, union 100 -> 0.5
    assert iou(a, b) == 0.5


def test_dice_and_iou_two_empty_masks_agree():
    z = np.zeros((8, 8), bool)
    assert dice(z, z) == 1.0
    assert iou(z, z) == 1.0


# ---- box IoU ----------------------------------------------------------------

def test_box_iou_identical_is_one():
    b = BBox(x=10, y=10, w=20, h=20)
    assert box_iou(b, b) == 1.0


def test_box_iou_disjoint_is_zero():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=50, y=50, w=10, h=10)
    assert box_iou(a, b) == 0.0


def test_box_iou_known_partial():
    a = BBox(x=0, y=0, w=10, h=10)   # area 100
    b = BBox(x=5, y=0, w=10, h=10)   # area 100, overlap 5x10 = 50
    # union = 150, iou = 50/150 = 1/3
    assert box_iou(a, b) == 50 / 150


# ---- localization matching --------------------------------------------------

def test_localization_perfect_recall_and_precision():
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    s = localization_scores(pred, gt, iou_thresh=0.5)
    assert s["recall"] == 1.0 and s["precision"] == 1.0


def test_localization_missed_gt_lowers_recall():
    gt = [BBox(x=0, y=0, w=10, h=10), BBox(x=50, y=50, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10)]  # found only the first
    s = localization_scores(pred, gt, iou_thresh=0.5)
    assert s["recall"] == 0.5
    assert s["precision"] == 1.0


def test_localization_spurious_pred_lowers_precision():
    gt = [BBox(x=0, y=0, w=10, h=10)]
    pred = [BBox(x=0, y=0, w=10, h=10), BBox(x=90, y=90, w=5, h=5)]  # one false positive
    s = localization_scores(pred, gt, iou_thresh=0.5)
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5


# ---- format compliance ------------------------------------------------------

def test_valid_mask_accepts_correct_shape_bool():
    assert valid_mask(np.zeros((16, 16), bool), 16, 16)


def test_valid_mask_rejects_wrong_shape():
    assert not valid_mask(np.zeros((8, 8), bool), 16, 16)


def test_valid_mask_rejects_non_array():
    assert not valid_mask(None, 16, 16)

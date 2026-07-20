"""Absolute segmentation + localization metrics. Pure numpy, dependency-light.

Two things are kept deliberately separate (see PLAN.md Phase 3):
  * **quality-given-valid** — Dice/IoU on outputs the backend actually produced,
  * **format compliance** — how often it produced a usable output at all.
A model that is accurate but crashes on 1-in-5 images must not hide behind a good
mean Dice, so `valid_mask` is scored on its own and the quality metrics are only
averaged over valid cases.
"""

from __future__ import annotations

import numpy as np

from anotmed.schema import BBox


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen–Dice of two boolean masks. Two empty masks agree (1.0)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. Two empty masks agree (1.0)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(a, b).sum()) / union


def box_iou(a: BBox, b: BBox) -> float:
    """IoU of two axis-aligned boxes in pixel space."""
    ix0, iy0 = max(a.x, b.x), max(a.y, b.y)
    ix1, iy1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def localization_scores(
    pred_boxes: list[BBox], gt_boxes: list[BBox], iou_thresh: float = 0.5
) -> dict:
    """Greedy box matching → recall / precision at an IoU threshold.

    Each GT box claims the best still-unmatched prediction that clears the
    threshold. recall = matched GT / all GT; precision = matched pred / all pred.
    Empty-vs-empty is treated as a perfect (1.0) score.
    """
    matched_pred: set[int] = set()
    tp = 0
    for gt in gt_boxes:
        best_j, best_iou = -1, iou_thresh
        for j, pb in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            v = box_iou(gt, pb)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched_pred.add(best_j)
            tp += 1
    recall = tp / len(gt_boxes) if gt_boxes else 1.0
    precision = tp / len(pred_boxes) if pred_boxes else (1.0 if not gt_boxes else 0.0)
    return {
        "recall": recall,
        "precision": precision,
        "tp": tp,
        "n_gt": len(gt_boxes),
        "n_pred": len(pred_boxes),
    }


def valid_mask(mask, rows: int, cols: int) -> bool:
    """Format compliance: is this a usable boolean HxW mask matching the image?"""
    return (
        isinstance(mask, np.ndarray)
        and mask.shape == (rows, cols)
        and mask.dtype == np.bool_
    )

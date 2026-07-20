"""The safety gate itself: synthetic ground truth, scoring, and floor enforcement.

Proves the harness machinery on the stub backend (zero GPU) and — critically —
proves the gate *catches bad output*: a segmenter that returns empty masks must
trip the Dice floor even though its format compliance stays perfect. That last
point is the whole reason the two are scored separately.
"""

from __future__ import annotations

import numpy as np

from anotmed.backends import build_backend
from anotmed.config import Config
from eval.datasets import synthetic_cases
from eval.run import Floors, Report, evaluate


def _stub_backend():
    return build_backend(Config(backend="stub"))


def _clinical_floors() -> Floors:
    return Floors(dice_mean=0.85, dice_p10=0.70, recall_iou30=0.80, format_compliance=0.98)


# ---- synthetic ground truth -------------------------------------------------

def test_synthetic_cases_carry_matching_boxes_and_masks():
    cases = synthetic_cases(n=4)
    assert len(cases) == 4
    for c in cases:
        assert len(c.gt_boxes) == len(c.gt_masks) >= 1
        assert c.gt_masks[0].dtype == np.bool_
        assert c.gt_masks[0].shape == (c.meta.rows, c.meta.cols)
        assert c.gt_masks[0].any()  # a real lesion, not empty


# ---- machinery runs end-to-end ----------------------------------------------

def test_evaluate_stub_produces_a_wellformed_report():
    report = evaluate(_stub_backend(), synthetic_cases(n=4))
    assert report.n_cases == 4 and report.n_lesions >= 4
    assert report.format_compliance == 1.0  # stub always returns a valid mask
    assert 0.0 <= report.dice_mean <= 1.0
    assert 0.0 <= report.recall_iou30 <= 1.0


def test_stub_evaluation_is_deterministic():
    b = _stub_backend()
    cases = synthetic_cases(n=3)
    r1, r2 = evaluate(b, cases), evaluate(b, cases)
    assert r1.dice_mean == r2.dice_mean
    assert r1.recall_iou30 == r2.recall_iou30


# ---- the gate logic ---------------------------------------------------------

def test_floors_pass_when_all_metrics_clear_them():
    r = Report(n_cases=1, n_lesions=2, format_compliance=1.0, dice_mean=0.9,
               dice_p10=0.8, iou_mean=0.85, recall_iou30=0.9, recall_iou50=0.8,
               precision_iou30=0.9)
    assert _clinical_floors().check(r) == []


def test_floors_flag_low_dice_and_recall():
    r = Report(n_cases=1, n_lesions=2, format_compliance=1.0, dice_mean=0.5,
               dice_p10=0.3, iou_mean=0.4, recall_iou30=0.5, recall_iou50=0.4,
               precision_iou30=0.5)
    violations = _clinical_floors().check(r)
    assert any("dice" in v.lower() for v in violations)
    assert any("recall" in v.lower() for v in violations)


def test_floors_load_merges_a_modality_section(tmp_path):
    y = tmp_path / "floors.yaml"
    y.write_text(
        "segmentation: {dice_mean: 0.85, dice_p10: 0.70}\n"
        "localization: {recall_iou30: 0.80}\n"
        "format: {compliance: 0.98}\n"
        "modalities:\n"
        "  dental:\n"
        "    segmentation: {dice_mean: 0.0, dice_p10: 0.0}\n"
        "    localization: {recall_iou30: 0.85}\n"
    )
    g = Floors.load(y)
    assert g.dice_mean == 0.85 and g.recall_iou30 == 0.80

    d = Floors.load(y, modality="dental")
    assert d.dice_mean == 0.0 and d.dice_p10 == 0.0  # box-only: seg not gated
    assert d.recall_iou30 == 0.85                     # dental localization floor
    assert d.format_compliance == 0.98                # inherited from global


def test_floors_load_unknown_modality_falls_back_to_global(tmp_path):
    y = tmp_path / "floors.yaml"
    y.write_text(
        "segmentation: {dice_mean: 0.85, dice_p10: 0.70}\n"
        "localization: {recall_iou30: 0.80}\n"
        "format: {compliance: 0.98}\n"
    )
    d = Floors.load(y, modality="nope")
    assert d.dice_mean == 0.85 and d.recall_iou30 == 0.80  # strict global baseline


def test_corrupted_segmenter_trips_the_gate_despite_valid_format():
    # Empty masks are a *valid* shape/dtype (format compliance 1.0) but score
    # Dice ~0 against real lesions — the gate must still fail.
    class _EmptySeg:
        name = "empty-seg"

        def segment(self, image, box, meta):
            return np.zeros(image.shape, dtype=bool)

    backend = _stub_backend()
    backend.segmenter = _EmptySeg()
    report = evaluate(backend, synthetic_cases(n=4))
    assert report.format_compliance == 1.0     # format is fine...
    assert report.dice_mean < 0.85             # ...but quality collapses
    assert _clinical_floors().check(report)    # gate trips

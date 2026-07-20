"""Dental detector Localizer — the box->Detection contract, testable on CPU.

A dedicated object detector replaces MedGemma as the Localizer (MedGemma proved a
describer, not a detector). This tests the pure conversion seam and the localizer
with an INJECTED fake predictor, so no torch/YOLO weights are needed — exactly the
pattern used for the vLLM client and the SAM2 segmenter.
"""

from __future__ import annotations

import numpy as np

from anotmed.backends.base import Detection
from anotmed.backends.detector import DetectorLocalizer, boxes_to_detections
from anotmed.config import Config
from anotmed.schema import ImageMeta


# ---- fakes: mimic ultralytics result.boxes.{xyxy,conf,cls} + result.names -----

class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)


class _FakeResult:
    def __init__(self, xyxy, conf, cls, names):
        self.boxes = _FakeBoxes(xyxy, conf, cls)
        self.names = names


class _FakePredictor:
    def __init__(self, xyxy=(), conf=(), cls=(), names=None, raise_exc=None):
        self._result = _FakeResult(xyxy, conf, cls, names or {0: "tooth"})
        self.raise_exc = raise_exc
        self.last_kwargs = None

    def predict(self, rgb, **kwargs):
        self.last_kwargs = kwargs
        if self.raise_exc:
            raise self.raise_exc
        return [self._result]


def _cfg(**kw):
    return Config(backend="detector+medsam", detector_weights="/w/dentex.pt", **kw)


def _meta():
    return ImageMeta(study_id="s", rows=100, cols=100, modality="OT")


# ---- boxes_to_detections (pure, no torch) -----------------------------------

def test_xyxy_becomes_pixel_bbox_with_label_and_score():
    dets = boxes_to_detections(
        xyxy=np.array([[10, 20, 40, 60]]), conf=np.array([0.9]), cls=np.array([0]),
        names={0: "tooth"}, rows=100, cols=100, min_conf=0.25, max_findings=8)
    assert len(dets) == 1
    d = dets[0]
    assert (d.box.x, d.box.y, d.box.w, d.box.h) == (10.0, 20.0, 30.0, 40.0)
    assert d.label == "tooth" and d.score == 0.9


def test_low_confidence_boxes_are_dropped():
    dets = boxes_to_detections(
        xyxy=np.array([[0, 0, 10, 10], [0, 0, 20, 20]]), conf=np.array([0.1, 0.8]),
        cls=np.array([0, 0]), names={0: "tooth"}, rows=100, cols=100,
        min_conf=0.25, max_findings=8)
    assert [d.score for d in dets] == [0.8]


def test_results_sorted_by_score_and_capped():
    dets = boxes_to_detections(
        xyxy=np.array([[0, 0, 5, 5], [0, 0, 6, 6], [0, 0, 7, 7]]),
        conf=np.array([0.3, 0.9, 0.6]), cls=np.array([0, 0, 0]), names={0: "t"},
        rows=100, cols=100, min_conf=0.0, max_findings=2)
    assert [d.score for d in dets] == [0.9, 0.6]  # sorted desc, capped at 2


def test_box_is_clamped_into_the_image():
    dets = boxes_to_detections(
        xyxy=np.array([[-10, -10, 200, 200]]), conf=np.array([0.9]), cls=np.array([0]),
        names={0: "t"}, rows=100, cols=100, min_conf=0.0, max_findings=8)
    b = dets[0].box
    assert b.x >= 0 and b.y >= 0 and b.x + b.w <= 100 and b.y + b.h <= 100


def test_label_falls_back_when_class_missing_from_names():
    dets = boxes_to_detections(
        xyxy=np.array([[0, 0, 5, 5]]), conf=np.array([0.9]), cls=np.array([7]),
        names={0: "tooth"}, rows=100, cols=100, min_conf=0.0, max_findings=8)
    assert dets[0].label == "finding"


def test_no_boxes_yields_empty_list():
    assert boxes_to_detections(np.empty((0, 4)), np.array([]), np.array([]),
                               {}, 100, 100, 0.25, 8) == []


# ---- DetectorLocalizer.propose (injected predictor, no weights) --------------

def test_propose_runs_detector_and_returns_detections():
    pred = _FakePredictor(xyxy=[[10, 20, 40, 60]], conf=[0.8], cls=[0], names={0: "tooth"})
    dets = DetectorLocalizer(_cfg(), predictor=pred).propose(
        np.zeros((100, 100), np.float32), _meta())
    assert len(dets) == 1 and isinstance(dets[0], Detection) and dets[0].label == "tooth"


def test_propose_forwards_conf_and_imgsz_to_predict():
    pred = _FakePredictor(names={0: "tooth"})
    DetectorLocalizer(_cfg(detector_conf=0.3, detector_imgsz=1024), predictor=pred).propose(
        np.zeros((100, 100), np.float32), _meta())
    assert pred.last_kwargs["conf"] == 0.3 and pred.last_kwargs["imgsz"] == 1024


def test_propose_never_raises_returns_empty_on_detector_error():
    pred = _FakePredictor(raise_exc=RuntimeError("cuda blew up"))
    assert DetectorLocalizer(_cfg(), predictor=pred).propose(
        np.zeros((100, 100), np.float32), _meta()) == []


def test_localizer_name_reflects_the_weights():
    assert DetectorLocalizer(_cfg(), predictor=_FakePredictor()).name == "detector:dentex.pt"


# ---- build_backend wiring: detector Localizer + MedGemma Reporter -------------

def test_build_backend_routes_detector_with_medgemma_as_reporter(monkeypatch):
    from anotmed.backends import build_backend, detector, medsam, vllm_medgemma

    class _Loc:
        name = "detector:dentex.pt"

    class _Seg:
        name = "seg"

    class _Client:
        def health(self):
            pass

    class _Rep:
        name = "vllm-reporter"
        client = _Client()

    monkeypatch.setattr(detector, "build", lambda cfg: _Loc())
    monkeypatch.setattr(medsam, "build", lambda cfg: _Seg())
    monkeypatch.setattr(vllm_medgemma, "build", lambda cfg: (object(), _Rep()))

    be = build_backend(Config(backend="detector+medsam"))
    assert be.name == "detector+medsam"
    assert be.localizer.name.startswith("detector:")
    assert be.reporter.name == "vllm-reporter"  # MedGemma is the reporter, not the localizer

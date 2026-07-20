"""Object-detector Localizer — the dedicated detector that replaces MedGemma.

Real-data validation showed MedGemma describes images well but localizes lesions
poorly (recall 0.37 on endoscopy, 0.00 on CT). A purpose-built detector does the
boxing; MedGemma stays on as the Reporter (its proven strength). This backend
implements the `Localizer` protocol: image -> a list of Detections.

The heavy dependency (ultralytics/YOLO) is imported lazily inside `_load`, behind
a config guard, so the CPU test suite never needs torch. The box->Detection
conversion is a pure, torch-free function (`boxes_to_detections`) that is fully
unit-tested by mocking the detector's raw output — same seam as the vLLM client.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np

from ..config import Config
from ..schema import BBox, ImageMeta
from .base import Detection

log = logging.getLogger("anotmed.backends.detector")


def _load(cfg: Config):
    # Check config BEFORE importing ultralytics, so a missing weights path yields
    # an actionable error, not a cryptic ImportError from an absent framework.
    if not cfg.detector_weights:
        raise RuntimeError("Detector needs ANOTMED_DETECTOR_WEIGHTS set.")
    from ultralytics import YOLO

    return YOLO(cfg.detector_weights)


def _np(x) -> np.ndarray:
    """Detector arrays may be torch tensors or numpy; normalize to numpy."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def boxes_to_detections(xyxy, conf, cls, names, rows: int, cols: int,
                        min_conf: float, max_findings: int,
                        label_map: Mapping[str, str] | None = None) -> list[Detection]:
    """Convert a detector's (xyxy, conf, cls) output to clamped Detections.

    Pixel xyxy (detector-native) -> top-left BBox clamped into the image; class id
    -> raw name via `names`, then remapped to a clinical finding name via
    `label_map` (e.g. the single-class YOLO name "item" -> "tooth"); confidence ->
    score. Drops sub-threshold boxes, sorts by score, caps at `max_findings`. Pure
    numpy — never raises on well-formed input.
    """
    dets: list[Detection] = []
    for (x0, y0, x1, y1), v, c in zip(np.asarray(xyxy).reshape(-1, 4), conf, cls):
        score = float(v)
        if score < min_conf:
            continue
        box = BBox(x=float(x0), y=float(y0),
                   w=max(1.0, float(x1) - float(x0)),
                   h=max(1.0, float(y1) - float(y0))).clamp(cols, rows)
        raw = str(names.get(int(c), "finding"))
        label = label_map.get(raw, raw) if label_map else raw
        dets.append(Detection(box=box, label=label, score=score))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:max_findings]


class DetectorLocalizer:
    def __init__(self, cfg: Config, predictor=None):
        self.cfg = cfg
        self._predictor = predictor if predictor is not None else _load(cfg)
        base = f"detector:{Path(cfg.detector_weights).name or 'yolo'}"
        # tag the modality for the audit trail (Provenance.localizer) when one is active
        self.name = f"{base}[{cfg.modality}]" if cfg.modality else base

    def propose(self, image: np.ndarray, meta: ImageMeta) -> list[Detection]:
        from ..io_dicom import to_display_array
        from ..modalities import get_profile

        try:
            rgb = to_display_array(image, meta)  # HxWx3 uint8; detector-native pixels
            result = self._predictor.predict(
                rgb, imgsz=self.cfg.detector_imgsz, conf=self.cfg.detector_conf,
                device=self.cfg.detector_device, verbose=False)[0]
            boxes = result.boxes
            return boxes_to_detections(
                _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls), result.names,
                meta.rows, meta.cols, self.cfg.detector_conf, self.cfg.max_findings,
                label_map=get_profile(self.cfg.modality).label_map)
        except Exception:  # a detector failure must not crash the study upload
            log.exception("detector failed for study %s", meta.study_id)
            return []


def build(cfg: Config) -> DetectorLocalizer:
    return DetectorLocalizer(cfg)

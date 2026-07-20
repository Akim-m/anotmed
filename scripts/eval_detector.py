#!/usr/bin/env python
"""Score a trained detector's localization on a COCO detection set, per modality.

Runs the anotmed DetectorLocalizer over held-out boxes and reports recall/precision
against the modality's localization floor — the head-to-head with any VLM localizer.
Uses ONLY the detector (memory-light: no SAM2/MedGemma resident).

env: IMAGES_DIR, DATA_COCO, WEIGHTS (or ANOTMED_DETECTOR_WEIGHTS), MODALITY, LIMIT.
example (dental): MODALITY=dental IMAGES_DIR=.../xrays DATA_COCO=.../heldout.json \\
                  WEIGHTS=/root/dental_weights/dentex_tooth/weights/best.pt python scripts/eval_detector.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anotmed.config import Config
from anotmed.backends.detector import DetectorLocalizer
from anotmed.modalities import get_profile
from eval.datasets import load_coco_boxes
from eval.run import Floors, localization_metrics

MODALITY = os.environ.get("MODALITY", "")
os.environ["ANOTMED_MODALITY"] = MODALITY  # so profile-backed Config defaults apply
prof = get_profile(MODALITY)
limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None

cases = load_coco_boxes(os.environ["IMAGES_DIR"], os.environ["DATA_COCO"],
                        modality=prof.dicom_modality, window=prof.window, limit=limit)
print(f"{len(cases)} cases, {sum(len(c.gt_boxes) for c in cases)} GT boxes  (modality={MODALITY or 'generic'})")

cfg = Config(backend="detector+medsam", modality=MODALITY, detector_device="cuda",
             detector_weights=os.environ.get("WEIGHTS", os.environ.get("ANOTMED_DETECTOR_WEIGHTS", "")))
m = localization_metrics(DetectorLocalizer(cfg), cases)
floor = Floors.load(modality=MODALITY).recall_iou30
print(f"  recall @0.3 {m['recall_iou30']:.3f}  @0.5 {m['recall_iou50']:.3f}  precision {m['precision_iou30']:.3f}")
print(f"  localization floor (recall@0.3 >= {floor}): "
      f"{'PASS' if m['recall_iou30'] >= floor else 'MISS'}")

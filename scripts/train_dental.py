#!/usr/bin/env python
"""Train a YOLOv8n tooth detector on DENTEX (the Localizer for the dental backend).

Memory-safe for an 8 GiB GPU under the >=1 GiB-free rule: nano model, small batch,
cache=False (stream from disk — never bulk-load images into the ~11 GiB RAM),
AMP. Kill any vLLM server first (never co-resident). Weights land outside the repo
at /root/dental_weights/, like the SAM2 checkpoints.

Env: DATA_YAML (ultralytics data.yaml), EPOCHS, BATCH, IMGSZ, SINGLE_CLS, OUT, NAME.

DENTEX is CC-BY-NC-SA-4.0 (non-commercial, attribution) and ultralytics is
AGPL-3.0 — the resulting weights are research/validation only. Cite Hamamci et al.,
arXiv:2305.19112 (DENTEX, MICCAI 2023).
"""
import os
from ultralytics import YOLO

DATA = os.environ["DATA_YAML"]
model = YOLO("yolov8n.pt")  # COCO-pretrained init (auto-downloaded)
model.train(
    data=DATA,
    epochs=int(os.environ.get("EPOCHS", "60")),
    imgsz=int(os.environ.get("IMGSZ", "1024")),
    rect=True,                                   # keep panoramic aspect (no square squash)
    batch=int(os.environ.get("BATCH", "4")),     # 4 @ 1024 ~ 3-4 GiB; raise only if free > 1 GiB
    cache=False,                                 # stream from disk -> protect the RAM floor
    workers=4,
    amp=True,
    single_cls=os.environ.get("SINGLE_CLS", "1") == "1",  # milestone 1: one "tooth" class
    patience=int(os.environ.get("PATIENCE", "15")),
    device=0,
    project=os.environ.get("OUT", "/root/dental_weights"),
    name=os.environ.get("NAME", "dentex_tooth"),
    exist_ok=True,
    plots=False,
)
print("done. best weights under", os.environ.get("OUT", "/root/dental_weights"))

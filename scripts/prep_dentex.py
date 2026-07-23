#!/usr/bin/env python
"""Convert a DENTEX subset to a YOLO training layout (config-driven).

Milestone 1 (default): single-class tooth detection (quadrant_enumeration).
Milestone 2: 4-class pathology (quadrant-enumeration-disease, category_id_3 =
Impacted/Caries/Periapical Lesion/Deep Caries).

Deterministic 70/15/15 split; images symlinked (large panoramics); the held-out
split is written as a COCO json (its `category_id` set to the chosen field) so the
eval loads it via eval.datasets.load_coco_boxes.

env: DENTEX_ROOT (raw data root), PREP_OUT (output root), SUBSET, JSON,
     CATFIELD ("" = single class), NAMES (comma-sep), OUT_NAME, HELDOUT_JSON.
     DENTEX is CC-BY-NC-SA-4.0; cite Hamamci et al., arXiv:2305.19112.

Paths default to /root/dental_data (durable, like /root/dental_weights) — NEVER a
session scratchpad, which is garbage-collected when the session ends and would make
every trained run non-reproducible.
"""
import json
import os
import random
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DENTEX_ROOT", "/root/dental_data/dentex"))  # raw DENTEX subset
OUT_ROOT = Path(os.environ.get("PREP_OUT", "/root/dental_data"))             # YOLO layout + held-out json
SUBSET = os.environ.get("SUBSET", "quadrant_enumeration")
JSON = os.environ.get("JSON", "train_quadrant_enumeration.json")
CATFIELD = os.environ.get("CATFIELD", "")                       # "" -> single class 0
NAMES = os.environ.get("NAMES", "tooth").split(",")
OUT = OUT_ROOT / os.environ.get("OUT_NAME", "dentex_yolo")
HELDOUT = OUT_ROOT / os.environ.get("HELDOUT_JSON", "dentex_heldout.json")

BASE = DATA_ROOT / "training_data" / SUBSET
XRAYS = BASE / "xrays"
COCO = json.loads((BASE / JSON).read_text())

imgs = {im["id"]: im for im in COCO["images"]}
anns_by_img: dict = {}
for a in COCO["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)


def cls_of(ann) -> int:
    return int(ann[CATFIELD]) if CATFIELD else 0


ids = sorted(imgs)
random.Random(0).shuffle(ids)
n = len(ids)
splits = {"train": ids[: int(0.70 * n)], "val": ids[int(0.70 * n): int(0.85 * n)],
          "heldout": ids[int(0.85 * n):]}
print({k: len(v) for k, v in splits.items()}, "classes:", NAMES)

for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img_id in splits[split]:
        im = imgs[img_id]
        link = OUT / "images" / split / im["file_name"]
        if not link.exists():
            os.symlink(XRAYS / im["file_name"], link)
        W, H = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            lines.append(f"{cls_of(a)} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (OUT / "labels" / split / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines))

names_yaml = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(NAMES))
(OUT / "data.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: {len(NAMES)}\nnames:\n{names_yaml}\n")

held_imgs = splits["heldout"]
held = {"images": [imgs[i] for i in held_imgs],
        "categories": [{"id": i, "name": nm} for i, nm in enumerate(NAMES)],
        "annotations": [{**a, "category_id": cls_of(a)} for i in held_imgs for a in anns_by_img.get(i, [])]}
HELDOUT.write_text(json.dumps(held))
print(f"data.yaml -> {OUT/'data.yaml'}  |  held-out -> {HELDOUT} ({len(held_imgs)} imgs), images at {XRAYS}")

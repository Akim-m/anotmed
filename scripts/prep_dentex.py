#!/usr/bin/env python
"""Convert the DENTEX tooth-enumeration subset to a YOLO training layout.

Milestone 1: single-class tooth detection (collapse all teeth to class 0 — the
localization gate ignores labels anyway). Deterministic 70/15/15 split. Images
are symlinked (not copied — they're large panoramics). The held-out split is
written as a COCO json so the eval loads it via eval.datasets.load_coco_boxes.

DENTEX is CC-BY-NC-SA-4.0 (non-commercial). Cite Hamamci et al., arXiv:2305.19112.
"""
import json
import os
import random
from pathlib import Path

SCR = Path("/tmp/claude-0/-home-akim-Coding-anotmed/b81e55b1-b9d2-4d01-a250-cb39c46af05b/scratchpad")
BASE = SCR / "dentex/training_data/quadrant_enumeration"
XRAYS = BASE / "xrays"
COCO = json.loads((BASE / "train_quadrant_enumeration.json").read_text())
OUT = SCR / "dentex_yolo"

imgs = {im["id"]: im for im in COCO["images"]}
anns_by_img: dict = {}
for a in COCO["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

ids = sorted(imgs)
random.Random(0).shuffle(ids)
n = len(ids)
splits = {"train": ids[: int(0.70 * n)],
          "val": ids[int(0.70 * n): int(0.85 * n)],
          "heldout": ids[int(0.85 * n):]}
print({k: len(v) for k, v in splits.items()})

for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img_id in splits[split]:
        im = imgs[img_id]
        stem = Path(im["file_name"]).stem
        link = OUT / "images" / split / im["file_name"]
        if not link.exists():
            os.symlink(XRAYS / im["file_name"], link)
        W, H = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            cx, cy, nw, nh = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

(OUT / "data.yaml").write_text(
    f"path: {OUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: tooth\n")

# held-out as COCO (images loaded from the shared xrays dir by load_coco_boxes)
held = {"images": [imgs[i] for i in splits["heldout"]],
        "categories": [{"id": 0, "name": "tooth"}],
        "annotations": [a for i in splits["heldout"] for a in anns_by_img.get(i, [])]}
(SCR / "dentex_heldout.json").write_text(json.dumps(held))
print(f"data.yaml -> {OUT/'data.yaml'}")
print(f"held-out COCO -> {SCR/'dentex_heldout.json'} ({len(held['images'])} imgs), "
      f"images at {XRAYS}")

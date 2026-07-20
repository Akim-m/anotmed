"""Labeled cases for the validation harness.

A `Case` is an image plus ground truth: per-lesion boxes and boolean masks. Two
sources:
  * `synthetic_cases()` — fully synthetic, derived from the same Gaussian lesions
    as examples/make_sample_dicom.py, so the harness runs today with no data.
  * `load_dir()` — the owner's real labeled set (Phase 3b): images + GT masks.

The synthetic ground truth is a filled disk at each lesion centre. It is a
plumbing fixture, not a clinical reference — the stub is not clinical, and neither
is this GT. Its only job is to exercise metrics, floors, and exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anotmed.modalities import WindowSpec, resolve_window
from anotmed.schema import BBox, ImageMeta

# (cy, cx, r) — MUST match examples/make_sample_dicom.py::synth_image so the stub
# sees the same blobs the ground truth describes.
_BASE_LESIONS = [(90, 100, 14, 900), (170, 165, 9, 820)]


@dataclass
class Case:
    image: np.ndarray
    meta: ImageMeta
    gt_boxes: list[BBox]
    gt_masks: list[np.ndarray]
    labels: list[str]


def _render(lesions, size: int, seed: int) -> np.ndarray:
    """Mirror synth_image: Gaussian blobs + fixed-seed noise → uint16."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.full((size, size), 40.0)
    for cy, cx, r, peak in lesions:
        img += peak * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    img += np.random.default_rng(seed).normal(0, 12, img.shape)
    return np.clip(img, 0, 4095).astype(np.uint16)


def _disk(cy: int, cx: int, r: float, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _mask_bbox(mask: np.ndarray) -> BBox:
    ys, xs = np.where(mask)
    return BBox(x=float(xs.min()), y=float(ys.min()),
                w=float(xs.max() - xs.min() + 1), h=float(ys.max() - ys.min() + 1))


def synthetic_cases(n: int = 6, size: int = 256) -> list[Case]:
    """A small deterministic distribution of synthetic cases (n images)."""
    cases: list[Case] = []
    for i in range(n):
        dy, dx = (i % 3 - 1) * 6, (i // 3 - 1) * 5  # deterministic jitter
        lesions = [(cy + dy, cx + dx, r, peak) for cy, cx, r, peak in _BASE_LESIONS]
        image = _render(lesions, size, seed=i).astype(np.float32)
        meta = ImageMeta(study_id=f"synth-{i}", rows=size, cols=size,
                         pixel_spacing_mm=(0.7, 0.7), modality="CT",
                         window_center=400, window_width=1200)
        # GT lesion extent ≈ the Gaussian's full-width-half-maximum radius (1.177·r).
        gt_masks = [_disk(cy, cx, 1.177 * r, size) for cy, cx, r, _ in lesions]
        gt_boxes = [_mask_bbox(m) for m in gt_masks]
        cases.append(Case(image=image, meta=meta, gt_boxes=gt_boxes,
                          gt_masks=gt_masks, labels=["lesion"] * len(lesions)))
    return cases


def _load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image
    return np.asarray(Image.open(path))


def _load_image(path: Path) -> np.ndarray:
    arr = _load_gray(path)
    if arr.ndim == 3:  # RGB(A) -> luminance
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.float32)


def _load_mask(path: Path) -> np.ndarray:
    arr = _load_gray(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _components(binary: np.ndarray) -> list[np.ndarray]:
    """Split a boolean mask into 4-connected components (pure numpy/Python)."""
    h, w = binary.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not binary[sr, sc] or seen[sr, sc]:
                continue
            comp = np.zeros((h, w), dtype=bool)
            stack = [(sr, sc)]
            seen[sr, sc] = True
            while stack:
                r, c = stack.pop()
                comp[r, c] = True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _instances(mask: np.ndarray) -> list[np.ndarray]:
    """One boolean mask per lesion. Label map -> per-value; binary -> per-component."""
    if int(mask.max()) <= 0:
        return []
    if int(mask.max()) > 1:  # explicit instance ids
        return [(mask == v) for v in sorted(np.unique(mask)) if v != 0]
    return _components(mask > 0)  # binary -> connected regions


def load_coco_boxes(images_dir: str | Path, coco_json: str | Path,
                    spacing: tuple[float, float] = (1.0, 1.0), modality: str = "OT",
                    category_field: str = "category_id", limit: int | None = None,
                    window: WindowSpec | None = None) -> list[Case]:
    """Load a COCO detection set (boxes only, no masks) into Cases — e.g. DENTEX.

    Each image becomes a Case with `gt_boxes` (from COCO bbox [x,y,w,h], clamped)
    and `gt_masks=[]`. So `localization_metrics` scores fully; `segmentation_metrics`
    sees 0 lesions and simply doesn't gate segmentation (no masks, no crash).
    `category_field` selects the annotation's class field (DENTEX carries several).
    `limit` caps how many images are loaded — keep RAM within budget on big sets.
    """
    import json

    coco = json.loads(Path(coco_json).read_text())
    cat_names = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    anns_by_img: dict = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    cases: list[Case] = []
    for im in coco.get("images", []):
        if limit is not None and len(cases) >= limit:
            break
        arr = _load_image(Path(images_dir) / im["file_name"])
        rows, cols = arr.shape
        boxes, labels = [], []
        for ann in anns_by_img.get(im["id"], []):
            x, y, w, h = ann["bbox"]
            boxes.append(BBox(x=float(x), y=float(y), w=float(w), h=float(h)).clamp(cols, rows))
            labels.append(cat_names.get(ann.get(category_field), "finding"))
        meta = ImageMeta(study_id=str(im["id"]), rows=rows, cols=cols,
                         pixel_spacing_mm=spacing, modality=modality)
        if window is not None:
            meta = resolve_window(meta, window)
        cases.append(Case(image=arr, meta=meta, gt_boxes=boxes, gt_masks=[], labels=labels))
    return cases


def load_dir(path: str | Path, spacing: tuple[float, float] = (1.0, 1.0),
             modality: str = "OT", window: WindowSpec | None = None) -> list[Case]:
    """Load a labeled set for the validation gate (Phase 3b).

    Expects <path>/images/<stem>.{png,npy} paired with <path>/masks/<stem>.{png,npy}
    (0 = background). Each lesion becomes a GT mask + bounding box. `spacing` is the
    (dy, dx) mm pixel spacing applied to every case (PNG/npy carry none of their own).
    `window` (from the active modality profile) overrides the display window.
    """
    root = Path(path)
    img_dir, mask_dir = root / "images", root / "masks"
    cases: list[Case] = []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".png", ".npy", ".jpg", ".jpeg"):
            continue
        mask_path = next((mask_dir / f"{img_path.stem}{ext}"
                          for ext in (".png", ".npy")
                          if (mask_dir / f"{img_path.stem}{ext}").exists()), None)
        if mask_path is None:
            raise FileNotFoundError(f"no mask for image {img_path.stem!r} in {mask_dir}")
        arr = _load_image(img_path)
        insts = _instances(_load_mask(mask_path))
        meta = ImageMeta(study_id=img_path.stem, rows=arr.shape[0], cols=arr.shape[1],
                         pixel_spacing_mm=spacing, modality=modality)
        if window is not None:
            meta = resolve_window(meta, window)
        cases.append(Case(image=arr, meta=meta, gt_boxes=[_mask_bbox(m) for m in insts],
                          gt_masks=list(insts), labels=["lesion"] * len(insts)))
    return cases

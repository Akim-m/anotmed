"""DICOM loading, windowing, de-identification, and rendering/export helpers.

Kept deliberately small: read pixels + spacing, turn them into something a
screen or a VLM can consume, and serialize accepted annotations. Anything that
needs a heavier library (DICOM-SEG via highdicom) degrades gracefully if that
library is absent.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .measure import _boundary_pixels, _convex_hull
from .schema import Annotation, ImageMeta, Study

# DICOM tags that carry patient identity. Stripped by `deidentify`.
_PHI_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "InstitutionName", "InstitutionAddress", "AccessionNumber",
    "StudyID", "DeviceSerialNumber",
]


def read_dataset(path: str | Path):
    """Read a DICOM file into a pydicom Dataset (kept so callers can retain it)."""
    import pydicom

    return pydicom.dcmread(str(path))


def load_dicom(path: str | Path, study_id: str = "") -> tuple[np.ndarray, ImageMeta]:
    """Read a DICOM file into a 2D float array plus metadata.

    Applies the rescale slope/intercept so pixel values are in modality units
    (e.g. Hounsfield for CT). Missing PixelSpacing is flagged, not guessed —
    measurements would otherwise be silently wrong.
    """
    return dataset_to_array_meta(read_dataset(path), study_id)


def dataset_to_array_meta(ds, study_id: str = "") -> tuple[np.ndarray, ImageMeta]:
    """Extract the pixel array (in modality units) and ImageMeta from a Dataset."""
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # RGB(A) — collapse to luminance
        arr = arr[..., :3].mean(axis=-1)
    elif arr.ndim == 3:  # multi-frame — take the middle frame
        arr = arr[arr.shape[0] // 2]

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    if spacing is not None and len(spacing) == 2:
        pixel_spacing = (float(spacing[0]), float(spacing[1]))
        spacing_default = False
    else:
        pixel_spacing = (1.0, 1.0)
        spacing_default = True

    wc, ww = _window_values(ds)
    meta = ImageMeta(
        study_id=study_id,
        rows=int(arr.shape[0]),
        cols=int(arr.shape[1]),
        pixel_spacing_mm=pixel_spacing,
        modality=str(getattr(ds, "Modality", "OT")),
        window_center=wc,
        window_width=ww,
        spacing_is_default=spacing_default,
    )
    return arr, meta


def _first_float(v) -> float | None:
    """DICOM window tags may be a single value or a MultiValue; take the first."""
    if v is None:
        return None
    if not isinstance(v, str) and hasattr(v, "__iter__"):
        seq = list(v)
        return float(seq[0]) if seq else None
    return float(v)


def _window_values(ds) -> tuple[float | None, float | None]:
    return _first_float(getattr(ds, "WindowCenter", None)), \
        _first_float(getattr(ds, "WindowWidth", None))


def deidentify(ds):
    """Blank known PHI tags in place. A pragmatic scrub, not a certified anonymizer."""
    for tag in _PHI_TAGS:
        if tag in ds:
            ds.data_element(tag).value = ""
    return ds


def apply_window(arr: np.ndarray, center: float | None, width: float | None) -> np.ndarray:
    """Window a float image to uint8 for display."""
    a = arr.astype(np.float64)
    if center is None or not width:
        lo, hi = float(a.min()), float(a.max())
    else:
        lo, hi = center - width / 2.0, center + width / 2.0
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_display_array(arr: np.ndarray, meta: ImageMeta | None = None) -> np.ndarray:
    """uint8 HxWx3 RGB — what SAM and MedGemma consume."""
    wc = meta.window_center if meta else None
    ww = meta.window_width if meta else None
    gray = apply_window(arr, wc, ww)
    return np.repeat(gray[:, :, None], 3, axis=2)


def to_display_rgb(arr: np.ndarray, meta: ImageMeta | None = None) -> Image.Image:
    return Image.fromarray(to_display_array(arr, meta), mode="RGB")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_png(arr: np.ndarray, meta: ImageMeta) -> bytes:
    return _png_bytes(to_display_rgb(arr, meta))


def mask_overlay_png(mask: np.ndarray, color=(220, 40, 40), alpha: int = 110) -> bytes:
    """Transparent RGBA PNG: the mask tinted, everything else clear (for UI overlay)."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = color
    rgba[mask, 3] = alpha
    return _png_bytes(Image.fromarray(rgba, mode="RGBA"))


def mask_to_polygon(mask: np.ndarray) -> list[float]:
    """Convex-hull polygon [x1, y1, x2, y2, ...] in pixel coords, for COCO segmentation.

    A hull approximation — faithful masks are exported separately as PNG and,
    when available, DICOM-SEG. Documented so no one mistakes it for the exact
    contour.
    """
    rows, cols = _boundary_pixels(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return []
    hull = _convex_hull(np.column_stack([cols, rows]).astype(float))
    return [float(v) for xy in hull for v in xy]


def annotations_to_coco(study: Study, annotations: list[Annotation],
                        masks: dict[str, np.ndarray]) -> dict:
    """Build a COCO-format dict. Caller is responsible for passing accepted-only."""
    coco = {
        "info": {"description": "anotmed annotations", "study_id": study.id,
                 "note": "AI-suggested, human-verified. Not a diagnosis."},
        "images": [{"id": 1, "file_name": f"{study.id}.png",
                    "width": study.meta.cols, "height": study.meta.rows,
                    "modality": study.meta.modality}],
        "categories": [{"id": 1, "name": "finding"}],
        "annotations": [],
    }
    for i, ann in enumerate(annotations, start=1):
        b = ann.box
        poly = mask_to_polygon(masks[ann.id]) if ann.id in masks else []
        coco["annotations"].append({
            "id": i, "image_id": 1, "category_id": 1, "iscrowd": 0,
            "bbox": [b.x, b.y, b.w, b.h],
            "area": ann.measurement.n_pixels if ann.measurement else b.w * b.h,
            "segmentation": [poly] if poly else [],
            "anotmed": {
                "label": ann.label,
                "status": ann.status.value,
                "edited": ann.edited,
                "reviewer_note": ann.reviewer_note,
                "report_text": ann.report_text,
                "measurement_mm": ann.measurement.model_dump() if ann.measurement else None,
                "provenance": ann.provenance.model_dump(mode="json"),
            },
        })
    return coco


def export_coco(study: Study, annotations: list[Annotation],
                masks: dict[str, np.ndarray], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coco = annotations_to_coco(study, annotations, masks)
    path = out / f"{study.id}_coco.json"
    path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return path


def export_dicom_seg(arr: np.ndarray, meta: ImageMeta, annotations: list[Annotation],
                     masks: dict[str, np.ndarray], out_path: str | Path) -> Path:
    """Export accepted masks as a DICOM-SEG object. Requires `highdicom`.

    Raises a clear error if highdicom is not installed rather than silently
    producing a lesser format.
    """
    try:
        import highdicom  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "DICOM-SEG export needs `highdicom` (pip install anotmed[seg]). "
            "COCO export is available without it."
        ) from e
    raise NotImplementedError(
        "DICOM-SEG export requires the source DICOM dataset (not just pixels). "
        "Wire this to your source series in WSL; highdicom is installed."
    )

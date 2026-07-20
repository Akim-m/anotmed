"""DICOM-SEG export via highdicom — accepted annotations only.

Builds a `highdicom.seg.Segmentation` that references the retained (de-identified)
source series, so the result overlays on the original study in a PACS. Input is
exclusively the accepted annotations the caller passes in — the new export format
enters through the same safety gate as COCO.

Segment property codes are intentionally generic (a "morphologically abnormal
structure") until the modality decision lets us assign specific codes (PLAN.md
§4). The algorithm is recorded as semi-automatic: the model proposes, a human
accepts — no pixel leaves without that acceptance.
"""

from __future__ import annotations

import io

import numpy as np

from .schema import Annotation, BBox, ImageMeta

# Type-2 attributes highdicom copies from the source; a SecondaryCapture may omit
# them, so we ensure they exist (empty) before building the SEG.
_REQUIRED_TYPE2 = ["PatientBirthDate", "PatientSex", "AccessionNumber",
                   "StudyID", "StudyDate", "StudyTime"]
_CATEGORY_CODE = ("49755003", "SCT", "Morphologically Abnormal Structure")


def _ensure_type2(ds) -> None:
    for tag in _REQUIRED_TYPE2:
        if tag not in ds:
            setattr(ds, tag, "")


def _box_mask(box: BBox, rows: int, cols: int) -> np.ndarray:
    m = np.zeros((rows, cols), dtype=bool)
    rs, cs = box.as_slice(cols, rows)
    m[rs, cs] = True
    return m


def build_segmentation(source_ds, meta: ImageMeta, annotations: list[Annotation],
                       masks: dict[str, np.ndarray], backend_name: str = "anotmed"):
    import highdicom as hd  # noqa: F401
    from highdicom.content import AlgorithmIdentificationSequence
    from highdicom.seg import (
        Segmentation,
        SegmentAlgorithmTypeValues,
        SegmentationTypeValues,
        SegmentDescription,
    )
    from highdicom.sr.coding import CodedConcept
    from pydicom.sr.codedict import codes
    from pydicom.uid import generate_uid

    if not annotations:
        raise ValueError("no accepted annotations to export")

    _ensure_type2(source_ds)
    rows, cols = meta.rows, meta.cols
    category = CodedConcept(*_CATEGORY_CODE)
    algo = AlgorithmIdentificationSequence(
        name="anotmed", family=codes.DCM.ArtificialIntelligence, version="0.1.0")

    descriptions = []
    layers = []
    for i, ann in enumerate(annotations, start=1):
        mask = masks.get(ann.id)
        if mask is None:  # accepted with only a coarse box (no pixel mask) — rasterize it
            mask = _box_mask(ann.box, rows, cols)
        layers.append(np.asarray(mask, dtype=bool).astype(np.uint8))
        descriptions.append(SegmentDescription(
            segment_number=i,
            segment_label=ann.label or f"finding {i}",
            segmented_property_category=category,
            segmented_property_type=category,
            algorithm_type=SegmentAlgorithmTypeValues.SEMIAUTOMATIC,
            algorithm_identification=algo,
        ))

    pixel_array = np.stack(layers, axis=-1)[np.newaxis]  # (1, rows, cols, n_segments)

    return Segmentation(
        source_images=[source_ds],
        pixel_array=pixel_array,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=descriptions,
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="anotmed",
        manufacturer_model_name=backend_name,
        software_versions="0.1.0",
        device_serial_number="anotmed",
    )


def export_dicom_seg_bytes(source_ds, meta: ImageMeta, annotations: list[Annotation],
                           masks: dict[str, np.ndarray], backend_name: str = "anotmed") -> bytes:
    seg = build_segmentation(source_ds, meta, annotations, masks, backend_name)
    buf = io.BytesIO()
    seg.save_as(buf)
    return buf.getvalue()

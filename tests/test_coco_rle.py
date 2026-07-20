"""COCO segmentation should be the exact mask (RLE), not a convex-hull approximation.

The faithful mask is already on disk, so the COCO export can carry it losslessly
as uncompressed RLE instead of the lossy hull polygon it used before.
"""

from __future__ import annotations

import numpy as np

from anotmed.io_dicom import annotations_to_coco, mask_to_rle
from anotmed.schema import Annotation, BBox, ImageMeta, Provenance, Study


def _rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for count in rle["counts"]:
        flat[pos:pos + count] = val
        pos += count
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


def test_mask_to_rle_roundtrips_exactly():
    m = np.zeros((6, 5), dtype=bool)
    m[1:4, 2:4] = True
    m[0, 0] = True  # foreground at the very first pixel -> needs a leading 0-run
    rle = mask_to_rle(m)
    assert rle["size"] == [6, 5]
    assert np.array_equal(_rle_decode(rle), m)  # exact, unlike a hull


def test_empty_mask_rle_is_all_background():
    m = np.zeros((4, 4), dtype=bool)
    assert np.array_equal(_rle_decode(mask_to_rle(m)), m)


def test_coco_emits_rle_when_a_mask_is_present():
    meta = ImageMeta(study_id="s", rows=6, cols=5)
    study = Study(id="s", source_name="x.dcm", meta=meta)
    ann = Annotation(id="a1", study_id="s", label="lesion",
                     box=BBox(x=2, y=1, w=2, h=3), provenance=Provenance(backend="stub"))
    mask = np.zeros((6, 5), dtype=bool)
    mask[1:4, 2:4] = True

    coco = annotations_to_coco(study, [ann], {"a1": mask})
    seg = coco["annotations"][0]["segmentation"]
    assert isinstance(seg, dict) and seg["size"] == [6, 5]
    assert np.array_equal(_rle_decode(seg), mask)

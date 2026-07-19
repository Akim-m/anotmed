"""The load-bearing safety property: only accepted annotations can be exported.

If this ever fails, the "radiologists only verify" guarantee is broken.
"""

import numpy as np

from anotmed.io_dicom import annotations_to_coco
from anotmed.schema import Annotation, BBox, ImageMeta, Provenance, ReviewStatus
from anotmed.store import Store


def _ann(sid, aid, status):
    return Annotation(
        id=aid, study_id=sid, label="finding",
        box=BBox(x=0, y=0, w=4, h=4),
        provenance=Provenance(backend="stub"), status=status,
    )


def test_only_accepted_annotations_are_exportable(tmp_path):
    store = Store(tmp_path)
    study = store.create_study(np.zeros((16, 16), np.float32),
                               ImageMeta(study_id="x", rows=16, cols=16), "s.dcm")
    sid = study.id
    for aid, st in [("a", ReviewStatus.PENDING),
                    ("b", ReviewStatus.ACCEPTED),
                    ("c", ReviewStatus.REJECTED)]:
        store.upsert_annotation(_ann(sid, aid, st))

    accepted = store.accepted_annotations(sid)
    assert [a.id for a in accepted] == ["b"]

    coco = annotations_to_coco(study, accepted, {})
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["anotmed"]["status"] == "accepted"

"""Orchestrates one image through localize -> segment -> measure -> report.

Two rules encoded here:
  * Measurements come from the geometry engine, never from the language model.
    The reporter writes prose; the numbers are computed from the mask.
  * Per-finding failures are counted and logged, not swallowed. A segmenter that
    throws on one box does not lose the others.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

from .backends import Backend, Detection
from .measure import measure_box, measure_mask
from .schema import Annotation, Provenance, ReviewStatus, Study
from .store import Store

log = logging.getLogger("anotmed.pipeline")


class Pipeline:
    def __init__(self, backend: Backend, store: Store, min_score: float = 0.0):
        self.backend = backend
        self.store = store
        self.min_score = min_score
        self.segment_failures = 0  # meter: surfaced in run() summary

    def _provenance(self, det: Detection) -> Provenance:
        return Provenance(
            backend=self.backend.name,
            localizer=self.backend.localizer.name,
            segmenter=self.backend.segmenter.name,
            reporter=self.backend.reporter.name,
            score=det.score,
        )

    def _annotate(self, study: Study, arr: np.ndarray, det: Detection,
                  ann_id: str | None = None, edited: bool = False) -> Annotation:
        meta = study.meta
        spacing = meta.pixel_spacing_mm
        ann_id = ann_id or uuid.uuid4().hex[:12]

        mask: np.ndarray | None = None
        try:
            mask = self.backend.segmenter.segment(arr, det.box, meta)
        except Exception:  # one finding failing must not lose the rest
            self.segment_failures += 1
            log.exception("segmenter failed for study %s box %s", study.id, det.box)

        if mask is not None and mask.any():
            measurement = measure_mask(mask, spacing)
            mask_path = self.store.save_mask(study.id, ann_id, mask)
        else:
            measurement = measure_box(det.box, spacing)  # coarse box fallback
            mask_path = None

        report_mask = mask if mask is not None else np.zeros((meta.rows, meta.cols), bool)
        prose = self.backend.reporter.describe(arr, det, report_mask, meta)
        summary = (f"Measured: max diameter {measurement.max_diameter_mm:.1f} mm, "
                   f"area {measurement.area_cm2:.2f} cm².")

        return Annotation(
            id=ann_id,
            study_id=study.id,
            slice_index=meta.slice_index,
            label=det.label,
            box=det.box,
            mask_path=mask_path,
            measurement=measurement,
            report_text=f"{prose}\n{summary}",
            status=ReviewStatus.PENDING,
            edited=edited,
            provenance=self._provenance(det),
        )

    def run(self, study: Study) -> list[Annotation]:
        """Localize, then annotate every proposed finding. Persists as it goes."""
        self.segment_failures = 0
        arr = self.store.load_image(study.id)
        dets = self.backend.localizer.propose(arr, study.meta)
        anns: list[Annotation] = []
        for det in dets:
            if det.score < self.min_score:
                continue
            ann = self._annotate(study, arr, det)
            self.store.upsert_annotation(ann)
            anns.append(ann)
        log.info("study %s: %d findings, %d segment failures",
                 study.id, len(anns), self.segment_failures)
        return anns

    def reannotate(self, study: Study, ann: Annotation, new_box, new_label=None) -> Annotation:
        """Re-run segment/measure/report after a reviewer edits the box.

        Returns a PENDING annotation carrying the original id and marked edited —
        the reviewer still has to accept the corrected version.
        """
        arr = self.store.load_image(study.id)
        det = Detection(box=new_box, label=new_label or ann.label,
                        score=ann.provenance.score or 0.0)
        updated = self._annotate(study, arr, det, ann_id=ann.id, edited=True)
        updated.reviewer_note = ann.reviewer_note
        self.store.upsert_annotation(updated)
        return updated

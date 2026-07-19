"""Filesystem-backed store. No database, so the system runs the moment it starts.

Layout under the storage dir:
    <study_id>/study.json          study metadata
    <study_id>/image.npy           float32 pixel array (modality units)
    <study_id>/annotations.json    list of Annotation
    <study_id>/masks/<ann_id>.npy  boolean mask per annotation

The safety invariant lives here: `accepted_annotations` is the ONLY way to get
data out for export, and it returns accepted annotations exclusively. Nothing a
radiologist has not accepted can be exported.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from .schema import Annotation, ImageMeta, ReviewStatus, Study


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, study_id: str) -> Path:
        return self.root / study_id

    # --- studies -----------------------------------------------------------
    def create_study(self, arr: np.ndarray, meta: ImageMeta, source_name: str,
                     source_path: str | None = None, deidentified: bool = False) -> Study:
        study_id = uuid.uuid4().hex[:12]
        meta = meta.model_copy(update={"study_id": study_id})
        study = Study(id=study_id, source_name=source_name, source_path=source_path,
                      meta=meta, deidentified=deidentified)
        d = self._dir(study_id)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        np.save(d / "image.npy", arr.astype(np.float32))
        (d / "study.json").write_text(study.model_dump_json(indent=2), encoding="utf-8")
        (d / "annotations.json").write_text("[]", encoding="utf-8")
        return study

    def get_study(self, study_id: str) -> Study:
        path = self._dir(study_id) / "study.json"
        if not path.exists():
            raise KeyError(study_id)
        return Study.model_validate_json(path.read_text(encoding="utf-8"))

    def list_studies(self) -> list[Study]:
        out = []
        for d in sorted(self.root.iterdir()):
            f = d / "study.json"
            if f.exists():
                out.append(Study.model_validate_json(f.read_text(encoding="utf-8")))
        return out

    def load_image(self, study_id: str) -> np.ndarray:
        return np.load(self._dir(study_id) / "image.npy")

    # --- annotations -------------------------------------------------------
    def get_annotations(self, study_id: str) -> list[Annotation]:
        path = self._dir(study_id) / "annotations.json"
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Annotation.model_validate(a) for a in raw]

    def save_annotations(self, study_id: str, annotations: list[Annotation]) -> None:
        path = self._dir(study_id) / "annotations.json"
        payload = [a.model_dump(mode="json") for a in annotations]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_annotation(self, study_id: str, ann_id: str) -> Annotation:
        for a in self.get_annotations(study_id):
            if a.id == ann_id:
                return a
        raise KeyError(ann_id)

    def upsert_annotation(self, ann: Annotation) -> None:
        anns = self.get_annotations(ann.study_id)
        replaced = False
        for i, a in enumerate(anns):
            if a.id == ann.id:
                anns[i] = ann
                replaced = True
                break
        if not replaced:
            anns.append(ann)
        self.save_annotations(ann.study_id, anns)

    # --- masks -------------------------------------------------------------
    def save_mask(self, study_id: str, ann_id: str, mask: np.ndarray) -> str:
        path = self._dir(study_id) / "masks" / f"{ann_id}.npy"
        np.save(path, np.asarray(mask, dtype=bool))
        return str(path)

    def load_mask(self, study_id: str, ann_id: str) -> np.ndarray | None:
        path = self._dir(study_id) / "masks" / f"{ann_id}.npy"
        return np.load(path) if path.exists() else None

    def load_masks(self, study_id: str, annotations: list[Annotation]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for a in annotations:
            m = self.load_mask(study_id, a.id)
            if m is not None:
                out[a.id] = m
        return out

    # --- the safety gate ---------------------------------------------------
    def accepted_annotations(self, study_id: str) -> list[Annotation]:
        """The only export path. Returns accepted annotations exclusively."""
        return [a for a in self.get_annotations(study_id)
                if a.status == ReviewStatus.ACCEPTED]

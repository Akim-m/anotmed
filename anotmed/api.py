"""HTTP API + review UI.

Endpoints cover the full loop: upload a DICOM (which runs the pipeline), inspect
the pending suggestions, accept/reject/edit each one, and export the accepted
set. Export goes exclusively through `store.accepted_annotations`, so the
human-in-the-loop gate cannot be bypassed via the API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .backends import build_backend
from .config import load_config
from .io_dicom import annotations_to_coco, image_png, load_dicom, mask_overlay_png
from .pipeline import Pipeline
from .schema import BBox, ReviewStatus
from .store import Store

app = FastAPI(title="anotmed", version="0.1.0")

_cfg = load_config()
_store = Store(_cfg.storage_dir)
_backend = None  # built lazily so importing this module never loads model weights


def _get_pipeline() -> Pipeline:
    global _backend
    if _backend is None:
        _backend = build_backend(_cfg)
    return Pipeline(_backend, _store, _cfg.min_score)


@app.get("/api/healthz")
def healthz():
    return {"status": "ok", "backend": _cfg.backend}


@app.post("/api/studies")
async def create_study(file: UploadFile = File(...)):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        arr, meta = load_dicom(tmp_path)
    except Exception as e:
        raise HTTPException(400, f"Could not read DICOM: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Only the pixel array and non-identifying ImageMeta are persisted; PHI tags
    # are never written to disk.
    study = _store.create_study(arr, meta, source_name=file.filename or "upload.dcm")
    anns = _get_pipeline().run(study)
    return {"study": study.model_dump(mode="json"),
            "annotations": [a.model_dump(mode="json") for a in anns]}


@app.get("/api/studies")
def list_studies():
    return [s.model_dump(mode="json") for s in _store.list_studies()]


@app.get("/api/studies/{sid}")
def get_study(sid: str):
    try:
        return _store.get_study(sid).model_dump(mode="json")
    except KeyError:
        raise HTTPException(404, "study not found")


@app.get("/api/studies/{sid}/image.png")
def study_image(sid: str):
    try:
        study = _store.get_study(sid)
    except KeyError:
        raise HTTPException(404, "study not found")
    arr = _store.load_image(sid)
    return Response(image_png(arr, study.meta), media_type="image/png")


@app.get("/api/studies/{sid}/annotations")
def study_annotations(sid: str):
    return [a.model_dump(mode="json") for a in _store.get_annotations(sid)]


@app.get("/api/studies/{sid}/annotations/{aid}/mask.png")
def annotation_mask(sid: str, aid: str):
    mask = _store.load_mask(sid, aid)
    if mask is None:
        raise HTTPException(404, "no mask for this annotation")
    return Response(mask_overlay_png(mask), media_type="image/png")


@app.post("/api/studies/{sid}/annotations/{aid}/review")
def review(sid: str, aid: str, payload: dict = Body(...)):
    """action: accept | reject | edit. `box` (x,y,w,h) triggers re-segmentation."""
    try:
        study = _store.get_study(sid)
        ann = _store.get_annotation(sid, aid)
    except KeyError:
        raise HTTPException(404, "not found")

    action = payload.get("action", "")
    if "label" in payload:
        ann.label = str(payload["label"])
    if "note" in payload:
        ann.reviewer_note = str(payload["note"])

    if "box" in payload and payload["box"]:
        b = payload["box"]
        new_box = BBox(x=float(b["x"]), y=float(b["y"]), w=float(b["w"]), h=float(b["h"]))
        ann = _get_pipeline().reannotate(study, ann, new_box, ann.label)

    if action == "accept":
        ann.status = ReviewStatus.ACCEPTED
    elif action == "reject":
        ann.status = ReviewStatus.REJECTED
    elif action == "edit":
        ann.status = ReviewStatus.PENDING
    elif action:
        raise HTTPException(400, f"unknown action {action!r}")

    _store.upsert_annotation(ann)
    return ann.model_dump(mode="json")


@app.post("/api/studies/{sid}/export")
def export(sid: str, format: str = "coco"):
    try:
        study = _store.get_study(sid)
    except KeyError:
        raise HTTPException(404, "study not found")

    accepted = _store.accepted_annotations(sid)  # the safety gate
    if not accepted:
        raise HTTPException(409, "nothing to export: no annotations accepted yet")

    if format == "coco":
        masks = _store.load_masks(sid, accepted)
        return JSONResponse(annotations_to_coco(study, accepted, masks))
    if format == "dicom-seg":
        raise HTTPException(501, "DICOM-SEG export not wired in this build; use COCO")
    raise HTTPException(400, f"unknown format {format!r}")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "web" / "review.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

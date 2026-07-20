"""HTTP API + review UI.

Endpoints cover the full loop: upload a DICOM (which runs the pipeline), inspect
the pending suggestions, accept/reject/edit each one, and export the accepted
set. Export goes exclusively through `store.accepted_annotations`, so the
human-in-the-loop gate cannot be bypassed via the API.
"""

from __future__ import annotations

import logging
import tempfile
from http import HTTPStatus
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .backends import build_backend
from .config import load_config
from .io_dicom import (
    annotations_to_coco,
    dataset_to_array_meta,
    deidentify,
    image_png,
    mask_overlay_png,
    read_dataset,
)
from .jobs import JobRegistry
from .pipeline import Pipeline
from .schema import BBox, ReviewStatus
from .store import Store

log = logging.getLogger("anotmed.api")

app = FastAPI(title="anotmed", version="0.1.0")


def _error_slug(code: int) -> str:
    try:
        return HTTPStatus(code).phrase.lower().replace(" ", "_")
    except ValueError:
        return "error"


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Uniform {error, message} envelope instead of FastAPI's default {detail}."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": _error_slug(exc.status_code), "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": str(exc)},
    )

_cfg = load_config()
_store = Store(_cfg.storage_dir)
_backend = None  # built lazily so importing this module never loads model weights
_jobs = JobRegistry()


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
        ds = read_dataset(tmp_path)
        arr, meta = dataset_to_array_meta(ds)
    except Exception as e:
        raise HTTPException(400, f"Could not read DICOM: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # The source series is retained (for DICOM-SEG referencing) only after PHI is
    # stripped in place; identifying tags are never written to disk.
    deidentify(ds)
    study = _store.create_study(arr, meta, source_name=file.filename or "upload.dcm",
                                source_ds=ds)

    if not _cfg.sync:
        # Real inference is slow: persist now, run on the worker, let the client
        # poll GET /api/jobs/{id}. Annotations still land as PENDING via the same
        # Store writes, so the export gate is unaffected.
        job = _jobs.submit(study.id, lambda s=study: _get_pipeline().run(s))
        return JSONResponse(
            status_code=202,
            content={"job_id": job.id, "study_id": study.id, "status": job.status},
        )

    anns = _get_pipeline().run(study)
    return {"study": study.model_dump(mode="json"),
            "annotations": [a.model_dump(mode="json") for a in anns]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    body = {"job_id": job.id, "study_id": job.study_id, "status": job.status}
    if job.status == "failed":
        body["error"] = job.error
    return body


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

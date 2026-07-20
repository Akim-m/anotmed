"""Async submit + poll — the in-process job model, and the export gate through it.

Two levels: the JobRegistry worker in isolation (completes / fails correctly),
and the full async HTTP loop proving the safety invariant still holds when the
pipeline runs on a background thread instead of inline.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from examples.make_sample_dicom import write_dicom


# ---- JobRegistry worker -----------------------------------------------------

def _wait_terminal(reg, job_id, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        job = reg.get(job_id)
        if job and job.status in ("completed", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never terminated (status={reg.get(job_id)})")


def test_registry_runs_job_to_completion():
    from anotmed.jobs import JobRegistry

    reg = JobRegistry()
    ran = []
    job = reg.submit("study-1", lambda: ran.append(True))
    assert job.status in ("pending", "processing", "completed")
    done = _wait_terminal(reg, job.id)
    assert done.status == "completed"
    assert ran == [True]


def test_registry_marks_job_failed_on_real_error():
    from anotmed.jobs import JobRegistry

    reg = JobRegistry()

    def boom():
        raise ValueError("segmenter exploded")

    job = reg.submit("study-1", boom)
    done = _wait_terminal(reg, job.id)
    assert done.status == "failed"
    assert "exploded" in done.error


def test_unknown_job_id_is_none():
    from anotmed.jobs import JobRegistry

    assert JobRegistry().get("nope") is None


# ---- async HTTP loop --------------------------------------------------------

@pytest.fixture
def async_client(tmp_path, monkeypatch):
    from anotmed import api
    from anotmed.config import Config
    from anotmed.store import Store

    monkeypatch.setattr(api, "_cfg", Config(backend="stub", storage_dir=tmp_path, sync=False))
    monkeypatch.setattr(api, "_store", Store(tmp_path))
    monkeypatch.setattr(api, "_backend", None)
    return TestClient(api.app)


def _submit_and_wait(client, tmp_path):
    p = tmp_path / "s.dcm"
    write_dicom(str(p), 256)
    with open(p, "rb") as f:
        r = client.post("/api/studies", files={"file": ("s.dcm", f, "application/dicom")})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    for _ in range(500):
        jr = client.get(f"/api/jobs/{job_id}").json()
        if jr["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert jr["status"] == "completed", jr
    return jr["study_id"]


def test_async_submit_returns_202_with_job_id(async_client, tmp_path):
    p = tmp_path / "s.dcm"
    write_dicom(str(p), 256)
    with open(p, "rb") as f:
        r = async_client.post("/api/studies", files={"file": ("s.dcm", f, "application/dicom")})
    assert r.status_code == 202
    assert r.json()["job_id"]
    assert r.json()["status"] in ("pending", "processing", "completed")


def test_async_path_preserves_the_export_gate(async_client, tmp_path):
    sid = _submit_and_wait(async_client, tmp_path)

    anns = async_client.get(f"/api/studies/{sid}/annotations").json()
    assert anns and all(a["status"] == "pending" for a in anns)

    # gate blocks until a human accepts
    assert async_client.post(f"/api/studies/{sid}/export?format=coco").status_code == 409
    async_client.post(f"/api/studies/{sid}/annotations/{anns[0]['id']}/review",
                      json={"action": "accept"})
    ex = async_client.post(f"/api/studies/{sid}/export?format=coco")
    assert ex.status_code == 200
    assert len(ex.json()["annotations"]) == 1  # only the accepted one


def test_get_unknown_job_returns_404(async_client):
    r = async_client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"

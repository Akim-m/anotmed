"""The full HTTP loop: upload -> review -> export, and the export gate."""

import pytest
from fastapi.testclient import TestClient

from examples.make_sample_dicom import write_dicom


@pytest.fixture
def client(tmp_path, monkeypatch):
    from anotmed import api
    from anotmed.config import Config
    from anotmed.store import Store

    monkeypatch.setattr(api, "_cfg", Config(backend="stub", storage_dir=tmp_path))
    monkeypatch.setattr(api, "_store", Store(tmp_path))
    monkeypatch.setattr(api, "_backend", None)
    return TestClient(api.app)


def _upload(client, tmp_path):
    p = tmp_path / "s.dcm"
    write_dicom(str(p), 256)
    with open(p, "rb") as f:
        r = client.post("/api/studies", files={"file": ("s.dcm", f, "application/dicom")})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_runs_pipeline(client, tmp_path):
    data = _upload(client, tmp_path)
    assert len(data["annotations"]) >= 1
    assert all(a["status"] == "pending" for a in data["annotations"])


def test_export_requires_acceptance(client, tmp_path):
    data = _upload(client, tmp_path)
    sid = data["study"]["id"]

    # Nothing accepted yet — the gate blocks export.
    assert client.post(f"/api/studies/{sid}/export?format=coco").status_code == 409

    aid = data["annotations"][0]["id"]
    r = client.post(f"/api/studies/{sid}/annotations/{aid}/review", json={"action": "accept"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"

    ex = client.post(f"/api/studies/{sid}/export?format=coco")
    assert ex.status_code == 200
    coco = ex.json()
    assert len(coco["annotations"]) == 1  # only the accepted one


def test_reject_then_export_is_still_blocked(client, tmp_path):
    data = _upload(client, tmp_path)
    sid = data["study"]["id"]
    aid = data["annotations"][0]["id"]
    client.post(f"/api/studies/{sid}/annotations/{aid}/review", json={"action": "reject"})
    assert client.post(f"/api/studies/{sid}/export?format=coco").status_code == 409

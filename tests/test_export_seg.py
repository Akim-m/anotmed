"""Phase 5: persist a de-identified source series, and export DICOM-SEG.

Retaining the source (PHI stripped) is what lets the SEG reference the original
series so it overlays correctly in a PACS. The safety gate is unchanged: only
accepted annotations become segments.
"""

from __future__ import annotations

import io

import numpy as np
import pydicom
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


def test_upload_persists_a_deidentified_source(client, tmp_path):
    study = _upload(client, tmp_path)["study"]
    assert study["source_path"], "source series should be retained for DICOM-SEG"
    assert study["deidentified"] is True

    ds = pydicom.dcmread(study["source_path"])
    assert str(ds.PatientName) == ""   # PHI blanked
    assert str(ds.PatientID) == ""
    assert ds.SeriesInstanceUID        # referencing UID preserved for overlay


def test_dicom_seg_export_is_accepted_only_and_roundtrips(client, tmp_path):
    from anotmed import api

    data = _upload(client, tmp_path)
    sid = data["study"]["id"]
    anns = data["annotations"]
    assert len(anns) >= 2, "need >=2 findings to prove accepted-only"

    # gate blocks while pending
    assert client.post(f"/api/studies/{sid}/export?format=dicom-seg").status_code == 409

    aid = anns[0]["id"]
    client.post(f"/api/studies/{sid}/annotations/{aid}/review", json={"action": "accept"})

    r = client.post(f"/api/studies/{sid}/export?format=dicom-seg")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/dicom")

    seg = pydicom.dcmread(io.BytesIO(r.content))
    assert seg.Modality == "SEG"
    assert len(seg.SegmentSequence) == 1  # exactly the one accepted finding

    # the segment pixels equal the accepted annotation's mask
    mask = api._store.load_mask(sid, aid)
    if mask is not None:
        assert np.array_equal(np.squeeze(seg.pixel_array).astype(bool), mask)

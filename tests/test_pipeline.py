"""End-to-end pipeline on the CPU stub backend: localize -> segment -> measure."""

import numpy as np

from examples.make_sample_dicom import synth_image

from anotmed.backends import build_backend
from anotmed.config import Config
from anotmed.pipeline import Pipeline
from anotmed.schema import ImageMeta, ReviewStatus
from anotmed.store import Store


def _study(tmp_path):
    store = Store(tmp_path)
    arr = synth_image(256).astype(np.float32)
    meta = ImageMeta(study_id="x", rows=256, cols=256, pixel_spacing_mm=(0.7, 0.7),
                     modality="CT", window_center=400, window_width=1200)
    return store, store.create_study(arr, meta, "sample.dcm")


def test_stub_pipeline_finds_and_measures(tmp_path):
    store, study = _study(tmp_path)
    backend = build_backend(Config(backend="stub", storage_dir=tmp_path))
    anns = Pipeline(backend, store).run(study)

    assert len(anns) >= 1
    for a in anns:
        assert a.status == ReviewStatus.PENDING          # nothing auto-accepted
        assert a.measurement is not None
        assert a.measurement.area_mm2 > 0                # real geometry, not a box
        assert a.provenance.localizer.startswith("stub")
        assert "verify" in a.report_text.lower()          # draft framing preserved

    assert len(store.get_annotations(study.id)) == len(anns)  # persisted

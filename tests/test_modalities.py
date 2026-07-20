"""Modality profiles — a registry so each modality declares its own detector,
windowing, labels, and floors, and adding one is a data change not a code change."""

from __future__ import annotations

import pytest

from anotmed.modalities import (
    PROFILES,
    ModalityProfile,
    WindowSpec,
    active_profile,
    get_profile,
    resolve_window,
)
from anotmed.schema import ImageMeta


def test_generic_profile_is_todays_defaults():
    p = get_profile("")
    assert p.name == "generic"
    assert p.window.mode == "dicom"      # keep the file's window (no change)
    assert p.detector_weights == ""


def test_dental_profile_carries_its_facts_and_license_flag():
    p = get_profile("dental")
    assert p.name == "dental"
    assert p.window.mode == "minmax"     # panoramic PNGs: min-max display
    assert p.dicom_modality == "PX"
    assert p.floors_key == "dental"
    assert "research" in p.notes.lower() or "NC" in p.notes  # NC/AGPL flag present


def test_unknown_profile_raises_listing_valid_names():
    with pytest.raises(ValueError) as exc:
        get_profile("does-not-exist")
    assert "dental" in str(exc.value)


def test_active_profile_reads_env(monkeypatch):
    monkeypatch.setenv("ANOTMED_MODALITY", "dental")
    assert active_profile().name == "dental"
    monkeypatch.setenv("ANOTMED_MODALITY", "")
    assert active_profile().name == "generic"


def test_profiles_are_frozen():
    with pytest.raises(Exception):
        get_profile("dental").detector_conf = 0.9  # type: ignore[misc]


def test_resolve_window_fixed_overrides_center_width():
    meta = ImageMeta(study_id="s", rows=10, cols=10)
    out = resolve_window(meta, WindowSpec(mode="fixed", center=-600, width=1500))
    assert out.window_center == -600 and out.window_width == 1500


def test_resolve_window_minmax_clears_the_window():
    meta = ImageMeta(study_id="s", rows=10, cols=10, window_center=40, window_width=400)
    out = resolve_window(meta, WindowSpec(mode="minmax"))
    assert out.window_center is None and out.window_width is None


def test_resolve_window_dicom_leaves_meta_untouched():
    meta = ImageMeta(study_id="s", rows=10, cols=10, window_center=40, window_width=400)
    out = resolve_window(meta, WindowSpec(mode="dicom"))
    assert out.window_center == 40 and out.window_width == 400


# ---- config precedence: explicit env > profile > hardcoded default ----------

def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_modality_flows_profile_defaults_into_config(monkeypatch):
    _clear(monkeypatch, "ANOTMED_DETECTOR_CONF", "ANOTMED_DETECTOR_IMGSZ")
    monkeypatch.setenv("ANOTMED_MODALITY", "dental")
    from anotmed.config import Config

    cfg = Config()
    assert cfg.modality == "dental"
    assert cfg.detector_imgsz == 1024   # from the dental profile


def test_explicit_env_overrides_the_profile(monkeypatch):
    monkeypatch.setenv("ANOTMED_MODALITY", "dental")
    monkeypatch.setenv("ANOTMED_DETECTOR_CONF", "0.5")
    from anotmed.config import Config

    assert Config().detector_conf == 0.5  # env beats the profile's 0.25


def test_no_modality_is_todays_behavior(monkeypatch):
    _clear(monkeypatch, "ANOTMED_MODALITY", "ANOTMED_DETECTOR_CONF", "ANOTMED_DETECTOR_IMGSZ")
    from anotmed.config import Config

    cfg = Config()
    assert cfg.detector_conf == 0.25 and cfg.detector_imgsz == 1024

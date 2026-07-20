"""Modality profiles — one reviewable declaration per imaging modality.

Validation this session proved the modality drives everything (the right detector,
the right display window, the right labels, the right safety floors). Rather than
scatter those across env vars and code, each modality is one frozen dataclass in
`PROFILES`; `ANOTMED_MODALITY` selects the active one. Adding a modality is a data
change — a single `ModalityProfile(...)` literal — not a code change.

Explicit env knobs still win over the profile (see config.py precedence), so the
profile is a set of good defaults, never a hard override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from .schema import ImageMeta


@dataclass(frozen=True)
class WindowSpec:
    """How to turn stored pixels into a display image.

    "dicom"  — keep the file's WindowCenter/Width (min-max fallback): today's behavior.
    "minmax" — force min-max (clear WC/WW): PNGs / non-DICOM sets.
    "fixed"  — force a center/width, overriding the file (e.g. a lung window on CT).
    """

    mode: str = "dicom"
    center: float | None = None
    width: float | None = None


@dataclass(frozen=True)
class ModalityProfile:
    name: str
    dicom_modality: str = "OT"                        # stamped into ImageMeta.modality
    detector_weights: str = ""                        # "" = env must provide
    detector_conf: float = 0.25
    detector_imgsz: int = 1024
    max_findings: int = 8                             # per-modality: ~8 lesions vs ~32 teeth
    window: WindowSpec = field(default_factory=WindowSpec)
    label_map: Mapping[str, str] = field(default_factory=dict)  # detector class NAME -> finding name
    sam_checkpoint: str = ""                          # "" = use the global ANOTMED_SAM_*
    sam_config: str = ""
    floors_key: str = ""                              # key into floors.yaml `modalities:`; "" = global
    notes: str = ""                                   # provenance/license flags


PROFILES: dict[str, ModalityProfile] = {
    # generic == today's behavior, byte-for-byte (no window change, no label remap).
    "generic": ModalityProfile(name="generic"),
    "dental": ModalityProfile(
        name="dental",
        dicom_modality="PX",                          # panoramic X-ray
        detector_conf=0.25,
        detector_imgsz=1024,
        max_findings=40,                              # a panoramic has up to ~32 teeth
        window=WindowSpec(mode="minmax"),
        floors_key="dental",
        notes=("DENTEX is CC-BY-NC-SA-4.0 (non-commercial) and ultralytics is AGPL-3.0 — "
               "the trained detector weights are RESEARCH-ONLY. Cite Hamamci et al., "
               "arXiv:2305.19112 (DENTEX, MICCAI 2023)."),
    ),
}


def get_profile(name: str) -> ModalityProfile:
    if not name:
        return PROFILES["generic"]
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown modality {name!r}; valid: {sorted(PROFILES)}"
        ) from None


def active_profile() -> ModalityProfile:
    """The profile named by ANOTMED_MODALITY (read at call time), or generic."""
    return get_profile(os.getenv("ANOTMED_MODALITY", ""))


def resolve_window(meta: ImageMeta, spec: WindowSpec) -> ImageMeta:
    """Apply a WindowSpec to ImageMeta once at ingest; everything downstream is
    already meta-driven, so nothing else needs to change."""
    if spec.mode == "fixed":
        return meta.model_copy(update={"window_center": spec.center, "window_width": spec.width})
    if spec.mode == "minmax":
        return meta.model_copy(update={"window_center": None, "window_width": None})
    return meta  # "dicom": unchanged

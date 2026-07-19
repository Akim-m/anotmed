"""Backend protocols.

Three small roles, so each model does only what it is good at:

  Localizer  — proposes candidate findings as boxes (MedGemma, or a stub).
  Segmenter  — turns one box into a pixel mask (MedSAM-2, or a stub).
  Reporter   — writes the draft description for a finding (MedGemma, or a stub).

A backend is just a (Localizer, Segmenter, Reporter) triple. The pipeline never
imports a concrete model — it depends only on these Protocols, so swapping the
stub for real weights is a one-line config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..schema import BBox, ImageMeta


@dataclass
class Detection:
    """A proposed finding, before segmentation."""

    box: BBox
    label: str
    score: float = 0.0


class Localizer(Protocol):
    name: str

    def propose(self, image: np.ndarray, meta: ImageMeta) -> list[Detection]:
        """Return candidate findings for a single 2D image."""


class Segmenter(Protocol):
    name: str

    def segment(self, image: np.ndarray, box: BBox, meta: ImageMeta) -> np.ndarray:
        """Return a boolean mask (same HxW as image) for the finding in `box`."""


class Reporter(Protocol):
    name: str

    def describe(
        self, image: np.ndarray, det: Detection, mask: np.ndarray, meta: ImageMeta
    ) -> str:
        """Return a short draft description of the finding."""


@dataclass
class Backend:
    localizer: Localizer
    segmenter: Segmenter
    reporter: Reporter
    name: str

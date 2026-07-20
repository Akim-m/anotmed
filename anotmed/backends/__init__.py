"""Backend selection.

`build_backend` returns a (Localizer, Segmenter, Reporter) triple chosen by
config. Heavy modules (torch, transformers, sam2) are imported only when the
real backend is requested, so the stub path stays dependency-light.
"""

from __future__ import annotations

from ..config import BACKEND_DETECTOR, BACKEND_MODELS, BACKEND_STUB, BACKEND_VLLM, Config
from .base import Backend, Detection, Localizer, Reporter, Segmenter

__all__ = ["Backend", "Detection", "Localizer", "Reporter", "Segmenter", "build_backend"]


def build_backend(cfg: Config) -> Backend:
    if cfg.backend == BACKEND_STUB:
        from .stub import StubLocalizer, StubReporter, StubSegmenter

        return Backend(
            localizer=StubLocalizer(cfg.max_findings),
            segmenter=StubSegmenter(),
            reporter=StubReporter(),
            name=BACKEND_STUB,
        )

    if cfg.backend == BACKEND_MODELS:
        from . import medgemma, medsam

        localizer, reporter = medgemma.build(cfg)
        return Backend(
            localizer=localizer,
            segmenter=medsam.build(cfg),
            reporter=reporter,
            name=BACKEND_MODELS,
        )

    if cfg.backend == BACKEND_VLLM:
        from . import medsam, vllm_medgemma

        localizer, reporter = vllm_medgemma.build(cfg)
        # Gate on vLLM readiness BEFORE loading the segmenter: a down server
        # fails fast with an actionable message and never touches torch.
        localizer.client.health()
        return Backend(
            localizer=localizer,
            segmenter=medsam.build(cfg),
            reporter=reporter,
            name=BACKEND_VLLM,
        )

    if cfg.backend == BACKEND_DETECTOR:
        from . import detector, medsam, vllm_medgemma

        localizer = detector.build(cfg)          # dedicated detector does the boxing
        _, reporter = vllm_medgemma.build(cfg)   # MedGemma kept ONLY as the Reporter
        reporter.client.health()                 # fail fast if the reporter is down
        return Backend(
            localizer=localizer,
            segmenter=medsam.build(cfg),
            reporter=reporter,
            name=BACKEND_DETECTOR,
        )

    raise ValueError(f"Unknown backend {cfg.backend!r}")

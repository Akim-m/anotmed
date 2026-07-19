"""Runtime configuration, read from the environment.

Kept dependency-free (plain os.getenv) so the core has no settings library to
install. Env is read at instantiation (via default_factory), not at import, so
`load_config()` reflects the current environment. See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_STUB = "stub"
BACKEND_MODELS = "medgemma+medsam"


@dataclass(frozen=True)
class Config:
    backend: str = field(
        default_factory=lambda: os.getenv("ANOTMED_BACKEND", BACKEND_STUB))
    storage_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ANOTMED_STORAGE", "./anotmed_data")).resolve())

    # Real-model settings (only read by the medgemma+medsam backend).
    device: str = field(default_factory=lambda: os.getenv("ANOTMED_DEVICE", "cuda"))
    medgemma_model: str = field(
        default_factory=lambda: os.getenv("ANOTMED_MEDGEMMA_MODEL", "google/medgemma-4b-it"))
    sam_checkpoint: str = field(default_factory=lambda: os.getenv("ANOTMED_SAM_CHECKPOINT", ""))
    sam_config: str = field(default_factory=lambda: os.getenv("ANOTMED_SAM_CONFIG", ""))

    # Suggestion limits.
    max_findings: int = field(default_factory=lambda: int(os.getenv("ANOTMED_MAX_FINDINGS", "8")))
    min_score: float = field(default_factory=lambda: float(os.getenv("ANOTMED_MIN_SCORE", "0.0")))


def load_config() -> Config:
    cfg = Config()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    return cfg

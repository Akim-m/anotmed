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
BACKEND_VLLM = "vllm"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


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

    # vLLM MedGemma backend (thin httpx client to a separate vLLM OpenAI server).
    vllm_url: str = field(
        default_factory=lambda: os.getenv("ANOTMED_VLLM_URL", "http://127.0.0.1:8000/v1"))
    vllm_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("ANOTMED_VLLM_TIMEOUT_S", "120")))
    guided_json: bool = field(default_factory=lambda: _env_bool("ANOTMED_GUIDED_JSON", True))

    # Suggestion limits.
    max_findings: int = field(default_factory=lambda: int(os.getenv("ANOTMED_MAX_FINDINGS", "8")))
    min_score: float = field(default_factory=lambda: float(os.getenv("ANOTMED_MIN_SCORE", "0.0")))


def load_config() -> Config:
    cfg = Config()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    return cfg

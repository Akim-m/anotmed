"""Runtime configuration, read from the environment.

Kept dependency-free (plain os.getenv) so the core has no settings library to
install. Env is read at instantiation (via default_factory), not at import, so
`load_config()` reflects the current environment. See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .modalities import active_profile

BACKEND_STUB = "stub"
BACKEND_MODELS = "medgemma+medsam"
BACKEND_VLLM = "vllm"
BACKEND_DETECTOR = "detector+medsam"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _envp(env: str, attr: str, default):
    """Resolve a knob by precedence: explicit env var > active modality profile > default."""
    raw = os.getenv(env)
    if raw not in (None, ""):
        return raw if isinstance(default, str) else type(default)(raw)
    val = getattr(active_profile(), attr)
    return val if val not in ("", None) else default


@dataclass(frozen=True)
class Config:
    backend: str = field(
        default_factory=lambda: os.getenv("ANOTMED_BACKEND", BACKEND_STUB))
    storage_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ANOTMED_STORAGE", "./anotmed_data")).resolve())

    # Real-model settings (only read by the medgemma+medsam backend).
    device: str = field(default_factory=lambda: os.getenv("ANOTMED_DEVICE", "cuda"))
    # Where the MedSAM-2 segmenter runs. Defaults to ANOTMED_DEVICE, but can be
    # pinned to "cpu" independently to free VRAM for vLLM when the GPU is tight.
    seg_device: str = field(default_factory=lambda: os.getenv(
        "ANOTMED_SEG_DEVICE", os.getenv("ANOTMED_DEVICE", "cuda")))
    medgemma_model: str = field(
        default_factory=lambda: os.getenv("ANOTMED_MEDGEMMA_MODEL", "google/medgemma-4b-it"))
    # Active modality profile (see modalities.py); "" = generic (today's behavior).
    modality: str = field(default_factory=lambda: os.getenv("ANOTMED_MODALITY", ""))

    # Knobs resolved by precedence: explicit env > modality profile > default.
    sam_checkpoint: str = field(default_factory=lambda: _envp("ANOTMED_SAM_CHECKPOINT", "sam_checkpoint", ""))
    sam_config: str = field(default_factory=lambda: _envp("ANOTMED_SAM_CONFIG", "sam_config", ""))

    # Object-detector Localizer (the "detector+medsam" backend). A dedicated
    # detector replaces MedGemma for localization; MedGemma stays as the Reporter.
    detector_weights: str = field(default_factory=lambda: _envp("ANOTMED_DETECTOR_WEIGHTS", "detector_weights", ""))
    detector_device: str = field(
        default_factory=lambda: os.getenv("ANOTMED_DETECTOR_DEVICE", os.getenv("ANOTMED_DEVICE", "cuda")))
    detector_conf: float = field(default_factory=lambda: _envp("ANOTMED_DETECTOR_CONF", "detector_conf", 0.25))
    detector_imgsz: int = field(default_factory=lambda: _envp("ANOTMED_DETECTOR_IMGSZ", "detector_imgsz", 1024))

    # vLLM MedGemma backend (thin httpx client to a separate vLLM OpenAI server).
    vllm_url: str = field(
        default_factory=lambda: os.getenv("ANOTMED_VLLM_URL", "http://127.0.0.1:8000/v1"))
    vllm_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("ANOTMED_VLLM_TIMEOUT_S", "120")))
    guided_json: bool = field(default_factory=lambda: _env_bool("ANOTMED_GUIDED_JSON", True))

    # Suggestion limits.
    max_findings: int = field(default_factory=lambda: int(os.getenv("ANOTMED_MAX_FINDINGS", "8")))
    min_score: float = field(default_factory=lambda: float(os.getenv("ANOTMED_MIN_SCORE", "0.0")))

    # Run the pipeline inline (True) or on the async worker (False). Defaults to
    # inline for the stub (fast tests/CI) and async for real backends, where
    # tens-of-seconds inference must not block the browser. ANOTMED_SYNC overrides.
    sync: bool = field(default_factory=lambda: _env_bool(
        "ANOTMED_SYNC", os.getenv("ANOTMED_BACKEND", BACKEND_STUB) == BACKEND_STUB))


def load_config() -> Config:
    cfg = Config()
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    return cfg

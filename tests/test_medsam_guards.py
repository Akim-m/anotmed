"""MedSAM-2 adapter guards — fail helpfully, and conform masks safely.

Two hardening contracts, both testable on CPU with no torch installed:

  1. `_load` must check config BEFORE importing torch/sam2, so a user who forgot
     to set ANOTMED_SAM_CHECKPOINT gets an actionable RuntimeError instead of a
     cryptic ModuleNotFoundError.
  2. `_conform_mask` must turn whatever the predictor returns (CxHxW, batched, or
     a mask at the model's internal resolution) into a bool array matching the
     image, so downstream measurement never sees a mismatched shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from anotmed.backends.medsam import _conform_mask, _load
from anotmed.config import Config


def _cfg_without_weights() -> Config:
    return Config(backend="medgemma+medsam", sam_checkpoint="", sam_config="")


def test_load_without_checkpoint_raises_runtimeerror_not_importerror():
    # torch is intentionally absent in the CPU venv; the guard must fire first.
    with pytest.raises(RuntimeError) as exc:
        _load(_cfg_without_weights())
    assert "ANOTMED_SAM_CHECKPOINT" in str(exc.value)


def test_conform_mask_passthrough_when_shape_matches():
    raw = np.zeros((32, 32), dtype=bool)
    raw[5, 5] = True
    out = _conform_mask(raw, 32, 32)
    assert out.shape == (32, 32)
    assert out[5, 5]


def test_conform_mask_collapses_leading_channel_dim():
    raw = np.zeros((1, 16, 16), dtype=bool)  # CxHxW / batched
    out = _conform_mask(raw, 16, 16)
    assert out.shape == (16, 16)


def test_conform_mask_resizes_model_resolution_to_image():
    raw = np.zeros((2, 2), dtype=bool)
    raw[0, 0] = True  # top-left quadrant only
    out = _conform_mask(raw, 4, 4)
    assert out.shape == (4, 4)
    assert out[0, 0]        # top-left stays set after nearest-neighbor upscale
    assert not out[3, 3]    # bottom-right stays clear


def test_conform_mask_returns_bool_dtype():
    raw = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)  # logits/probs
    out = _conform_mask(raw, 2, 2)
    assert out.dtype == np.bool_

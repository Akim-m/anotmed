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

from anotmed.backends.medsam import MedSAM2Segmenter, _conform_mask, _load
from anotmed.config import Config
from anotmed.schema import BBox, ImageMeta


def _cfg_without_weights() -> Config:
    return Config(backend="medgemma+medsam", sam_checkpoint="", sam_config="")


# --- fakes so segment() can be driven end-to-end with no torch ----------------

class _FakeTorch:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    def inference_mode(self):
        return _FakeTorch._Ctx()


class _FakePredictor:
    """Stands in for SAM2ImagePredictor: records inputs, replays a canned mask."""

    def __init__(self, mask):
        self.mask = mask
        self.image = None
        self.box = None
        self.set_image_calls = 0

    def set_image(self, rgb):
        self.image = rgb
        self.set_image_calls += 1

    def predict(self, box=None, multimask_output=False):
        self.box = box
        return np.stack([self.mask]), np.array([0.9]), None


def _segmenter(mask) -> MedSAM2Segmenter:
    cfg = Config(backend="medgemma+medsam", sam_checkpoint="/w/medsam2.pt", sam_config="/w/c.yaml")
    return MedSAM2Segmenter(cfg, predictor=_FakePredictor(mask), torch_mod=_FakeTorch())


def _meta() -> ImageMeta:
    return ImageMeta(study_id="s", rows=64, cols=64, modality="CT",
                     window_center=40, window_width=400)


# --- segment() shape contract: mask always comes out (rows, cols) bool --------

def test_segment_returns_image_shaped_bool_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    out = _segmenter(mask).segment(np.zeros((64, 64), np.float32),
                                   BBox(x=8, y=8, w=16, h=16), _meta())
    assert out.shape == (64, 64) and out.dtype == np.bool_
    assert out[15, 15]


def test_segment_resizes_model_resolution_mask_to_image():
    mask = np.zeros((32, 32), dtype=bool)  # predictor returns at half resolution
    mask[:16, :16] = True
    out = _segmenter(mask).segment(np.zeros((64, 64), np.float32),
                                   BBox(x=0, y=0, w=64, h=64), _meta())
    assert out.shape == (64, 64)
    assert out[0, 0] and not out[63, 63]


def test_segment_feeds_display_rgb_to_the_predictor():
    seg = _segmenter(np.zeros((64, 64), dtype=bool))
    seg.segment(np.zeros((64, 64), np.float32), BBox(x=0, y=0, w=10, h=10), _meta())
    assert seg._predictor.image.shape == (64, 64, 3)
    assert seg._predictor.image.dtype == np.uint8


def test_segment_clamps_box_into_the_image_before_predict():
    seg = _segmenter(np.zeros((64, 64), dtype=bool))
    seg.segment(np.zeros((64, 64), np.float32), BBox(x=-20, y=-20, w=200, h=200), _meta())
    x0, y0, x1, y1 = seg._predictor.box[0]
    assert x0 >= 0 and y0 >= 0 and x1 <= 64 and y1 <= 64


def _img_with_square(r0, c0):
    a = np.zeros((64, 64), np.float32)
    a[r0:r0 + 10, c0:c0 + 10] = 300.0
    return a


def test_segment_caches_embedding_across_findings_on_same_image():
    # The expensive set_image encode must run once, not once per finding.
    seg = _segmenter(np.zeros((64, 64), dtype=bool))
    img = _img_with_square(5, 5)
    seg.segment(img, BBox(x=6, y=6, w=8, h=8), _meta())
    seg.segment(img, BBox(x=30, y=30, w=8, h=8), _meta())  # same image, different box
    assert seg._predictor.set_image_calls == 1


def test_segment_re_encodes_when_the_image_changes():
    seg = _segmenter(np.zeros((64, 64), dtype=bool))
    seg.segment(_img_with_square(5, 5), BBox(x=6, y=6, w=8, h=8), _meta())
    seg.segment(_img_with_square(40, 40), BBox(x=6, y=6, w=8, h=8), _meta())
    assert seg._predictor.set_image_calls == 2


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

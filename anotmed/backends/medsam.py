"""MedSAM-2 / SAM2 adapter — box-prompted segmentation.

Takes a bounding box (from MedGemma) and returns a pixel mask. MedSAM-2 is
SAM2 fine-tuned on medical imaging; both expose the same image-predictor
interface, so its checkpoint drops into the config below.

Integration notes:
  * `sam2`/`torch` are imported lazily. Set ANOTMED_SAM_CHECKPOINT and
    ANOTMED_SAM_CONFIG to your MedSAM-2 weights + model config.
  * If your MedSAM-2 build exposes a differently named predictor, adjust
    `_load`. The predict call (box prompt, single mask) is stable across the
    SAM2 family.
  * Exercised in WSL with real weights, not by the CPU test suite.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..schema import BBox, ImageMeta


def _load(cfg: Config):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not cfg.sam_checkpoint or not cfg.sam_config:
        raise RuntimeError(
            "MedSAM-2 needs ANOTMED_SAM_CHECKPOINT and ANOTMED_SAM_CONFIG set."
        )
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = build_sam2(cfg.sam_config, cfg.sam_checkpoint, device=device)
    return SAM2ImagePredictor(model), torch


class MedSAM2Segmenter:
    name = "medsam2-segmenter"

    def __init__(self, cfg: Config):
        self._predictor, self._torch = _load(cfg)
        self.name = f"medsam2-segmenter:{cfg.sam_checkpoint.split('/')[-1] or 'sam2'}"

    def segment(self, image: np.ndarray, box: BBox, meta: ImageMeta) -> np.ndarray:
        from ..io_dicom import to_display_array

        h, w = image.shape
        rgb = to_display_array(image, meta)  # HxWx3 uint8
        b = box.clamp(w, h)
        xyxy = np.array([b.x, b.y, b.x + b.w, b.y + b.h], dtype=np.float32)

        self._predictor.set_image(rgb)
        with self._torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                box=xyxy[None, :], multimask_output=False
            )
        mask = np.asarray(masks[0]).astype(bool)
        if mask.shape != (h, w):  # some builds return CxHxW or resized masks
            mask = np.asarray(masks).reshape(-1, *masks.shape[-2:])[0].astype(bool)
        return mask


def build(cfg: Config) -> MedSAM2Segmenter:
    return MedSAM2Segmenter(cfg)

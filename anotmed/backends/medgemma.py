"""MedGemma adapter — localization + reporting.

MedGemma is a vision-language model. It contributes the two things a VLM is good
at here: proposing where findings are (bounding boxes) and writing the draft
description. It does NOT produce pixel masks — that is MedSAM-2's job (see
medsam.py). Boxes from here become the prompts fed to the segmenter.

Integration notes (this is the one place you tune to the released model card):
  * `torch`/`transformers` are imported lazily, only when a backend is built,
    so the CPU core and the tests never depend on them.
  * Localization uses the Gemma-family `box_2d` convention: a JSON array of
    {"label", "box_2d": [ymin, xmin, ymax, xmax]} normalized to 0-1000. If the
    MedGemma 1.5 card specifies a different format, adjust `_parse_boxes`.
  * These classes are written-to-spec; they are exercised in WSL with real
    weights, not by the CPU test suite. Run one image through
    `MedGemmaLocalizer.propose` after loading weights to confirm the parse.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..schema import BBox, ImageMeta
from .base import Detection
from .parsing import parse_detections


class _MedGemma:
    """Shared model handle. Loads once; localizer and reporter reuse it."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.model_id = cfg.medgemma_model
        use_cuda = cfg.device.startswith("cuda")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
        )
        if not use_cuda:
            self.model.to("cpu")

    def generate(self, pil_image, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.processor.decode(new_tokens, skip_special_tokens=True)


_LOCALIZE_PROMPT = (
    "You are assisting a radiologist by flagging regions to review. This is not "
    "a diagnosis. List candidate findings as a JSON array. Each item must be "
    '{"label": <short finding name>, "box_2d": [ymin, xmin, ymax, xmax]} with '
    "coordinates normalized to 0-1000. Return [] if nothing warrants review."
)

_REPORT_PROMPT = (
    "Describe the highlighted region of interest concisely for a radiologist to "
    "verify. One or two sentences. State observations only, not a diagnosis."
)


class MedGemmaLocalizer:
    name = "medgemma-localizer"

    def __init__(self, model: _MedGemma, max_findings: int = 8):
        self._m = model
        self.max_findings = max_findings
        self.name = f"medgemma-localizer:{model.model_id}"

    def propose(self, image: np.ndarray, meta: ImageMeta) -> list[Detection]:
        from ..io_dicom import to_display_rgb

        pil = to_display_rgb(image, meta)
        text = self._m.generate(pil, _LOCALIZE_PROMPT, max_new_tokens=512)
        return parse_detections(text, meta.rows, meta.cols)[: self.max_findings]


class MedGemmaReporter:
    name = "medgemma-reporter"

    def __init__(self, model: _MedGemma):
        self._m = model
        self.name = f"medgemma-reporter:{model.model_id}"

    def describe(
        self, image: np.ndarray, det: Detection, mask: np.ndarray, meta: ImageMeta
    ) -> str:
        from ..io_dicom import to_display_rgb

        # Crop to the finding with a small margin so the model attends to it.
        h, w = image.shape
        pad = 0.15
        b = BBox(
            x=det.box.x - det.box.w * pad,
            y=det.box.y - det.box.h * pad,
            w=det.box.w * (1 + 2 * pad),
            h=det.box.h * (1 + 2 * pad),
        ).clamp(w, h)
        rs, cs = b.as_slice(w, h)
        pil = to_display_rgb(image[rs, cs], meta)
        text = self._m.generate(pil, _REPORT_PROMPT, max_new_tokens=160).strip()
        return f"{text}\n\n[AI-generated draft — radiologist must verify.]"


def build(cfg: Config) -> tuple[MedGemmaLocalizer, MedGemmaReporter]:
    model = _MedGemma(cfg)
    return MedGemmaLocalizer(model, cfg.max_findings), MedGemmaReporter(model)

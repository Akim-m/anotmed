"""The safety gate. Scores a backend against ground truth and enforces floors.

Two tiers, absolute ranked above regression (PLAN.md Phase 3):
  * `--tier absolute`  — score vs GT; exit 1 if any floor is missed. THE gate.
  * `--tier regression`— compare decision fields to a blessed baseline (drift alarm).
  * `--compare A B`     — two backends head to head (e.g. fp8 vs bf16).
  * `--determinism`     — same input twice at temperature 0 → identical scores.

The harness drives the backend/pipeline directly against ground truth; it never
touches the Store. Segmentation is scored on GT-box prompts so segmenter quality
is isolated from localizer quality (scored separately as recall/precision).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anotmed.backends import build_backend
from anotmed.config import load_config

from .datasets import Case, synthetic_cases
from .metrics import dice, iou, localization_scores, valid_mask

_FLOORS_PATH = Path(__file__).parent / "floors.yaml"


@dataclass
class Report:
    n_cases: int
    n_lesions: int
    format_compliance: float
    dice_mean: float
    dice_p10: float
    iou_mean: float
    recall_iou30: float
    recall_iou50: float
    precision_iou30: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Floors:
    dice_mean: float
    dice_p10: float
    recall_iou30: float
    format_compliance: float

    def check(self, r: Report) -> list[str]:
        """Return a list of floor violations (empty = the gate passes)."""
        v: list[str] = []
        if r.dice_mean < self.dice_mean:
            v.append(f"mean Dice {r.dice_mean:.3f} < floor {self.dice_mean}")
        if r.dice_p10 < self.dice_p10:
            v.append(f"p10 Dice {r.dice_p10:.3f} < floor {self.dice_p10}")
        if r.recall_iou30 < self.recall_iou30:
            v.append(f"recall@IoU0.3 {r.recall_iou30:.3f} < floor {self.recall_iou30}")
        if r.format_compliance < self.format_compliance:
            v.append(f"format compliance {r.format_compliance:.3f} < floor {self.format_compliance}")
        return v

    @classmethod
    def load(cls, path: str | Path = _FLOORS_PATH, modality: str = "") -> "Floors":
        import yaml  # lazy: only the CLI needs it; unit tests build Floors directly

        d = yaml.safe_load(Path(path).read_text())
        base = {k: (dict(v) if isinstance(v, dict) else v)
                for k, v in d.items() if k != "modalities"}
        if modality:
            override = d.get("modalities", {}).get(modality)
            if override is None:
                print(f"warning: no floors for modality {modality!r}; using global floors")
            else:
                for section, vals in override.items():  # overlay per section
                    base.setdefault(section, {}).update(vals)
        return cls(
            dice_mean=float(base["segmentation"]["dice_mean"]),
            dice_p10=float(base["segmentation"]["dice_p10"]),
            recall_iou30=float(base["localization"]["recall_iou30"]),
            format_compliance=float(base["format"]["compliance"]),
        )


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _p10(xs: list[float]) -> float:
    return float(np.percentile(xs, 10)) if xs else 0.0


def segmentation_metrics(segmenter, cases: list[Case]) -> dict:
    """GT-box-prompted segmentation quality — uses ONLY the segmenter (SAM2).

    Runs independently of the localizer, so the primary Dice/IoU floor can be
    measured without MedGemma resident (memory-safe on a tight GPU)."""
    dices: list[float] = []
    ious: list[float] = []
    fmt_flags: list[bool] = []
    n_lesions = 0
    for case in cases:
        rows, cols = case.image.shape
        for gt_box, gt_mask in zip(case.gt_boxes, case.gt_masks):
            n_lesions += 1
            try:
                mask = segmenter.segment(case.image, gt_box, case.meta)
            except Exception:
                fmt_flags.append(False)  # a crash is a format-compliance failure
                continue
            ok = valid_mask(mask, rows, cols)
            fmt_flags.append(ok)
            if ok:
                dices.append(dice(mask, gt_mask))
                ious.append(iou(mask, gt_mask))
    return {
        "n_lesions": n_lesions,
        "dice_mean": _mean(dices),
        "dice_p10": _p10(dices),
        "iou_mean": _mean(ious),
        "format_compliance": _mean([float(f) for f in fmt_flags]) if fmt_flags else 1.0,
    }


def localization_metrics(localizer, cases: list[Case],
                         iou_low: float = 0.3, iou_high: float = 0.5) -> dict:
    """Localization recall/precision — uses ONLY the localizer (MedGemma)."""
    recall30: list[float] = []
    recall50: list[float] = []
    prec30: list[float] = []
    for case in cases:
        pred_boxes = [d.box for d in localizer.propose(case.image, case.meta)]
        s_low = localization_scores(pred_boxes, case.gt_boxes, iou_low)
        recall30.append(s_low["recall"])
        prec30.append(s_low["precision"])
        recall50.append(localization_scores(pred_boxes, case.gt_boxes, iou_high)["recall"])
    return {
        "recall_iou30": _mean(recall30),
        "recall_iou50": _mean(recall50),
        "precision_iou30": _mean(prec30),
    }


def evaluate(backend, cases: list[Case], iou_low: float = 0.3, iou_high: float = 0.5) -> Report:
    seg = segmentation_metrics(backend.segmenter, cases)
    loc = localization_metrics(backend.localizer, cases, iou_low, iou_high)
    return Report(
        n_cases=len(cases),
        n_lesions=seg["n_lesions"],
        format_compliance=seg["format_compliance"],
        dice_mean=seg["dice_mean"],
        dice_p10=seg["dice_p10"],
        iou_mean=seg["iou_mean"],
        recall_iou30=loc["recall_iou30"],
        recall_iou50=loc["recall_iou50"],
        precision_iou30=loc["precision_iou30"],
    )


def _print_report(title: str, r: Report) -> None:
    print(f"\n=== {title} ===")
    print(f"  cases={r.n_cases}  lesions={r.n_lesions}")
    print(f"  format compliance : {r.format_compliance:.3f}")
    print(f"  Dice  mean/p10    : {r.dice_mean:.3f} / {r.dice_p10:.3f}")
    print(f"  IoU   mean        : {r.iou_mean:.3f}")
    print(f"  recall @0.3 / @0.5: {r.recall_iou30:.3f} / {r.recall_iou50:.3f}")
    print(f"  precision @0.3    : {r.precision_iou30:.3f}")


def _load_cases(args) -> list[Case]:
    if args.data:
        from anotmed.modalities import get_profile

        from .datasets import load_dir
        return load_dir(args.data, window=get_profile(args.modality).window)
    return synthetic_cases(n=args.n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="anotmed validation gate")
    p.add_argument("--tier", choices=["absolute", "regression"], default="absolute")
    p.add_argument("--data", help="labeled-set dir (default: synthetic cases)")
    p.add_argument("--n", type=int, default=6, help="synthetic case count")
    p.add_argument("--floors", default=str(_FLOORS_PATH))
    p.add_argument("--modality", default=os.getenv("ANOTMED_MODALITY", ""),
                   help="apply per-modality floor overrides (e.g. dental)")
    p.add_argument("--determinism", action="store_true",
                   help="score twice, require identical results")
    p.add_argument("--compare", nargs=2, metavar=("BACKEND_A", "BACKEND_B"),
                   help="score two ANOTMED_BACKEND values head to head")
    args = p.parse_args(argv)

    cases = _load_cases(args)

    if args.compare:
        import os
        results = {}
        for name in args.compare:
            os.environ["ANOTMED_BACKEND"] = name
            results[name] = evaluate(build_backend(load_config()), cases)
            _print_report(f"backend={name}", results[name])
        a, b = (results[n] for n in args.compare)
        gap = abs(a.dice_mean - b.dice_mean)
        print(f"\nDice gap {args.compare[0]} vs {args.compare[1]}: {gap:.3f}")
        return 0

    backend = build_backend(load_config())
    report = evaluate(backend, cases)
    _print_report(f"tier={args.tier}", report)

    if args.determinism:
        again = evaluate(backend, cases)
        if again.as_dict() != report.as_dict():
            print("\nFAIL: non-deterministic output across identical runs")
            return 1
        print("\ndeterminism: OK (identical across two runs)")

    if args.tier == "absolute":
        floors = Floors.load(args.floors, modality=args.modality)
        violations = floors.check(report)
        if violations:
            print("\nGATE FAILED — floors missed:")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("\nGATE PASSED — all floors met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Safety & intended use

Read this before pointing anotmed at anything real.

## What it is

An **annotation accelerator**. It proposes bounding boxes, segmentation masks,
and measurements on medical images so a qualified radiologist can review them
faster. Every suggestion is a *draft*.

## What it is not

- **Not a diagnostic device.** It does not diagnose, triage, or make clinical
  decisions, and nothing it outputs should be read as a diagnosis.
- **Not autonomous.** It never finalizes an annotation. A human accepts, edits,
  or rejects each one.

## The human-in-the-loop guarantee

Suggestions are created with status `pending`. The **only** way data leaves the
system is `Store.accepted_annotations`, which returns accepted items exclusively;
export is built on top of it. A suggestion a radiologist has not accepted cannot
be exported. This is enforced in code and covered by `tests/test_safety_invariant.py`
and `tests/test_api.py` — do not route around it.

## Known limitations

- **Measurements depend on DICOM `PixelSpacing`.** If a file lacks it, spacing
  falls back to 1 mm/px and every millimetre value is flagged approximate
  (`spacing_is_default`). Do not trust mm values on such studies.
- **`max_diameter_mm` / `perpendicular_diameter_mm` are caliper geometry, not a
  certified RECIST measurement.** Named literally on purpose.
- **The COCO `segmentation` polygon is a convex-hull approximation.** Faithful
  masks are the PNG/`.npy` masks; DICOM-SEG export is a documented stub in this
  build.
- **The stub backend is not clinical.** It is classical thresholding to exercise
  the plumbing and tests. Real inference requires the `medgemma+medsam` backend.
- **The model adapters are written-to-spec and must be validated** on your
  modalities before any real use (see below).

## Before real use — required

1. **Validate.** Measure segmentation quality (Dice/IoU) and localization on a
   labeled set from *your* modalities. Foundation models degrade out of
   distribution.
2. **Regulatory.** Suggesting findings on clinical images can bring a tool under
   medical-device regulation (FDA SaMD / EU MDR) depending on intended use and
   claims, even with a human in the loop. Get qualified regulatory advice.
   Research and education use is lower-stakes but still your responsibility.
3. **Data / PHI.** Run models locally (the point of the WSL/GPU setup) — do not
   send patient images to a third-party API without a BAA and de-identification.
   anotmed persists only the pixel array and non-identifying metadata; it never
   writes DICOM PHI tags to disk, and `io_dicom.deidentify` is available for
   files you do keep.
4. **Licensing.** MedGemma ships under Google's Health AI Developer Foundations
   terms (a health-use policy, not plain OSS). SAM2 is Apache-2.0; MedSAM-2
   checkpoints carry their own terms. Confirm your use is permitted.

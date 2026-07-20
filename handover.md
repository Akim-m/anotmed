# Handover — anotmed

For whoever picks this up next (including a future you, or a fresh session).
Read this first, then [PLAN.md](PLAN.md) for the full staged roadmap,
[README.md](README.md) for run steps, and [SAFETY.md](SAFETY.md) for
intended-use limits.

## TL;DR

A medical-image **annotation accelerator**: MedGemma proposes boxes + draft text,
MedSAM-2 turns each box into a pixel mask, a pure-numpy engine computes the
millimetre measurements, and a radiologist accepts/edits/rejects every suggestion
in a web UI. Nothing exports until a human accepts it. The whole thing runs on
CPU today via a **stub backend**; the real models plug in behind two clearly
marked adapters for a GPU/WSL run. **The CPU core is now verified green** — the
full test suite passes and the upload→review→export loop was exercised against
the live HTTP server (see Session log). The real-model adapters remain
written-to-spec, not yet run against weights; that is where the work resumes.

## Current state — what's real vs. not

| Component | State |
|---|---|
| Domain schema, config, filesystem store | complete |
| Measurement engine (`measure.py`) | complete, test-covered (geometry only) |
| DICOM load / window / de-id / COCO export | complete |
| Pipeline (localize→segment→measure→report) | complete |
| Stub backend (Otsu + connected components) | complete — **not clinical**, exercises plumbing/tests |
| FastAPI service + review UI | complete |
| MedGemma adapter (`backends/medgemma.py`) | **written-to-spec, not run** — needs 1 alignment pass |
| MedSAM-2 adapter (`backends/medsam.py`) | **written-to-spec, not run** — needs your checkpoint |
| COCO export | complete |
| DICOM-SEG export | **documented stub** — raises with root cause; needs source series wired |
| 3D volumes | not implemented — per-slice scaffolding only (`slice_index` exists) |

## Session log — 2026-07-20 (Phase 1: MedGemma behind vLLM, GPU-free)

Built the whole vLLM MedGemma path and its tests without a GPU — the app is now
a thin httpx client to a separate vLLM server, so the entire backend is proven by
mocking one request. Landed in four green sub-commits (suite 30 → 51):

- **`backends/schemas.py`** — self-contained guided-JSON schema (`box_2d` = 4 ints
  0-1000, capped at `max_findings`) for vLLM's grammar-constrained decoder, plus
  `FindingBox`/`FindingList` to validate responses. Confidence is advisory only.
- **`parsing.py`** — extracted `to_pixel_bbox` so the guided path and the P0
  free-text fallback share one clamp/scale and can't drift.
- **`backends/vllm_medgemma.py`** — `_VllmClient` (injectable http, `temperature=0`,
  data-URL image, `guided_json`, actionable `health()`), `VllmLocalizer`
  (guided-JSON → validated Detections; prose → `parsing.py` fallback; garbage → []
  never raises), `VllmReporter` (crop + label → draft + verify disclaimer).
- **Wiring** — `build_backend` routes `ANOTMED_BACKEND=vllm` and **health-gates the
  server before loading the segmenter** (down server fails fast, never imports
  torch). Uniform `{error, message}` API envelope. `httpx` promoted to a runtime
  dep. `scripts/serve_vllm.sh` launches vLLM with the §2.1 two-model VRAM budget.

Config knobs added: `ANOTMED_VLLM_URL`, `ANOTMED_VLLM_TIMEOUT_S`,
`ANOTMED_GUIDED_JSON`. **Live GPU run is owner-gated (Phase 2)**; the deferred
deletion of `backends/medgemma.py` waits until that live E2E passes.

## Session log — 2026-07-20 (Phase 0: adapter hardening, WSL/CPU)

Hardened the two real-model adapters against messy output — TDD, tests written
and watched fail first. Six latent bugs that would only bite once real weights
run, now fixed and CPU-covered (19 new tests; suite 11 → 30, all green):

- **New `backends/parsing.py`** (torch-free, unit-tested) owns MedGemma output
  parsing: string-aware array extraction (`json.raw_decode`, so a `]` inside a
  label no longer truncates), per-element `try/except` (one bad coord drops one
  box, not the study), coord clamp `0-1000` → `BBox.clamp`, **default score 1.0**
  (a `0.0` default was filtered out the instant `ANOTMED_MIN_SCORE>0` — see
  `pipeline.py:87`), and a "never raises" guarantee. `medgemma.py` now delegates
  to it; its buggy `_extract_json_array`/`_parse_boxes` are gone.
- **`medsam.py`**: config guard now runs *before* the `torch`/`sam2` import (a
  missing checkpoint yields an actionable `RuntimeError`, not `ModuleNotFoundError`
  from an uninstalled torch), and `_conform_mask` collapses leading dims + does a
  numpy nearest-neighbor resize to the image grid + a shape assert.

Pushed to `main` (`dd7c199`). Backlog #2 is partly addressed: the parser is now
robust; the remaining half is a **1-image verification against real weights**.

## Session log — 2026-07-19 (first execution, WSL/CPU)

First time the code was ever run. Set up a dedicated `.venv`, `pip install -e
".[dev]"` (installs pytest+httpx; `requirements.txt` alone is not enough), and:

- **`pytest -q` → 11/11 pass.** One initial failure was a first-run bug: the stub
  reporter said "for radiologist review" but `test_pipeline` asserts the draft
  invites *verification* (the project's "radiologists only verify" doctrine).
  Aligned the reporter wording to "review and verify" (`backends/stub.py`).
- **Defused a pydicom-4.0 time-bomb** in `examples/make_sample_dicom.py`: the
  deprecated `is_little_endian` / `is_implicit_VR` attribute assignments now go
  through `save_as(little_endian=, implicit_vr=)` on pydicom 3.x with a
  `try/except TypeError` fallback for the `>=2.4` floor. Warnings 9 → 1 (the one
  remaining is FastAPI's own `TestClient`, not our code).
- **Live-server end-to-end verified** (not just TestClient): booted
  `anotmed-serve`, uploaded the synthetic DICOM → 2 findings (the 2 planted
  lesions), export blocked 409 while pending, accepted 1 of 2, export returned
  exactly the 1 accepted (safety gate holds through the real HTTP path),
  image/mask PNGs render, dicom-seg returns its documented 501.

The install pulled the latest stack (pydicom 3.0.2, numpy 2.5.1, pydantic 2.13.4,
fastapi 0.139.2, pytest 9.1.1) with no code changes needed beyond the above.

Environment note: the shell's default `python` is another project's venv
(`troke`), with no numpy/pydicom. Always use `anotmed/.venv/bin/python`.

## Verification status & plan

The CPU core has now been run (see Session log). Written tests and their
coverage boundaries:

| Test | Proves | Does NOT cover |
|---|---|---|
| `test_measure.py` | area/diameter correct on circle, rectangle, empty, single-pixel; exact under anisotropic spacing | real anatomical masks; 3D |
| `test_safety_invariant.py` | export returns accepted-only | concurrency; race between edit + export |
| `test_pipeline.py` | stub finds + measures, marks PENDING, persists | real model quality |
| `test_dicom_roundtrip.py` | DICOM → pixels + spacing + PNG | scanner-DICOM variety (one synthetic file) |
| `test_api.py` | upload→review→export loop + the 409 export gate | model backend; multi-slice |

To verify (WSL, CPU, no GPU):

```bash
cd ~/anotmed && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The stub path is CPU-only and uses negligible RAM / zero VRAM, so it is safe to
run alongside GPU workloads.

## Design decisions & rationale

Recorded so they don't get re-litigated:

- **Two models, not one.** MedGemma is a vision-language model — good at *where*
  and *what to call it*, cannot emit pixel masks. MedSAM-2 does masks. Splitting
  the roles (`backends/base.py` protocols) is why the stub can stand in cleanly.
- **Measurements are geometry, never LLM output.** The numbers a clinician reads
  must be reproducible and auditable, so `pipeline.py` computes them from the
  mask via `measure.py`; the reporter only writes prose.
- **Physical (mm) space throughout `measure.py`.** CT/MR spacing is often
  anisotropic (dy ≠ dx); computing Feret diameters in pixel space would be wrong.
- **Human-in-the-loop is enforced in code, not convention.** `Store.accepted_annotations`
  is the sole export path. This is the "radiologists only verify" requirement made
  unbypassable.
- **Filesystem store, no DB.** So it runs the instant you start it. Swap for a DB
  only if you outgrow a single workstation.
- **Stub backend is first-class.** It makes the entire system testable on CPU with
  no weights — the reason there is a test suite at all.

## The one invariant you must not break

Only `ReviewStatus.ACCEPTED` annotations may leave the system, and only through
`Store.accepted_annotations`. If you add an export format, an API route, or a
batch job, route it through that method. Two tests guard this — keep them green.

## Two integration points (WSL, with weights)

Both are the only places you should need to touch to go from stub to real:

1. **`backends/medgemma.py::_parse_boxes`** — expects Gemma-family `box_2d` JSON
   (normalized 0–1000). Load weights, run one image through
   `MedGemmaLocalizer.propose`, print the raw generation, and align the prompt/
   parser to whatever MedGemma 1.5 actually emits.
2. **`backends/medsam.py::_load`** — standard SAM2 image-predictor API. If your
   MedSAM-2 build names its predictor differently, fix that one function. Set
   `ANOTMED_SAM_CHECKPOINT` / `ANOTMED_SAM_CONFIG`.

## Known risks / uncertainties

- **pydicom 2.x vs 3.x** — `save_as` semantics differ; the sample writer targets
  both but a run confirms it. First `pytest` will surface any issue.
- **MedGemma output format** — see integration point #1; unverified until run.
- **Stub is not diagnostic** — do not demo it as clinical output.
- **OneDrive + WSL** — copy the repo to a native WSL path; keep venv/model cache
  out of OneDrive.
- **COCO `segmentation` is a convex-hull approximation** — faithful masks are the
  `.npy`/PNG; DICOM-SEG is the clinical-fidelity path (still a stub).

## Backlog — ranked

1. ~~Run `pytest -q` in WSL — confirm the CPU core is green.~~ **DONE 2026-07-19**
   (11/11 green + live-server loop verified; see Session log). Real work resumes at #2.
2. **Align the MedGemma adapter** to the real model card; verify on 1 image.
3. **Align the MedSAM-2 loader** to your checkpoint; verify box→mask on 1 image.
4. **Validate** Dice/IoU + localization on a labeled set from *your* modalities
   (SAFETY requirement — do before any real use).
5. **Wire DICOM-SEG export** — retain the source series (`Study.source_path` is
   already there) and build the segmentation via highdicom.
6. **Confirm target modality**; tune default windowing; add 3D volume support if
   needed (loop the pipeline over slices — `slice_index` is already plumbed).
7. **Hardening if scaling** beyond one workstation: auth, multi-user, a real DB.

## Open questions for the owner

- **Which modality?** I defaulted to 2D CT/MR slices (you didn't pin one). This
  drives the MedSAM-2 checkpoint choice and the validation set more than anything.
- **Deployment shape?** Single radiologist workstation, or a shared service?
- **Canonical export?** COCO is live now; DICOM-SEG matters if this feeds a PACS.

## Resuming cold

Everything needed to rediscover state is on disk: this file, [README.md](README.md),
[SAFETY.md](SAFETY.md), the `tests/` (which document expected behavior), and the
`← integration point` comments in the two adapter files. Start at step #1 of the
backlog.

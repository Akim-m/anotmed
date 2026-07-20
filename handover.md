# Handover — anotmed

For whoever picks this up next (including a future you, or a fresh session).
Read this first, then [PLAN.md](PLAN.md) for the full staged roadmap,
[README.md](README.md) for run steps, and [SAFETY.md](SAFETY.md) for
intended-use limits.

## TL;DR

A medical-image **annotation accelerator**: MedGemma proposes boxes + draft text,
MedSAM-2 turns each box into a pixel mask, a pure-numpy engine computes the
millimetre measurements, and a radiologist accepts/edits/rejects every suggestion
in a web UI. Nothing exports until a human accepts it.

**Status: all CPU/solo-buildable work is done and verified green (88 tests + a
full live-HTTP smoke).** MedGemma now runs behind a **vLLM server** (the app is a
thin client — real inference is GPU, app + tests are GPU-free); there is an
**async submit+poll** job model, an **absolute Dice/IoU safety gate** (`eval/`),
and real **DICOM-SEG + exact-RLE COCO** export. What remains is **irreducibly the
owner's**: run the models on your WSL/GPU box against real weights, validate the
safety gate on a labeled set in your modality, and decide the modality. See the
"Owner-gated remainder" section and PLAN.md §4.

## Current state — what's real vs. not

| Component | State |
|---|---|
| Domain schema, config, filesystem store | complete |
| Measurement engine (`measure.py`) | complete, test-covered (geometry only) |
| DICOM load / window / de-id / COCO export | complete |
| Pipeline (localize→segment→measure→report) | complete |
| Stub backend (Otsu + connected components) | complete — **not clinical**, exercises plumbing/tests |
| FastAPI service + review UI | complete; async submit+poll + uniform `{error,message}` errors |
| MedGemma via vLLM (`backends/vllm_medgemma.py`) | **LIVE on the RTX 4060** (FP8, vLLM 0.23) — describes images correctly, emits valid guided JSON, driven by `VllmLocalizer` end-to-end |
| MedGemma in-process (`backends/medgemma.py`) | legacy fallback; **delete after the vLLM live run passes** |
| MedSAM-2 adapter (`backends/medsam.py`) | **LIVE on the RTX 4060** — real box→mask verified with SAM2.1-hiera-small weights (focused mask, shape contract holds). A MedSAM-2 medical checkpoint is a drop-in (same predictor API) |
| Safety gate (`eval/`) | **complete on stub+synthetic**; correctly refuses the stub. Real run needs a labeled set (P3b) |
| Async jobs (`jobs.py`) | complete — one worker = GPU serialization point |
| COCO export | complete — now emits exact-mask RLE (not a hull approximation) |
| DICOM-SEG export | **complete** — highdicom, accepted-only, references the de-identified source |
| 3D volumes | not implemented — `slice_index` plumbed; gated on the modality decision (P6) |

## Session log — 2026-07-20 (MedSAM-2 segmenter LIVE + real memory findings)

Brought the segmenter up on real weights. Installed `sam2` into the vLLM venv
(reusing its torch 2.11+cu130, `SAM2_BUILD_CUDA=0` — no nvcc here), downloaded
`sam2.1_hiera_small.pt` (176 MB), and ran anotmed's `MedSAM2Segmenter` end-to-end:
a box around a synthetic lesion → a focused (256,256) bool mask, 1321 px (2% of
image), correctly covering the lesion. **The `segment()` wiring works with real
weights; a MedSAM-2 medical checkpoint drops in unchanged** (same SAM2 predictor
API). Config used: `ANOTMED_SAM_CHECKPOINT=/root/sam2_checkpoints/sam2.1_hiera_small.pt`,
`ANOTMED_SAM_CONFIG=configs/sam2.1/sam2.1_hiera_s.yaml`.

**Real memory findings (correct the §2.1 estimate):**
- MedGemma-4b FP8 needs **≥ ~0.70** `--gpu-memory-utilization` — at 0.60 vLLM
  fails with "No available memory for the cache blocks" (weights ~4.5 GiB +
  overhead already exceed 0.60×8 GiB before any KV cache). MedGemma-only: ~0.85.
- Two-model coexistence on 8 GiB is tight (~6.9 GiB: vLLM at 0.72 + SAM2 ~1 GiB).
  It *fits*, but per Akim we don't force it. **The full pipeline interleaves
  MedGemma (localize+report) with SAM2 (segment), so it needs both resident** —
  so for this box, prefer running the segmenter on CPU (`ANOTMED_SEG_DEVICE=cpu`,
  the knob added in Phase 2) or serialize, rather than the tight GPU co-fit.
- Both models are verified **individually** on the GPU; that's the milestone.

## Session log — 2026-07-20 (MedGemma running LIVE on the GPU)

Turns out this box HAS the GPU (RTX 4060, 8 GiB) and MedGemma-4b-it was already
cached (8.1 GiB). Stood up **vLLM 0.23 serving MedGemma with FP8** and drove it
through anotmed's real backend. What it took (all captured in `serve_vllm.sh`):

- **FP8, not bf16** — bf16 4B (~8.5 GiB) won't fit 8 GiB; FP8 (~4.5 GiB) does.
- **`HF_HUB_OFFLINE=1` + `--chat-template <snapshot>/chat_template.jinja`** — the
  repo is gated and the cache lacks `chat_template.json` (only `.jinja`), so an
  online boot 401s. Offline + the local template fixes it, no token needed.
- **`--gpu-memory-utilization` sized to free VRAM** (0.85 here) after killing
  stragglers; `--enforce-eager`, `--max-num-seqs 1`, `--max-model-len 2048`.

Result: MedGemma correctly described a test image ("two bright circular objects…
a test phantom"), and `VllmLocalizer.propose` parsed its guided JSON into
Detections. Box quality on the synthetic phantom is poor/degenerate (whole-image,
sometimes duplicate) — expected on non-anatomy; **this is precisely what the
Phase 3b safety-gate run on real labeled data is for.** Phase 1 🟡 → ✅ live.

## Session log — 2026-07-20 (real-data validation + the modality finding)

Ran the safety gate on **real medical data** (Kvasir-SEG, 1000 colonoscopy polyps
+ masks + boxes; loader layout via `eval/datasets.load_dir`). Two honest misses,
both tracing to the **wrong modality**:

- **Segmentation** (GT-box-prompted, 200 cases / 239 lesions):
  base SAM2.1-tiny **0.880 / 0.500** Dice mean/p10 · small 0.858 / 0.333 ·
  **MedSAM2 (medical) 0.824 / 0.222** — the medical fine-tune is **worse** on
  endoscopy (−0.056 Dice, −0.278 p10 at matched tiny architecture).
- **Localization** (MedGemma, 50 cases): recall@0.3 **0.370**, @0.5 0.100.

**Finding:** MedSAM2's checkpoints (`CTLesion`, `MRI_LiverLesion`, `US_Heart`) are
**radiology**-trained; colonoscopy RGB is out-of-domain, so the fine-tune *narrowed*
base SAM2's broad capability. Both misses point the same way — **the lever is the
modality, not the checkpoint.** Also: even the best model's p10 (0.500) misses the
0.70 floor on polyps — a likely modality-inherent tail. The two-tier floor (mean +
p10) earned its keep by exposing it.

**In progress:** radiology validation on **MSD Task06 Lung CT** (in-domain for both
MedSAM2 and MedGemma) to test the modality hypothesis — base SAM2 vs
`MedSAM2_CTLesion`, and MedGemma localization on CT.

Tooling used (repo): `eval/datasets.load_dir` (real sets), `eval/run` split into
`segmentation_metrics`/`localization_metrics` (one model at a time, memory-safe).
Checkpoints live in `/root/sam2_checkpoints/` (not in the repo).

## Owner-gated remainder — what only your hardware/data/decisions can close

Everything below needs something I cannot supply in a CPU sandbox. The code paths
that lead into each are built and CPU-tested; these are the live/decision steps.

1. ~~**Run MedGemma on the GPU (P1 🟡).**~~ **DONE 2026-07-20** — live on the RTX
   4060 via `scripts/serve_vllm.sh` (FP8, offline). Remaining P2 work is the
   *segmenter*: MedSAM-2 on real weights coexisting with vLLM in the VRAM budget.
2. **Point MedSAM-2 at real weights (P2).** Set `ANOTMED_SAM_CHECKPOINT`/`_CONFIG`
   (checkpoint choice is modality-adjacent — see §4). Then delete `medgemma.py`.
3. **Validate on real data (P3b) — the safety requirement.** `eval/datasets.load_dir()`
   is now **implemented + tested** (put images in `<dir>/images/<stem>.{png,npy}` and
   masks in `<dir>/masks/<stem>.{png,npy}`; binary masks split into per-lesion
   components, label maps use per-value instances). Verified end-to-end on the stub:
   `python -m eval.run --tier absolute --data <dir>` scores and gates. **All that's
   left is your labeled held-out set** (30–50 curated cases is a meaningful floor
   check) + real models; sign the floors in `eval/floors.yaml` first.
4. **Pick the modality (P6).** Drives the checkpoint, windowing, validation set,
   and whether 3D volumes are needed.
5. **(Conditional) QLoRA finetune (P7)** — only if P3b shows a Dice gap.
6. Live PACS-viewer check of an exported DICOM-SEG over the source series.

## Session log — 2026-07-20 (Phase 2 code: seg-device, testable segmenter, preflight)

Finished all of Phase 2 except the live checkpoint run. `ANOTMED_SEG_DEVICE`
(CPU-segmenter fallback to free VRAM for vLLM); `MedSAM2Segmenter` takes an
injectable predictor so `segment()` is now tested end-to-end on CPU (+4 tests);
`scripts/check_gpu.sh` preflight; a **real-HTTP** vLLM integration test (stand-up
server, drives the localizer over a socket). Suite 82 → 88. Also ran a full
live-server smoke over real HTTP: upload → PNGs → gate → COCO(RLE) → DICOM-SEG,
all healthy.

## Session log — 2026-07-20 (Phase 5: DICOM-SEG export + COCO-RLE, CPU)

Replaced the DICOM-SEG 501 stub with a real, tested writer and tightened COCO
fidelity. `highdicom` installed and round-trip-verified. Suite 77 → 82.

- **Privacy posture changed (owner-approved):** upload now retains a
  **de-identified** copy of the source DICOM (`io_dicom.deidentify` strips PHI
  *before* write) so the SEG can reference the original series. New invariant:
  *no PHI on disk, but a de-identified source is retained.* SAFETY.md updated.
- **`io_dicom`**: split `read_dataset` + `dataset_to_array_meta` so the API can
  keep the Dataset. **`store`**: `create_study(source_ds=…)` saves
  `<sid>/source.dcm`; `source_dataset()` reads it back.
- **`anotmed/export_seg.py`**: `highdicom.seg.Segmentation` from the source +
  accepted masks — one segment per accepted annotation, generic property codes
  (modality-specific codes await §4), semi-automatic algorithm. Ensures the
  Type-2 attrs highdicom copies from a SecondaryCapture source exist.
- **`api.py`**: `format=dicom-seg` streams `application/dicom`, still 409 while
  pending, still fed **only** by `accepted_annotations` — same gate. Round-trip
  test: accept 1 of 2 → re-read → exactly one segment matching the accepted mask.
- **COCO-RLE**: `io_dicom.mask_to_rle` emits exact uncompressed RLE;
  `annotations_to_coco` uses it instead of the convex-hull approximation.

**Backlog #5 (DICOM-SEG) done.** Remaining owner-gated: live PACS-viewer check.

## Session log — 2026-07-20 (Phase 4: async submit + poll, CPU)

Real inference is slow; made the upload non-blocking without adding any infra
(no Redis/RQ). Suite 71 → 77.

- **`anotmed/jobs.py`**: `Job` + `JobRegistry` — a `queue.Queue` and **exactly one
  worker thread**. That single worker is also the GPU serialization point:
  MedGemma and MedSAM-2 can never run concurrently, so the §2.1 VRAM budget is a
  structural guarantee. Jobs are ephemeral tickets; studies/annotations stay
  durable in the Store. Only real errors mark FAILED (a parse miss = 0 findings).
- **`config.py`**: `ANOTMED_SYNC` — defaults to sync for the stub (fast CI) and
  async for real backends; explicit override respected.
- **`api.py`**: async path returns `202 {job_id, study_id, status}`; new
  `GET /api/jobs/{id}`. Review/export endpoints untouched → **export gate safe by
  construction**. `review.html` upload handles both 200 (inline) and 202 (poll).

Re-proved the safety invariant THROUGH the async path (tests + a **live uvicorn
drive**: 202 → poll → completed → 409 until accept → accepted-only export).

## Session log — 2026-07-20 (Phase 3a: the safety validation gate, CPU)

Built the absolute Dice/IoU validation harness — the project's hard safety
requirement (backlog #4). Runs entirely on the stub + synthetic data (no GPU);
the real-data run is Phase 3b (owner: modality + labeled set). Suite 51 → 71.

- **`eval/metrics.py`** (pure numpy): Dice, IoU, box IoU, greedy localization
  recall/precision, and `valid_mask`. **Format compliance is scored separately
  from quality-given-valid** — an accurate-but-flaky model can't hide behind a
  good mean Dice. Values pinned to hand-computed ground truth.
- **`eval/datasets.py`**: synthetic `Case`s (GT boxes + FWHM-disk masks from the
  same Gaussian lesions as `make_sample_dicom`), so the gate runs today.
  `load_dir()` is the Phase 3b hook.
- **`eval/run.py`**: `evaluate()` scores segmentation on **GT-box prompts**
  (isolates segmenter from localizer) + localization separately; `Floors.check()`
  enforces floors; CLI `--tier absolute` (exit 1 on miss), `--determinism`,
  `--compare` (fp8 vs bf16). Never touches the Store.
- **`eval/floors.yaml`**: suggested clinical floors, **marked owner-must-sign**.
  Loaded via the `eval` extra (`pyyaml`); the fast `tests/` build Floors directly.

Verified: harness runs, determinism OK, and the gate **correctly refuses the
non-clinical stub** (Dice 0.733 < 0.85 → exit 1) while format compliance stays
1.0 — the two axes visibly diverge. When real MedSAM-2 lands (Phase 2), this same
gate scores it with no new code. **Backlog #4 machinery is ready; the signed run
is owner-gated (P3b).**

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

The whole CPU/solo build-out is done (see the session logs and the "Owner-gated
remainder" section near the top). What's left is the owner-gated tail:

1. ~~Run `pytest -q`; confirm the CPU core is green.~~ **DONE 2026-07-19** (now 88 green).
2. ~~Harden the MedGemma/MedSAM-2 adapters; build the vLLM MedGemma backend.~~
   **DONE 2026-07-20** (Phases 0–1). Remaining: the 🟡 live vLLM/GPU run.
3. ~~Make `MedSAM2Segmenter.segment()` testable; seg-device knob; preflight.~~
   **DONE 2026-07-20** (Phase 2 code). Remaining: point at your real checkpoint.
4. **Validate** Dice/IoU on a labeled set from *your* modality — harness is built
   (`eval/`, Phase 3a); this is the real run (P3b). **SAFETY requirement.**
5. ~~**Wire DICOM-SEG export**~~ **DONE 2026-07-20** (Phase 5). Remaining: open the
   exported SEG in a real PACS viewer over the source series.
6. **Confirm target modality**; tune windowing; add 3D if needed (P6, gated).
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

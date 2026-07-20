# anotmed — Staged Implementation Plan

*Design/architecture plan, 2026-07-19. Covers the full backlog, folds in patterns
mined from a sibling project (troke), and is calibrated to a single-radiologist
workstation with the GPU on a Windows host reachable via WSL2. Modality
deliberately left agnostic. See [handover.md](handover.md) for current state,
[README.md](README.md) for run steps, [SAFETY.md](SAFETY.md) for intended-use limits.*

---

## 1. Context

**What anotmed is.** A medical-image **annotation accelerator** — explicitly *not* a
diagnostic device. The flow: MedGemma (VLM) proposes bounding boxes + draft text →
MedSAM-2 converts each box to a pixel mask → a pure-numpy engine computes millimetre
measurements from the mask geometry (never from the LLM) → a radiologist
accepts/edits/rejects every suggestion in a web UI → **only accepted annotations can
be exported**.

**Verified state (this session, first-ever execution, WSL2/CPU):**
- Fresh `.venv`, `pip install -e ".[dev]"`, **pytest 11/11 green** (pydicom 3.0.2,
  numpy 2.5.1, pydantic 2.13.4, fastapi 0.139.2).
- Fixed a first-run reporter wording bug and a pydicom-4.0 deprecation time-bomb.
- Drove the **full radiologist loop over live HTTP** with the stub (CPU) backend:
  upload synthetic DICOM → 2 findings → export blocked `409` while pending → accept
  1 of 2 → export returned exactly the 1 accepted. The safety gate holds end-to-end.
- The two real-model adapters (`backends/medgemma.py`, `backends/medsam.py`) are
  written-to-spec but **have never run**; code review found 6 latent bugs between
  them (detailed in Phase 0).

**Why this plan exists.** The owner asked for "a plan for everything," is open to a
better stack, and wants the useful troke patterns folded in. Backlog item #1 (run the
tests) is done; this plan sequences #2–#7 plus the stack migration.

### The one invariant

> **`Store.accepted_annotations(study_id)` is the sole export path. Only ACCEPTED
> annotations ever leave the system.**

Guarded today by `tests/test_safety_invariant.py`. Every phase below — new backends,
async jobs, DICOM-SEG, 3D, finetuned checkpoints — must route exports through this
gate, and every phase's exit criteria include this test suite staying green plus a
new-path-specific invariant test where the phase adds an export or persistence surface.

---

## 2. Stack Recommendation

### 2.1 The decision: vLLM server for MedGemma, in-process torch for MedSAM-2

**Recommendation: migrate MedGemma serving from in-process `transformers.generate()`
to a standalone vLLM OpenAI-compatible server running in WSL2, with anotmed as a thin
`httpx` client. Keep MedSAM-2 as an in-process torch model inside the anotmed app
process, behind the existing `Segmenter` Protocol.**

**Why vLLM for MedGemma** (troke's biggest lesson — it *deleted* exactly the
in-process path anotmed has now):

| Concern | In-process transformers (current `medgemma.py`) | vLLM server (recommended) |
|---|---|---|
| App startup | Loads ~4–9 GB of weights into the API process | App stays GPU-free; weights live in one long-running server |
| Crash isolation | OOM/torch crash kills the UI the radiologist is using | vLLM restarts independently; app returns a clean error |
| FP8 on 8 GB VRAM | Not supported on-the-fly cleanly | `--quantization fp8` quantizes at load (MedGemma-4B ≈ 4.3 GB vs 8.6 GB bf16); env-switchable to bf16 for accuracy runs |
| Testing without GPU | Must mock deep inside transformers | Mock **one** `httpx.post` — troke proved this gives zero-GPU CI |
| Determinism | Manual `do_sample=False` | `temperature=0`, chat template applied server-side |
| Structured output | Hand-rolled regex/bracket parsing (the source of 4 of the 6 latent bugs) | **Guided JSON decoding** — see below |

**The anotmed-specific divergence from troke (important):** troke deferred vLLM
guided/structured decoding as YAGNI because its output is labeled-line free text.
anotmed's output is **bounding-box JSON** — the exact case structured decoding exists
for. anotmed should **enable vLLM guided-json** with a strict schema
(`[{"box_2d": [int 0–1000] × 4, "label": str, "confidence": number}]`). This
constrains generation itself, making the fragile `_parse_boxes` path (string-unaware
bracket balancing, all-or-nothing coord parsing) a *fallback* rather than the
load-bearing parser. This directly attacks latent bugs #1, #3, #4 at the source.

**Why MedSAM-2 stays in-process (not a second microservice):**

- **Scale:** one radiologist, one request at a time. A segmentation microservice buys
  crash isolation and independent scaling that a single workstation doesn't need; it
  costs a third process to launch, monitor, and version.
- **Existing seam:** `backends/base.py`'s `Segmenter` Protocol already isolates it. If
  VRAM contention or torch instability ever proves painful, extracting `medsam.py`
  into a small FastAPI/torch service later is a mechanical change behind the Protocol —
  the pipeline never notices. The decision is cheap to reverse; take the simple option now.
- **Call pattern:** the pipeline is strictly sequential (localize → segment per box →
  report), so MedGemma and SAM2 never *compute* concurrently even though both are
  resident. Phase 4's single worker thread makes this a hard guarantee.

**Two-model VRAM topology on one consumer GPU** (assume 8 GB RTX 4060-class until the
owner confirms):

| Component | Budget | Notes |
|---|---|---|
| MedGemma-4B FP8 weights (vLLM) | ~4.3 GB | `--quantization fp8` on-the-fly, no pre-quantized checkpoint |
| vLLM KV cache + overhead | ~0.5–1 GB | Single user: `--max-num-seqs 1`, `--max-model-len 2048`, `--enforce-eager`, `--no-enable-prefix-caching`, `--limit-mm-per-prompt '{"image":1}'` |
| **vLLM cap** | **`--gpu-memory-utilization 0.60`** (≈4.8 GB of 8) | ⚠️ **MEASURED WRONG — see correction below.** |
| SAM2/MedSAM-2 fp16 + activations @1024² | ~1.5–2.5 GB | Loaded lazily by the app after vLLM boots |
| Headroom | ~0.7–1.2 GB | Display/WSL overhead |

**Correction (measured 2026-07-20 on the actual RTX 4060):** `--gpu-memory-utilization
0.60` **fails** — vLLM aborts with "No available memory for the cache blocks" because
MedGemma-4b FP8 weights (~4.5 GB) + overhead already exceed 0.60×8 GB before any KV
cache. The real floor is **≥ ~0.70**; MedGemma-only sweet spot is **~0.85** (~7.1 GB
used). Two models therefore do **not** co-fit comfortably (~6.9 GB, tight). Resolution
(the memory-optimization plan): the reliable default on 8 GB is a **warm vLLM server
that never restarts** + **`ANOTMED_SEG_DEVICE=cpu`** (SAM2 on CPU, seconds/image), and
the app caches the SAM2 image embedding so a study encodes once, not per finding. The
faster target is **vLLM sleep/wake** (`SLEEP_MODE=1` in `serve_vllm.sh`): between phases,
`POST /sleep` offloads the weights to RAM (VRAM → ~0) so GPU SAM2 fits, then `POST
/wake_up` — no 2–3 min reload. On 12–16 GB, raise the cap, run SAM2 on GPU, consider
bf16 (still Dice-gated, Phase 3).

**Deployment shape:** everything runs **inside WSL2 with NVIDIA GPU passthrough**
(`docker --gpus all`, exactly as troke does). No cross-OS RPC to native Windows; the
troke "agent-bridge" is a dev chat tool, not a GPU relay — ignored. The only
Windows-side artifact worth keeping in the back pocket: a `netsh portproxy` + firewall
rule **only if** the UI must ever be reached from another LAN machine (fragile — WSL IP
changes per restart — so don't build it until asked).

### 2.2 troke patterns: steal vs. skip

| STEAL | Why it fits anotmed |
|---|---|
| vLLM OpenAI server + thin httpx client | See above; troke's hardest-won lesson |
| FP8 env-switchable quantization + 8 GB tuning flags | Same GPU class; but anotmed **must Dice-gate FP8 vs bf16** (troke deferred that gate; anotmed's safety requirement forbids deferring it) |
| `/health` readiness gate before accepting work | Prevents confusing failures on cold start |
| Mock-the-single-httpx-post for zero-GPU tests | Keeps CI green forever without weights |
| Never-fail-on-parse (`{raw, structured}`, structured=None on miss) | A parse miss must yield "0 findings + raw text in provenance," never a 500 |
| Uniform error envelope `{error, message}` via central handlers | Cheap, makes the UI's error handling trivial |
| pydantic `Literal`-enum schemas for model output | Doubles as the guided-json schema |
| Async submit + poll (202 + job_id, states pending/processing/completed/failed) | 30–90 s inference would time out sync HTTP — but **in-process queue, not Redis** |
| Two-tier test split (fast mocked `tests/` vs slow real-inference `eval/`) | Maps directly onto backlog #4 |
| `finetune/evaluate.py` shape (absolute metrics + format-compliance over held-out set) | The correct template for the safety harness — **not** `eval/run.py`'s regression snapshot (that's the secondary tier) |
| QLoRA config.yaml layout with per-GPU `hardware_overrides` | Only if Phase 3 shows a gap |

| SKIP (deliberately) | Why it's overkill here |
|---|---|
| Redis/RQ broker + stateless worker fan-out | One user, one GPU, one box. A `queue.Queue` + one worker thread gives the same UX with zero infra. |
| sha256 API keys, dual rate limits, per-key quotas, tenant isolation | No tenants. The workstation's OS login *is* the auth boundary. Revisit only at backlog #7 (multi-user), which is explicitly out of scope. |
| Worker replicas / horizontal scaling | Nothing to scale; the GPU is the singleton. |
| agent-bridge | Dev-only AI-agent chat tool; not an architecture component. |
| netsh portproxy by default | Fragile (WSL IP churn); build only on explicit LAN-exposure request. |
| Manual checkpoint promotion (troke's finetune flow) | anotmed automates promotion behind the Dice floor gate (Phase 7). |

---

## 3. Phased Roadmap

**Dependency legend:** 🟢 doable solo now (CPU, no weights) · 🟡 needs GPU/weights ·
🔴 needs owner input (modality, labeled data, VRAM confirmation)

| Phase | Name | Depends on owner? | Unblocks |
|---|---|---|---|
| P0 | Adapter hardening (latent bugs) | 🟢 solo now | P1, P2 |
| P1 | MedGemma via vLLM backend | 🟢 code+tests solo · 🟡 live run | P2 |
| P2 | MedSAM-2 bring-up + GPU topology | 🟡 checkpoint + GPU · 🔴 VRAM size | P3b |
| P3 | **Safety harness (the gate)** | 🟢 scaffold solo (P3a) · 🔴 labeled set + modality (P3b) | Real use. Everything. |
| P4 | Async submit + poll | 🟢 solo now | Pleasant UI at real latencies |
| P5 | DICOM-SEG export | 🟢 solo now | PACS-grade output |
| P6 | Modality tuning + 3D volumes | 🔴 modality choice | Volume workflows |
| P7 | QLoRA finetune + promotion gate | 🔴 only if P3 shows a gap | Closing a measured gap |

Ordering rationale vs. the obvious alternatives: P3 stays ahead of P4 because the
safety gate outranks UX, and the harness drives the backend directly (not over HTTP),
so it doesn't need async. The one refinement worth making: **split P3 into a scaffold
(P3a, buildable today against the stub backend) and the real gated run (P3b)** — this
removes the labeled-data decision from the critical path of writing code, and means the
moment the owner delivers data + weights, the gate runs the same day.

---

### Phase 0 — Adapter hardening: fix the six latent bugs 🟢

**Goal:** make both never-run adapters correct on paper and CPU-testable, so GPU
bring-up debugs *serving*, not *parsing*. None of this needs weights.

**troke pattern applied:** never-fail-on-parse.

| File | Change |
|---|---|
| `anotmed/backends/parsing.py` **(new)** | Extract and fix box parsing into a reusable module (it becomes P1's fallback parser, so this work is not throwaway). Fixes: **(bug 1)** replace raw-char bracket balancing with `json.JSONDecoder().raw_decode` from the first `[` — string-aware, so a label like `"mass [large]"` no longer nukes all detections; **(bug 3)** per-element try/except — one malformed box is skipped with a warning, the rest survive; **(bug 4)** clamp coords to [0, 1000] then to image bounds via the existing `BBox.clamp`; on total parse failure return `[]` and surface the raw text (never raise out of the parser). |
| `anotmed/backends/medgemma.py` | Use `parsing.py`. **(bug 2)** score fix: default `Detection.score = 1.0` when the backend produces no calibrated score, and document that `ANOTMED_MIN_SCORE` only filters backends that emit real scores (the stub does). This ends the failure mode where `min_score > 0` silently vanishes every real finding. |
| `anotmed/backends/medsam.py` | **(bug 5)** move the `ANOTMED_SAM_CHECKPOINT`/`ANOTMED_SAM_CONFIG` guard *before* the lazy torch/sam2 imports → missing sam2 now yields the intended actionable message, not a raw ImportError. **(bug 6)** harden the mask-shape fallback: squeeze channel dims *and* resize model-resolution masks (256/1024) to `(rows, cols)` (nearest-neighbor on the bool mask), then `assert mask.shape == (meta.rows, meta.cols)` before returning — a wrong-shaped mask must never reach `measure.py`. |
| `tests/test_parsing.py` **(new)** | Table-driven parser tests: brackets-in-label, one-bad-box-of-three, out-of-range coords, pure garbage → `[]`, empty response → `[]`. |
| `tests/test_medsam_guards.py` **(new)** | Config-guard-before-import (monkeypatch to simulate missing sam2) and mask-resize fallback (feed fake 256×256 masks). |

**Exit criteria:** all new tests + the existing 11 green on CPU; `test_safety_invariant.py`
untouched and green; parser provably never raises.

---

### Phase 1 — MedGemma behind vLLM (`backends/vllm_medgemma.py`) 🟢 code / 🟡 live

**Goal:** a new backend implementing the existing `Localizer` and `Reporter` Protocols
as a thin async-capable httpx client against a vLLM OpenAI endpoint, with guided-json box
decoding. Zero torch/transformers in the app for the VLM path.

**troke patterns applied:** vLLM OpenAI server, base64 `data:` URL image inlining (MIME
sniffed from magic bytes, no decode/re-encode), FP8 env switch, readiness gate,
mock-one-httpx-post testing, never-fail-parse — **plus the divergence: guided-json enabled.**

| File | Change |
|---|---|
| `anotmed/backends/vllm_medgemma.py` **(new)** | `VllmLocalizer.propose()`: render `meta`-windowed display image (reuse `io_dicom.to_display_rgb`) → PNG → base64 data URL → `POST {ANOTMED_VLLM_URL}/chat/completions` with `temperature=0` and guided-json. Parse structured output; on guided-decode unavailability or refusal, fall back to P0's `parsing.py`; on total miss return `[]` and stash raw text in `Provenance` (never-fail). `VllmReporter.describe()`: same client, finding crop + label context → draft text. Both share one `httpx.Client` with `ANOTMED_VLLM_TIMEOUT_S`. |
| `anotmed/backends/schemas.py` **(new)** | pydantic `Literal`-constrained output model, exported as JSON Schema for guided decoding: `list[{"box_2d": [conint(ge=0, le=1000)]×4, "label": str, "confidence": confloat(ge=0, le=1)}]`, `maxItems = ANOTMED_MAX_FINDINGS`. Confidence is recorded as advisory provenance only — LLM self-reported confidence is uncalibrated and must not gate findings. |
| `anotmed/backends/base.py` | `build_backend`: add backend name `"vllm"` → `VllmLocalizer` + `MedSam2Segmenter` + `VllmReporter`. Add a vLLM `/health` readiness check at backend build with an actionable error ("vLLM not reachable at $ANOTMED_VLLM_URL — run scripts/serve_vllm.sh"). |
| `anotmed/config.py` | New knobs (same plain-`os.getenv` style — no settings lib): `ANOTMED_VLLM_URL` (default `http://127.0.0.1:8000/v1`), `ANOTMED_VLLM_TIMEOUT_S` (default 120), `ANOTMED_GUIDED_JSON` (default on). |
| `scripts/serve_vllm.sh` **(new)** + optional `docker-compose.yml` | Launch `vllm/vllm-openai` in WSL2 with `--gpus all`, `google/medgemma-4b-it`, `--quantization ${QUANTIZATION:-fp8}` (empty = bf16), `--max-model-len 2048`, `--gpu-memory-utilization 0.60` ← **not troke's 0.85, see §2.1**, `--max-num-seqs 1`, `--enforce-eager`, `--no-enable-prefix-caching`, `--limit-mm-per-prompt '{"image":1}'`, `VLLM_USE_FLASHINFER_SAMPLER=0`. |
| `tests/test_vllm_backend.py` **(new)** | Mock the single `httpx` post: well-formed guided JSON → N detections; free-text-with-boxes → fallback parser works; garbage → `[]` + raw in provenance; connection refused → actionable error; assert the outbound request carries `temperature=0`, a data-URL image, and the guided schema. **Runs entirely GPU-free.** |
| `anotmed/api.py` | Add central exception handlers → uniform `{error, message}` envelope (troke pattern; small, do it while touching the API anyway). |

**Deferred deletion:** once P2's live E2E passes, **delete `backends/medgemma.py`** (troke's
lesson: don't keep the dead in-process path). `parsing.py` survives as the fallback parser.

**Exit criteria:** full mocked suite green with no GPU; then (🟡, with owner's GPU)
`ANOTMED_BACKEND=vllm` + stub segmenter produces real MedGemma boxes on a sample image
via the live server; safety-invariant tests green.

---

### Phase 2 — MedSAM-2 bring-up + two-model topology 🟡🔴

**Goal:** the fixed in-process `medsam.py` running on the real checkpoint, coexisting
with vLLM on one GPU under the §2.1 VRAM budget.

| File | Change |
|---|---|
| `anotmed/backends/medsam.py` | Align `_load` to the owner's actual checkpoint/config pairing (checkpoint choice is modality-adjacent → owner decision §4). fp16 inference; honor new `ANOTMED_SEG_DEVICE`. |
| `anotmed/config.py` | `ANOTMED_SEG_DEVICE` (default = `ANOTMED_DEVICE`) — enables the CPU-segmenter fallback if VRAM is tight, and mixed placement generally. |
| `scripts/check_gpu.sh` **(new)** | Preflight: `nvidia-smi` visible in WSL2, VRAM total/free, vLLM `/health`, SAM2 checkpoint present. First thing run on the owner's machine. |
| `tests/test_medsam_guards.py` | Extend with shape-contract tests against a fake predictor (mask == `(rows, cols)` bool, always). |

**Bring-up sequence on the owner's machine:** `check_gpu.sh` → start vLLM (claims its
0.60 cap) → start anotmed (SAM2 lazy-loads into the remainder) → upload sample DICOM →
boxes → masks → measurements → review UI. Watch `nvidia-smi` for the coexistence budget;
if over, drop to `ANOTMED_SEG_DEVICE=cpu` and file the VRAM number into §4.

**Exit criteria:** live E2E on real GPU: upload → MedGemma boxes → MedSAM masks → mm
measurements → accept one → export contains exactly the accepted one. Both models
resident within budget. Then delete `backends/medgemma.py`.

---

### Phase 3 — THE SAFETY GATE: absolute validation harness (`eval/`) 🟢 scaffold / 🔴 gated run

**Goal:** backlog #4, the hard safety requirement. An **absolute** Dice/IoU + localization
scorer with a minimum floor that must pass before any real clinical-adjacent use. This is
the phase that unblocks reality; everything else is plumbing.

**troke patterns applied:** the template is troke's `finetune/evaluate.py` (absolute
metrics + format-compliance over a held-out set), **not** its `eval/run.py` regression
snapshot — because anotmed's requirement is absolute correctness, which troke deliberately
skipped. Build both tiers, rank absolute above regression. Also: two-tier test split,
format-compliance as a first-class metric, determinism check, dependency-light harness.

| File | Purpose |
|---|---|
| `eval/datasets.py` | Loader for the labeled set (owner-supplied, chosen modality): image + GT boxes + GT masks per case; also loads synthetic stub-compatible cases (extend `examples/make_sample_dicom.py`) so the harness runs today. |
| `eval/metrics.py` | Pure-numpy (dependency-light, reuse `measure.py` conventions): mask Dice, mask IoU, box IoU, localization recall/precision at IoU ≥ 0.3 and ≥ 0.5, **format-compliance rate** ("did the backend produce a usable box list / a valid HxW bool mask at all") reported *separately* from quality-given-valid. |
| `eval/run.py` | The gate. Modes: `--tier absolute` (score backend vs GT; `sys.exit(1)` if any floor missed — the real gate), `--tier regression` (compare to `eval/baseline.json` blessed outputs; exit 1 on decision-field drift — the cheap drift alarm), `--compare A B` (two configs, e.g. **fp8 vs bf16** — FP8 is only servable if it matches bf16 within tolerance on Dice/recall; troke deferred this gate, anotmed may not), `--determinism` (same input twice at temperature=0 → identical output; batching-invariance analogue). |
| `eval/floors.yaml` | Floors, versioned in-repo. **Setting the numbers is a clinical judgment the owner must sign** (§4); suggested starting points to negotiate from: GT-box-prompted segmentation mean Dice ≥ 0.85 / p10 ≥ 0.70; localization recall ≥ 0.80 @ IoU 0.3; format compliance ≥ 0.98. |
| `pyproject.toml` | Mark `eval/` excluded from the fast suite; `tests/` stays fast+mocked (tier 1), `eval/` is slow+real (tier 2), run manually or nightly. |

**Sub-phasing:**
- **P3a (🟢 now):** everything above, exercised end-to-end against the **stub backend on
  synthetic data** — proves the harness machinery (metrics, floors, exit codes, compare
  mode) with zero GPU.
- **P3b (🔴 owner):** the real run — needs modality choice, a labeled held-out set (even
  30–50 curated cases is a meaningful floor check), and P1+P2 live.

**Exit criteria:** P3a — harness runs green on stub+synthetic, and deliberately-corrupted
outputs trip the floor (test the gate itself). P3b — real backend scored on real labeled
data; floors either **pass (real use unblocked)** or fail with a measured gap (→ Phase 7).
FP8-vs-bf16 comparison recorded before FP8 is the serving default. Harness never touches
the Store (it drives backend/pipeline directly against ground truth).

---

### Phase 4 — Async submit + poll (in-process, no Redis) 🟢

**Goal:** 30–90 s real inference must not block or time out the browser. troke's async job
*model* with none of its infrastructure.

**troke patterns applied:** 202 + `job_id`, poll `GET /jobs/{id}`, states
`pending/processing/completed/failed`, jobs fail only on real errors (a parse miss is 0
findings, not a failed job). **Skipped:** Redis/RQ, worker fan-out.

| File | Change |
|---|---|
| `anotmed/jobs.py` **(new)** | `Job` dataclass (`id, status, study_id, error, created_at`) + in-memory registry + `queue.Queue` + **exactly one worker thread**. The single worker doubles as the GPU serialization point — MedGemma and SAM2 can never be driven concurrently, which locks in the §2.1 VRAM assumption as a hard guarantee. Jobs are ephemeral progress tickets; studies/annotations persist in `Store`, so losing the registry on restart loses nothing durable. |
| `anotmed/api.py` | `POST /api/studies` → save upload, enqueue, return `202 {job_id}` (config `ANOTMED_SYNC=1` keeps the current synchronous path for tests and the stub). New `GET /api/jobs/{job_id}` with the troke state machine and the `{error, message}` envelope on failure. Review/export endpoints unchanged — annotations still land as PENDING via the same `Store` writes, so **the export gate is untouched by construction**. |
| `tests/test_jobs.py` **(new)** | With the stub backend: submit → poll to `completed` → annotations present and PENDING → export still 409 until accept → accepted-only export. This re-proves the invariant *through the async path*. |
| Web UI | Swap the upload call to submit+poll with a progress state. |

**Exit criteria:** live-server drive of the async loop with the stub (mirroring this
session's verification); invariant test through the async path green; sync mode still
available for CI.

---

### Phase 5 — DICOM-SEG export via highdicom 🟢

**Goal:** replace the documented 501 stub with a real DICOM-SEG writer. `Study.source_path`
is already retained for exactly this.

| File | Change |
|---|---|
| `anotmed/export_seg.py` **(new)** | `highdicom.seg.Segmentation` built from the source dataset at `Study.source_path`: one segment per annotation, `BINARY` type, generic `SegmentDescription` category/type codes until the modality decision allows specific codes; algorithm identification = anotmed + backend provenance; de-identification consistent with `io_dicom`. **Input is exclusively `Store.accepted_annotations(study_id)`** — the new format enters through the same gate. |
| `anotmed/api.py` | Wire `format=dicom-seg` to the writer; keep the 409-while-pending behavior identical to COCO. Add `pyproject.toml` dep `highdicom`. |
| `tests/test_export_seg.py` **(new)** | Synthetic DICOM (existing `examples/make_sample_dicom.py`) → accept 1 of 2 → DICOM-SEG export → **re-read with pydicom/highdicom and assert exactly one segment whose pixels match the accepted mask**; assert PENDING/REJECTED absent; assert the 409-pending case. |

Also worth 30 minutes here: fix the known COCO limitation (segmentation = convex-hull
approximation) by emitting RLE from the true mask in `io_dicom.annotations_to_coco` — the
mask is already on disk.

**Exit criteria:** round-trip test green; live-server drive: accept → export → open the SEG
in a DICOM viewer over the source series. COCO remains the default export until the owner
designates the canonical format (§4).

---

### Phase 6 — Modality tuning + 3D volumes 🔴 modality / 🟢 3D scaffolding

**Goal:** backlog #6. Everything modality-generic is built now; modality-specific defaults
land as data/config, not architecture.

| File | Change |
|---|---|
| `anotmed/io_dicom.py` | Per-modality default window presets (CT: standard organ presets; MR/X-ray: percentile/VOI-LUT-driven) selected off the existing `ImageMeta.modality`; keep `spacing_is_default` honesty flag. |
| `anotmed/pipeline.py` | 3D: accept a multi-slice study and loop the existing single-slice pipeline per slice, stamping the already-plumbed `slice_index` on each annotation. Deliberate scope cut: **no cross-slice mask propagation in this phase** — MedSAM-2's video-style propagation is a powerful follow-up, but per-slice box→mask is a complete, reviewable, exportable increment that doesn't gamble the review UX on tracking quality. Revisit propagation only after P3b establishes per-slice quality. |
| `anotmed/api.py` + UI | Multi-file/multi-frame upload; slice navigation; review remains per-annotation (statuses are already independent). |
| `anotmed/export_seg.py` | Multi-slice SEG assembly (highdicom handles multi-frame natively). |
| `examples/make_sample_dicom.py` | Synthetic multi-slice series generator → 3D tests run on CPU with the stub today. |
| `eval/` | Extend datasets/metrics with per-slice → per-volume aggregation once labeled 3D data exists. |

**Exit criteria:** synthetic volume through stub: N slices → per-slice annotations with
correct `slice_index` → accept a subset across slices → COCO and DICOM-SEG exports contain
exactly that subset (invariant, per-slice). Modality presets land only after the owner's
choice.

---

### Phase 7 — QLoRA finetuning with an automated promotion gate 🔴 conditional

**Goal:** **only if** P3b shows the base MedGemma misses the localization floor. Do not
start this phase on vibes; start it on a failing `eval/run.py` report.

**troke patterns applied:** QLoRA recipe (4-bit NF4 base + LoRA via peft + trl
`SFTTrainer`, collator masks the user turn, `load_best_model_at_end` by eval_loss,
`config.yaml` with lora/quantization/training blocks + `hardware_overrides` for
rtx4060/t4/a100). **Divergence:** troke's promotion is manual; anotmed's is automated.

| File | Change |
|---|---|
| `finetune/config.yaml` **(new)** | troke-shaped config; training data = the radiologist's own **accepted** annotations (read via `Store.accepted_annotations` — the invariant means the finetune corpus is human-approved by construction, a genuinely nice property) + any public labeled set for the chosen modality, converted to box_2d chat format. Strict split from the P3 held-out set. |
| `finetune/train.py`, `finetune/prepare_data.py` **(new)** | QLoRA run per troke recipe. |
| `finetune/promote.py` **(new)** | **The automated gate:** a candidate checkpoint is servable only if `eval/run.py --tier absolute` passes all floors **and** it matches-or-beats the incumbent on the held-out set. On pass: merge/emit weights, print the vLLM serve line pointing at them. On fail: refuse, with the metric diff. |

**Exit criteria:** promotion gate demonstrated in both directions (a bad checkpoint
refused, a good one promoted); served finetuned weights re-pass the full P3 gate via vLLM.

---

## 4. Decisions Needed From the Owner

| # | Decision | Why it blocks | Recommendation to react to |
|---|---|---|---|
| 1 | **Modality** (CT / MR / X-ray) | P3b labeled set, P6 windowing, MedSAM checkpoint choice, SEG codes | Pick the modality with the most day-to-day annotation volume; everything upstream is agnostic by design |
| 2 | **Labeled validation set** (source, size, licensing) + who signs the `eval/floors.yaml` numbers | P3b is *the* gate; floors are a clinical judgment | 30–50 curated held-out cases minimum to start; owner signs floors |
| 3 | **MedSAM-2 checkpoint + config pairing** | P2 `_load` alignment | Follows from #1 |
| 4 | **GPU VRAM + WSL2 passthrough confirmed?** (run `nvidia-smi` inside WSL2) | §2.1 budget, fp8-vs-bf16, `ANOTMED_SEG_DEVICE` | 8 GB → FP8 + 0.60 cap; 12 GB → roomier; 16 GB+ → bf16 viable (still Dice-compared) |
| 5 | **bf16 vs FP8 as serving default** | P1 serve script default | FP8 on 8 GB — but only after it passes the P3 `--compare` gate vs bf16 |
| 6 | **Canonical export: COCO or DICOM-SEG?** (feeds a PACS/viewer, or an ML pipeline?) | P5 default + UI emphasis | Both stay supported; PACS-bound → DICOM-SEG canonical |
| 7 | **Docker-in-WSL2 vs bare-metal pip vLLM** | P1 serve script shape | Docker (`vllm/vllm-openai`) — matches troke, pins the CUDA stack |
| 8 | **Is 3D/volume work actually needed soon?** | Whether P6 is scheduled or parked | Park until #1 is answered |
| 9 | **Any LAN exposure ever?** (second machine viewing the UI) | Whether the netsh portproxy artifact gets built | Default no; workstation-local only |

---

## 5. Verification Strategy

The template is exactly how this session verified the CPU core: **fast unit tests + a real
end-to-end drive against the live server**, with the safety invariant asserted on every new path.

| Layer | What | When | GPU? |
|---|---|---|---|
| Tier-1 unit/integration (`tests/`) | Existing 11 + P0 parser/guard tests + P1 mocked-httpx backend tests + P4 async-path invariant + P5 SEG round-trip. The vLLM backend is tested by mocking the **single** `httpx.post`. | Every change; CI | **Never** |
| Live E2E drive | The session's proven loop — upload → findings → 409-while-pending → accept subset → export exactly that subset — re-driven at each phase boundary: stub/sync (done ✅), vllm+stub-seg (P1), full real models (P2), async (P4), DICOM-SEG (P5), multi-slice (P6) | Phase exits | P2+ only |
| Tier-2 absolute gate (`eval/run.py --tier absolute`) | Dice/IoU/localization/format-compliance vs floors; **exit 1 = not fit for real use.** Drives backend directly, no HTTP, no Store. | Before any real use; after any model/prompt/quantization/checkpoint change | Yes |
| Precision gate (`--compare`) | FP8 vs bf16 within tolerance before FP8 is default | Once per weights change | Yes |
| Determinism check (`--determinism`) | temperature=0, identical input → identical output | With tier-2 | Yes |
| Tier-2 regression (`--tier regression`) | Blessed-output snapshot diff, exit 1 on decision drift | Nightly / pre-upgrade | Yes |
| Invariant, permanently | `tests/test_safety_invariant.py` + one invariant test **per new surface**: async job path (P4), DICOM-SEG (P5), per-slice (P6), finetune corpus sourcing (P7). Every export format and every backend routes through `Store.accepted_annotations` — no exceptions, ever. | Every change | Never |

**Immediate next actions:** P0 today (pure CPU, no owner input); P1 code + mocked tests and
P3a harness scaffold in parallel (both solo); hand the owner §4 — items 1–4 are the critical
path to the only thing that matters: running the safety gate on real models and real labeled data.

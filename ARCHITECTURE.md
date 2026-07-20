# anotmed — Architecture & Flow

Visual companion to [PLAN.md](PLAN.md) (detailed roadmap), [README.md](README.md)
(run steps), [SAFETY.md](SAFETY.md) (intended-use limits), and
[handover.md](handover.md) (session log). Diagrams are Mermaid — they render on
GitHub.

anotmed is a medical-image **annotation accelerator**, not a diagnostic device.
It proposes boxes, masks, and millimetre measurements; a radiologist accepts,
edits, or rejects every suggestion; **nothing leaves the system until a human
accepts it.**

---

## 1. The core flow

```mermaid
flowchart LR
    U["DICOM upload"] --> DZ["de-identify<br/>+ persist source"]
    DZ --> L["MedGemma<br/>localize + draft report"]
    L --> S["MedSAM-2<br/>box → pixel mask"]
    S --> M["measure.py<br/>mm measurements<br/>(geometry, not the LLM)"]
    M --> R{"Review UI<br/>radiologist"}
    R -->|accept| EX["Export<br/>COCO · DICOM-SEG"]
    R -->|"edit, re-segment"| S
    R -->|reject| DROP["stays in system"]
    R -->|pending| DROP

    style EX fill:#1f7a3d,color:#fff
    style R fill:#8a5a00,color:#fff
    style DROP fill:#5a1f1f,color:#fff
```

**Two rules encoded in code, not convention:**
- Measurements come from the **geometry engine** (`measure.py`) computed from the
  mask — never from the language model. The reporter writes prose; the numbers
  are reproducible and auditable.
- Only `ReviewStatus.ACCEPTED` annotations can be exported, exclusively via
  `Store.accepted_annotations`. See §4.

---

## 2. Deployment topology (why the app has no torch on the VLM path)

MedGemma runs in a **separate vLLM server process**; the anotmed app is a thin
HTTP client. MedSAM-2 can't be served by vLLM, so it runs **in-process** (torch).

```mermaid
flowchart TB
    subgraph APP["anotmed app process (no torch for the VLM path)"]
        API["FastAPI + Review UI"]
        JOBS["jobs.py<br/>1 worker = GPU serializer"]
        PIPE["pipeline.py"]
        VC["VllmLocalizer / VllmReporter<br/>thin httpx client"]
        SEG["MedSAM-2 segmenter<br/>in-process torch"]
        API --> JOBS --> PIPE
        PIPE --> VC
        PIPE --> SEG
    end
    subgraph SRV["vLLM server process"]
        VLLM["MedGemma-4b · FP8<br/>guided JSON decoding"]
    end
    VC -->|"HTTP /v1/chat/completions"| VLLM

    style SRV fill:#12324a,color:#fff
    style APP fill:#1a1a1a,color:#fff
```

Payoff: the entire backend is **unit-tested on CPU by mocking one HTTP call**
(88 tests, no GPU). GPU only enters at live-run time.

---

## 3. Two models, one 8 GiB GPU — sequential execution

On an RTX 4060 (8 GiB) MedGemma (FP8 ≈ 4.5–7 GiB) and MedSAM-2 (≈ 1 GiB) are a
tight fit together. `scripts/run_sequential.py` loads them **one at a time**: all
MedGemma work first (the reporter crops by box, so it never needs the mask), then
vLLM is fully stopped to free VRAM, then SAM2 runs.

```mermaid
sequenceDiagram
    autonumber
    participant O as run_sequential.py
    participant V as vLLM (MedGemma)
    participant G as GPU 8 GiB
    participant S as SAM2 (in-process)

    O->>V: start server (isolated child)
    V->>G: load weights (~7 GiB)
    O->>V: localize + draft report — every finding
    V-->>O: boxes + prose
    Note over O,V: Phase A done
    O->>V: STOP (killpg own group +<br/>only nvidia-smi GPU PIDs)
    V--xG: VRAM released (→ ~0 MiB)
    O->>S: load SAM2 (~1 GiB)
    O->>S: segment each saved box
    S-->>O: masks → measure.py → mm
    O->>O: assemble PENDING annotations
    Note over O,S: Phase B done — safety gate holds
```

**Teardown safety:** vLLM is a child launched with `start_new_session=True`; it is
killed by (a) its own process group and (b) *only* PIDs `nvidia-smi` reports as
holding the GPU. Those are model workers — never the shell or `/init`.

Alternative when you don't want to stop/start vLLM per study: keep vLLM at a
lower util and run the segmenter on CPU via `ANOTMED_SEG_DEVICE=cpu`.

---

## 4. The one invariant — human-in-the-loop, enforced in code

```mermaid
stateDiagram-v2
    [*] --> PENDING: pipeline proposes
    PENDING --> ACCEPTED: radiologist accepts
    PENDING --> REJECTED: radiologist rejects
    PENDING --> PENDING: edit box → re-segment
    ACCEPTED --> EXPORT: Store.accepted_annotations
    REJECTED --> [*]
    EXPORT --> [*]
    note right of EXPORT
        The ONLY export path.
        PENDING / REJECTED can
        never reach it.
    end note
```

Guarded by `tests/test_safety_invariant.py` and `tests/test_api.py`, and re-proven
through the async path and the live server.

---

## 5. Phase roadmap — status

```mermaid
flowchart TD
    P0["P0 · Adapter hardening<br/>✅"] --> P1["P1 · MedGemma via vLLM<br/>✅ LIVE"]
    P1 --> P2["P2 · MedSAM-2<br/>✅ LIVE"]
    P0 --> P3a["P3a · Safety gate<br/>✅ refuses stub"]
    P1 --> P4["P4 · Async submit+poll<br/>✅"]
    P1 --> P5["P5 · DICOM-SEG + RLE<br/>✅"]
    P2 --> SEQ["Sequential runner<br/>✅"]
    P3a --> P3b["P3b · Gate on REAL data<br/>✅ Kvasir / CT / DENTEX"]
    P3b --> DET["Detector-as-Localizer<br/>✅ dental recall 0.994"]
    DET --> MM["Multi-modality profiles<br/>✅ registry + per-mod floors"]
    MM --> MOD["More modalities<br/>🔴 chest X-ray next"]
    DET --> PATH["Dental pathology<br/>🔴 milestone-2"]

    classDef done fill:#1f7a3d,color:#fff;
    classDef gated fill:#5a1f1f,color:#fff;
    class P0,P1,P2,P3a,P4,P5,SEQ,P3b,DET,MM done;
    class MOD,PATH gated;
```

| Phase | State | Note |
|---|---|---|
| P0–P5, sequential runner | ✅ | hardening, vLLM MedGemma (live), MedSAM-2 (live), gate, async, DICOM-SEG/RLE |
| P3b Gate on **real data** | ✅ | ran on Kvasir (polyps), MSD Lung CT, DENTEX — honest measured misses drove the pivot |
| **Detector-as-Localizer** | ✅ | MedGemma localized poorly (0.0–0.37) → a YOLOv8n detector: **recall 0.994** on dental |
| **Multi-modality profiles** | ✅ | `modalities.py` registry; per-modality window/labels/`max_findings`/floors |
| Dental pathology (milestone-2) | 🔴 | 4-class caries/lesion detector — the clinical findings |
| More modalities | 🔴 | VinDr-CXR chest X-ray next (box-only path reused) |

**132 tests green.** Full detect→segment→measure→report→gate→export loop verified live on
real dental. The **finding of the whole exercise**: the localizer was the bottleneck, and a
dedicated detector — not a bigger VLM or a medical checkpoint — is what fixes it.

---

## 6. Memory budget (measured on RTX 4060, 8 GiB)

```mermaid
flowchart LR
    subgraph G["RTX 4060 · 8 GiB VRAM"]
        direction TB
        A["MedGemma-4b FP8<br/>needs util >= ~0.70<br/>0.60 fails: no KV cache room<br/>~4.5 to 7.1 GiB"]
        B["MedSAM-2 / SAM2<br/>~1 GiB, in-process"]
    end
    A -.->|"co-fit ~6.9 GiB: tight, not forced - run sequentially"| B
```

- **FP8, not bf16** — bf16 4B (~8.5 GiB) won't fit; FP8 (~4.5 GiB) does.
- MedGemma-only sweet spot: `--gpu-memory-utilization 0.85` (~7.1 GiB).
- Weights are cached + **gated**: run vLLM with `HF_HUB_OFFLINE=1` and pass the
  local `chat_template.jinja` (the repo ships `.jinja`, not the `chat_template.json`
  vLLM looks for). `scripts/serve_vllm.sh` handles all of this automatically.

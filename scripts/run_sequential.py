#!/usr/bin/env python
"""Run the full anotmed pipeline SEQUENTIALLY on a memory-constrained GPU.

MedGemma (vLLM) and MedSAM-2 don't fit together on a tight GPU (e.g. 8 GiB), so
this loads them one at a time:

  Phase A — start vLLM, run ALL MedGemma work (localize every finding AND write
            its draft report; the reporter crops by box, it never needs the
            mask), then FULLY STOP vLLM to free VRAM.
  Phase B — load SAM2 for the masks + measurements, assemble the annotations,
            persist them PENDING. Nothing exports without human acceptance.

Slower than co-resident, but the only way when the two models don't fit together.

SAFETY: vLLM is launched as an isolated child (start_new_session=True) and torn
down by killing (a) that child's own process group and (b) only PIDs that
`nvidia-smi` reports as holding the GPU. It never kills a parent, `/init`, or the
caller's shell — the GPU-holder list cannot contain them.

Run it with an interpreter that has torch + sam2 + httpx (e.g. the vLLM venv);
the script adds the repo root to sys.path so anotmed need not be installed there.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from anotmed.config import Config  # noqa: E402
from anotmed.io_dicom import annotations_to_coco, dataset_to_array_meta, read_dataset  # noqa: E402
from anotmed.measure import measure_box, measure_mask  # noqa: E402
from anotmed.schema import Annotation, ImageMeta, Provenance, ReviewStatus  # noqa: E402
from anotmed.store import Store  # noqa: E402


def _gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else 0


def _gpu_pids() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _looks_like_vllm(pid: int) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").lower()
    except OSError:
        return False
    return b"vllm" in cmd or b"enginecore" in cmd


def start_vllm(cfg: Config, extra_args: list[str]) -> subprocess.Popen:
    """Launch scripts/serve_vllm.sh as an isolated child process group."""
    env = dict(os.environ, GPU_MEM_UTIL=os.environ.get("GPU_MEM_UTIL", "0.85"),
               MODEL=cfg.medgemma_model)
    proc = subprocess.Popen(
        ["bash", str(REPO / "scripts" / "serve_vllm.sh"), *extra_args],
        env=env, start_new_session=True,  # its own session -> safe to killpg
    )
    return proc


def wait_health(url: str, timeout: float = 300.0) -> None:
    import httpx

    root = url.rstrip("/")
    health = (root[:-3] if root.endswith("/v1") else root) + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(health, timeout=3).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"vLLM did not become healthy at {health} within {timeout}s")


def stop_vllm(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Stop vLLM and wait for VRAM to be released — WITHOUT touching any parent.

    Kills only: our own child's process group, and PIDs nvidia-smi reports as
    holding the GPU that look like vLLM. Both are guaranteed not to be the shell
    or /init (those never appear as GPU compute apps and are not our child group).
    """
    # (a) our child's own group
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    # (b) any GPU-holding vLLM worker that detached from our group (EngineCore)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        holders = [p for p in _gpu_pids() if p != os.getpid() and _looks_like_vllm(p)]
        for p in holders:
            try:
                os.kill(p, signal.SIGKILL)  # a specific GPU-holding vLLM PID only
            except ProcessLookupError:
                pass
        if _gpu_used_mib() < 400:
            return
        time.sleep(2)
    print(f"warning: GPU still shows {_gpu_used_mib()} MiB used after stopping vLLM")


def load_image(source: str | None) -> tuple[np.ndarray, ImageMeta]:
    if source:
        ds = read_dataset(source)
        return dataset_to_array_meta(ds)
    from examples.make_sample_dicom import synth_image
    arr = synth_image(256).astype(np.float32)
    meta = ImageMeta(study_id="", rows=256, cols=256, pixel_spacing_mm=(0.7, 0.7),
                     modality="CT", window_center=400, window_width=1200)
    return arr, meta


def phase_a(cfg: Config, arr: np.ndarray, meta: ImageMeta) -> list[dict]:
    from anotmed.backends.vllm_medgemma import VllmLocalizer, VllmReporter, _VllmClient

    client = _VllmClient(cfg)
    client.health()
    loc, rep = VllmLocalizer(client, cfg), VllmReporter(client, cfg)
    dets = loc.propose(arr, meta)
    findings = []
    for d in dets:
        prose = rep.describe(arr, d, np.zeros((meta.rows, meta.cols), bool), meta)
        findings.append({"box": d.box, "label": d.label, "score": d.score, "prose": prose})
    print(f"Phase A (MedGemma): {len(findings)} finding(s) localized + reported")
    return findings


def phase_b(cfg: Config, arr: np.ndarray, meta: ImageMeta, findings: list[dict],
            store: Store, study_id: str) -> list[Annotation]:
    from anotmed.backends.medsam import MedSAM2Segmenter

    seg = MedSAM2Segmenter(cfg)
    anns: list[Annotation] = []
    for f in findings:
        aid = uuid.uuid4().hex[:12]
        mask = seg.segment(arr, f["box"], meta)
        if mask.any():
            meas = measure_mask(mask, meta.pixel_spacing_mm)
            mask_path = store.save_mask(study_id, aid, mask)
        else:
            meas = measure_box(f["box"], meta.pixel_spacing_mm)
            mask_path = None
        summary = f"Measured: max diameter {meas.max_diameter_mm:.1f} mm, area {meas.area_cm2:.2f} cm²."
        ann = Annotation(
            id=aid, study_id=study_id, label=f["label"], box=f["box"], mask_path=mask_path,
            measurement=meas, report_text=f"{f['prose']}\n{summary}", status=ReviewStatus.PENDING,
            provenance=Provenance(backend="sequential(vllm+medsam)", localizer=cfg.medgemma_model,
                                  segmenter="medsam2/sam2", reporter=cfg.medgemma_model, score=f["score"]),
        )
        store.upsert_annotation(ann)
        anns.append(ann)
    print(f"Phase B (SAM2): {len(anns)} finding(s) segmented + measured")
    return anns


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="anotmed sequential (one-model-at-a-time) pipeline")
    p.add_argument("--dicom", help="source DICOM (default: synthetic phantom)")
    p.add_argument("--storage", default=str(REPO / "anotmed_data"))
    p.add_argument("--sam-checkpoint", default=os.environ.get("ANOTMED_SAM_CHECKPOINT", ""),
                   help="MedSAM-2/SAM2 checkpoint (.pt); or set ANOTMED_SAM_CHECKPOINT")
    p.add_argument("--sam-config", default=os.environ.get(
        "ANOTMED_SAM_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"),
                   help="SAM2 hydra config name")
    p.add_argument("--no-serve", action="store_true",
                   help="assume vLLM is already running (don't start/stop it)")
    args = p.parse_args(argv)

    if not args.sam_checkpoint:
        p.error("Phase B needs a segmenter checkpoint: pass --sam-checkpoint "
                "or set ANOTMED_SAM_CHECKPOINT")

    cfg = Config(storage_dir=Path(args.storage),
                 sam_checkpoint=args.sam_checkpoint, sam_config=args.sam_config)
    store = Store(cfg.storage_dir)
    arr, meta = load_image(args.dicom)

    proc = None
    if not args.no_serve:
        print("starting vLLM (Phase A)...")
        proc = start_vllm(cfg, [])
        wait_health(cfg.vllm_url)
    try:
        findings = phase_a(cfg, arr, meta)
    finally:
        if proc is not None:
            print("stopping vLLM to free VRAM before SAM2...")
            stop_vllm(proc)
            print(f"  GPU now: {_gpu_used_mib()} MiB used")

    study = store.create_study(arr, meta, Path(args.dicom).name if args.dicom else "synth.dcm")
    anns = phase_b(cfg, arr, meta, findings, store, study.id)

    print(f"\nstudy {study.id}: {len(anns)} PENDING annotation(s). Nothing exports until accepted.")
    print(f"accepted so far: {len(store.accepted_annotations(study.id))}")
    if anns:
        for a in anns:
            m = a.measurement
            print(f"  - {a.label!r} box=({a.box.x:.0f},{a.box.y:.0f},{a.box.w:.0f},{a.box.h:.0f}) "
                  f"max_diam={m.max_diameter_mm:.1f}mm area={m.area_cm2:.2f}cm²")
    return 0


if __name__ == "__main__":
    sys.exit(main())

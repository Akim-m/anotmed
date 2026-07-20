"""In-process async jobs — troke's job *model* with none of its infrastructure.

Real inference takes tens of seconds and must not block the browser. A study
upload persists immediately (to the Store) and returns a job ticket; the pipeline
runs on a background worker and the client polls for completion.

**Exactly one worker thread.** That single thread is also the GPU serialization
point: MedGemma and MedSAM-2 can never be driven concurrently, which makes the
two-model VRAM budget (PLAN.md §2.1) a structural guarantee rather than a hope.

Jobs are ephemeral progress tickets held in memory. Studies and annotations are
durable in the Store, so losing the registry on restart loses nothing that
matters — a lost job just means "re-upload", never "lost annotation".
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger("anotmed.jobs")

PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    study_id: str
    status: str = PENDING
    error: str = ""
    created_at: datetime = field(default_factory=_now)


class JobRegistry:
    """A queue + one worker thread + an in-memory {id: Job} map."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[tuple[str, Callable[[], object]]]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, study_id: str, fn: Callable[[], object]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], study_id=study_id)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job.id, fn))
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name="anotmed-worker", daemon=True)
                self._worker.start()

    def _set(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                for k, v in changes.items():
                    setattr(job, k, v)

    def _run(self) -> None:
        while True:
            job_id, fn = self._queue.get()
            self._set(job_id, status=PROCESSING)
            try:
                fn()
                self._set(job_id, status=COMPLETED)
            except Exception as e:  # only a REAL error fails the job
                log.exception("job %s failed", job_id)
                self._set(job_id, status=FAILED, error=str(e))
            finally:
                self._queue.task_done()

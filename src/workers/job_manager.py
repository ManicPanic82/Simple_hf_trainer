"""GPU-exclusive job orchestration foundation.

Unsloth and Heretic are heavyweight engines that must never hold GPU memory at
the same time. This manager owns that rule: a job may only start if it acquires
the single GPU lock, and every terminal path (success, failure, cancellation,
worker crash) releases it. The rule lives here, not in the GUI.

A second submission while the GPU is busy is no longer rejected. It is queued
(FIFO) in :class:`JobState.QUEUED` and started automatically as soon as the
current owner reaches a terminal state. The owning tab is told to build/launch
the worker through the :attr:`JobManager.job_start_requested` signal; the
payload's ``state`` distinguishes a newly queued job from one that was just
promoted to :class:`JobState.STARTING`.

Terminal snapshots are also persisted to a small ``jobs.json`` file so a new
manager can expose them through :meth:`JobManager.history`. Persistence is
best-effort: a corrupt or unwritable file never breaks orchestration.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from app import paths

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_LIMIT = 200


class JobState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    QUEUED = "queued"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)

    @property
    def is_active(self) -> bool:
        return not self.is_terminal


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    engine: str
    model_ref: str
    state: JobState
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    message: str = ""

    @property
    def is_active(self) -> bool:
        return self.state.is_active

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record (used by the history store)."""
        return {
            "job_id": self.job_id,
            "engine": self.engine,
            "model_ref": self.model_ref,
            "state": str(self.state),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, record: object) -> JobSnapshot | None:
        """Rebuild a terminal snapshot, or ``None`` if the record is unusable."""
        if not isinstance(record, dict):
            return None
        job_id = record.get("job_id")
        engine = record.get("engine")
        if not isinstance(job_id, str) or not job_id:
            return None
        if not isinstance(engine, str) or not engine:
            return None
        try:
            state = JobState(record.get("state"))
        except (TypeError, ValueError):
            return None
        if not state.is_terminal:
            return None
        created_at = _as_number(record.get("created_at"))
        if created_at is None:
            return None
        model_ref = record.get("model_ref", "")
        message = record.get("message", "")
        exit_code = record.get("exit_code")
        return cls(
            job_id=job_id,
            engine=engine,
            model_ref=model_ref if isinstance(model_ref, str) else "",
            state=state,
            created_at=created_at,
            started_at=_as_number(record.get("started_at")),
            finished_at=_as_number(record.get("finished_at")),
            exit_code=exit_code
            if isinstance(exit_code, int) and not isinstance(exit_code, bool)
            else None,
            message=message if isinstance(message, str) else "",
        )


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class GpuBusyError(RuntimeError):
    """Legacy error kept for API compatibility; ``submit`` now queues instead."""

    def __init__(self, owner_job_id: str, owner_engine: str, requested_engine: str) -> None:
        super().__init__(
            f"The GPU is in use by {owner_engine} job {owner_job_id}; "
            f"cannot start {requested_engine}. Unsloth and Heretic never share the GPU."
        )
        self.owner_job_id = owner_job_id
        self.owner_engine = owner_engine
        self.requested_engine = requested_engine


class GpuLock:
    """A single-owner lock representing exclusive GPU access."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._owner: str | None = None
        self._acquired_at: float | None = None

    def try_acquire(self, owner: str) -> bool:
        with self._mutex:
            if self._owner is not None:
                return False
            self._owner = owner
            self._acquired_at = time.monotonic()
            return True

    def release(self, owner: str) -> bool:
        """Release only if ``owner`` holds the lock. Returns whether it did."""
        with self._mutex:
            if self._owner != owner:
                return False
            self._owner = None
            self._acquired_at = None
            return True

    @property
    def owner(self) -> str | None:
        with self._mutex:
            return self._owner

    @property
    def is_locked(self) -> bool:
        return self.owner is not None


class JobManager(QObject):
    """Tracks engine jobs, queues overflow, and enforces exclusive GPU ownership."""

    job_changed = Signal(object)  # JobSnapshot
    gpu_ownership_changed = Signal(object)  # str | None
    job_start_requested = Signal(object)  # JobSnapshot - QUEUED or STARTING

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        history_file: Path | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._lock = GpuLock()
        self._mutex = threading.RLock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._ids = count(1)
        self._queue: deque[str] = deque()
        self._history_limit = max(1, history_limit)
        self._history_file = (
            Path(history_file)
            if history_file is not None
            else paths.settings_path().parent / "jobs.json"
        )
        self._history: list[JobSnapshot] = self._load_history()
        logger.info(
            "JobManager initialized; GPU lock free, %d persisted terminal record(s).",
            len(self._history),
        )

    # -- queries ---------------------------------------------------------

    @property
    def gpu_owner(self) -> str | None:
        return self._lock.owner

    @property
    def history_file(self) -> Path:
        return self._history_file

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        with self._mutex:
            return self._jobs.get(job_id)

    def jobs(self) -> list[JobSnapshot]:
        with self._mutex:
            return sorted(self._jobs.values(), key=lambda job: job.created_at)

    def active_job(self) -> JobSnapshot | None:
        with self._mutex:
            owner = self._lock.owner
            if owner is not None:
                current = self._jobs.get(owner)
                if current is not None and current.is_active:
                    return current
            for job in self._jobs.values():
                if job.is_active:
                    return job
        return None

    def can_accept(self) -> bool:
        return not self._lock.is_locked

    def queued_jobs(self) -> list[JobSnapshot]:
        """FIFO view of jobs waiting for the GPU."""
        with self._mutex:
            jobs = []
            for job_id in self._queue:
                job = self._jobs.get(job_id)
                if job is not None and job.state is JobState.QUEUED:
                    jobs.append(job)
            return jobs

    def history(self) -> list[JobSnapshot]:
        """Persisted terminal snapshots, oldest first (bounded, most recent last)."""
        with self._mutex:
            return list(self._history)

    # -- lifecycle -------------------------------------------------------

    def submit(self, engine: str, model_ref: str = "") -> JobSnapshot:
        """Start tracking a new job, queueing it if the GPU is already busy."""
        engine = engine.strip().lower()
        if not engine:
            raise ValueError("engine must be a non-empty name")

        with self._mutex:
            job_id = f"{engine}-{next(self._ids)}"
            candidate = JobSnapshot(
                job_id=job_id,
                engine=engine,
                model_ref=model_ref,
                state=JobState.STARTING,
                created_at=self._clock(),
            )
            if self._lock.try_acquire(job_id):
                job = candidate
                self._jobs[job_id] = job
                self._emit_gpu_owner()
                queued = False
            else:
                job = replace(candidate, state=JobState.QUEUED)
                self._jobs[job_id] = job
                self._queue.append(job_id)
                queued = True
                owner_id = self._lock.owner or "another"
                logger.info("GPU busy (%s); queued job %s.", owner_id, job_id)

        self.job_changed.emit(job)
        if queued:
            self.job_start_requested.emit(job)
        return job

    def mark_running(self, job_id: str, *, pid: int | None = None) -> JobSnapshot | None:
        logger.debug("Marking job %s as RUNNING (pid=%s).", job_id, pid)

        def transition(job: JobSnapshot) -> JobSnapshot:
            if job.state is not JobState.STARTING:
                return job
            message = f"pid {pid}" if pid is not None else job.message
            return replace(job, state=JobState.RUNNING, started_at=self._clock(), message=message)

        result = self._transition(job_id, transition)
        if result:
            logger.info("Job %s state changed to RUNNING.", job_id)
        return result

    def request_cancel(self, job_id: str) -> bool:
        """Cancel a job. A queued job is removed without touching the GPU lock."""
        logger.info("Cancellation requested for job %s.", job_id)
        before = self.snapshot(job_id)
        if before is None or not before.is_active:
            logger.warning("Cancellation request ignored; job %s not active.", job_id)
            return False

        if before.state is JobState.QUEUED:
            with self._mutex:
                current = self._jobs.get(job_id)
                if current is None or current.state is not JobState.QUEUED:
                    return False
                cancelled = replace(
                    current,
                    state=JobState.CANCELLED,
                    finished_at=self._clock(),
                    message="Cancelled by user.",
                )
                self._jobs[job_id] = cancelled
                self._discard_from_queue(job_id)
                self._append_history_locked(cancelled)
            self.job_changed.emit(cancelled)
            return True

        def transition(job: JobSnapshot) -> JobSnapshot:
            if not job.is_active:
                return job
            return replace(job, state=JobState.CANCELLING)

        self._transition(job_id, transition)
        logger.debug("Job %s state set to CANCELLING.", job_id)
        return True

    def mark_succeeded(self, job_id: str, *, message: str = "") -> JobSnapshot | None:
        logger.info("Job %s succeeded. %s", job_id, message)
        return self._finish(job_id, JobState.SUCCEEDED, message=message, exit_code=0)

    def mark_failed(
        self, job_id: str, *, message: str, exit_code: int | None = None
    ) -> JobSnapshot | None:
        logger.error(
            "Job %s failed with exit_code=%s. Message: %s",
            job_id,
            exit_code,
            message,
        )
        return self._finish(job_id, JobState.FAILED, message=message, exit_code=exit_code)

    def mark_cancelled(
        self, job_id: str, *, message: str = "Cancelled by user."
    ) -> JobSnapshot | None:
        logger.info("Job %s cancelled. %s", job_id, message)
        return self._finish(job_id, JobState.CANCELLED, message=message)

    def handle_worker_exit(self, job_id: str, exit_code: int) -> JobSnapshot | None:
        """Reconcile worker process death with the job state and the GPU lock.

        Safe to call for jobs that already reached a terminal state and safe to
        call twice: the lock is only released by its current owner.
        """
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is not None and job.is_active and job.state is not JobState.QUEUED:
                if job.state is JobState.CANCELLING:
                    return self._finish(
                        job_id,
                        JobState.CANCELLED,
                        message="Cancelled by user.",
                        exit_code=exit_code,
                        require_lock_owner=False,
                    )
                if exit_code == 0:
                    return self._finish(
                        job_id,
                        JobState.FAILED,
                        message=(
                            "The engine process exited without reporting completion; "
                            "treating the job as failed."
                        ),
                        exit_code=exit_code,
                        require_lock_owner=False,
                    )
                return self._finish(
                    job_id,
                    JobState.FAILED,
                    message=f"The engine process exited unexpectedly (exit code {exit_code}).",
                    exit_code=exit_code,
                    require_lock_owner=False,
                )
            # Terminal, unknown or queued job: still make sure the lock is not stranded.
            released = self._lock.release(job_id)
            promoted = self._promote_next_locked() if released else None
            emit_owner = released or promoted is not None
            result = job

        if emit_owner:
            self._emit_gpu_owner()
        if promoted is not None:
            self.job_changed.emit(promoted)
            self.job_start_requested.emit(promoted)
        return result

    # -- internals -------------------------------------------------------

    def _transition(
        self, job_id: str, change: Callable[[JobSnapshot], JobSnapshot]
    ) -> JobSnapshot | None:
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = change(job)
            if updated == job:
                return job
            self._jobs[job_id] = updated
        self.job_changed.emit(updated)
        return updated

    def _finish(
        self,
        job_id: str,
        state: JobState,
        *,
        message: str,
        exit_code: int | None = None,
        require_lock_owner: bool = True,
    ) -> JobSnapshot | None:
        with self._mutex:
            job = self._jobs.get(job_id)
            finished: JobSnapshot | None = None
            promoted: JobSnapshot | None = None
            emit_owner = False

            if job is None:
                released = self._lock.release(job_id)
                promoted = self._promote_next_locked() if released else None
                emit_owner = released or promoted is not None
                result = None
            elif job.state.is_terminal:
                # Prevent a stale terminal transition from moving a job backwards.
                released = self._lock.release(job_id)
                promoted = self._promote_next_locked() if released else None
                emit_owner = released or promoted is not None
                result = job
            else:
                was_queued = job.state is JobState.QUEUED
                if was_queued:
                    self._discard_from_queue(job_id)
                updated = replace(
                    job,
                    state=state,
                    message=message,
                    exit_code=exit_code,
                    finished_at=self._clock(),
                )
                self._jobs[job_id] = updated
                if was_queued:
                    # A queued job never owned the GPU: no unlock, no promotion.
                    released = False
                else:
                    released = self._lock.release(job_id)
                    if require_lock_owner and not released:
                        warning = (message + " " if message else "") + (
                            "(warning: the GPU lock was not owned by this job)"
                        )
                        updated = replace(updated, message=warning)
                        self._jobs[job_id] = updated
                    promoted = (
                        self._promote_next_locked() if released or not require_lock_owner else None
                    )
                    emit_owner = released or promoted is not None
                self._append_history_locked(updated)
                finished = updated
                result = updated

        if emit_owner:
            self._emit_gpu_owner()
        if finished is not None:
            self.job_changed.emit(finished)
        if promoted is not None:
            self.job_changed.emit(promoted)
            self.job_start_requested.emit(promoted)
        return result

    def _promote_next_locked(self) -> JobSnapshot | None:
        """Promote the oldest still-queued job. Caller holds ``self._mutex``.

        The GPU lock is acquired inside the same critical section, so a second
        call can never promote the same job twice.
        """
        if self._lock.is_locked:
            return None
        while self._queue:
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            if job is None or job.state is not JobState.QUEUED:
                continue
            if not self._lock.try_acquire(job_id):
                self._queue.appendleft(job_id)
                return None
            promoted = replace(job, state=JobState.STARTING)
            self._jobs[job_id] = promoted
            logger.info("Promoted queued job %s to STARTING.", job_id)
            return promoted
        return None

    def _discard_from_queue(self, job_id: str) -> None:
        with contextlib.suppress(ValueError):
            self._queue.remove(job_id)

    def _emit_gpu_owner(self) -> None:
        self.gpu_ownership_changed.emit(self._lock.owner)

    # -- job history -----------------------------------------------------

    def _load_history(self) -> list[JobSnapshot]:
        if not self._history_file.exists():
            return []
        try:
            payload: object = json.loads(self._history_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("Could not read job history %s: %s", self._history_file, error)
            return []
        if not isinstance(payload, list):
            logger.warning("Job history %s is not a list; ignored.", self._history_file)
            return []
        records = [
            snapshot for item in payload if (snapshot := JobSnapshot.from_dict(item)) is not None
        ]
        return records[-self._history_limit :]

    def _append_history_locked(self, snapshot: JobSnapshot) -> None:
        self._history.append(snapshot)
        if len(self._history) > self._history_limit:
            del self._history[: -self._history_limit]
        self._write_history_locked()

    def _write_history_locked(self) -> None:
        """Atomically replace the history file. Failures are non-fatal."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._history_file.with_name(self._history_file.name + ".tmp")
            payload = [snapshot.to_dict() for snapshot in self._history]
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self._history_file)
        except OSError as error:
            logger.warning("Could not persist job history %s: %s", self._history_file, error)

    # -- testing / diagnostics ------------------------------------------

    def terminal_jobs(self) -> Iterable[JobSnapshot]:
        return (job for job in self.jobs() if job.state.is_terminal)

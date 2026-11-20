"""GPU-exclusive job orchestration foundation.

Unsloth and Heretic are heavyweight engines that must never hold GPU memory at
the same time. This manager owns that rule: a job may only start if it acquires
the single GPU lock, and every terminal path (success, failure, cancellation,
worker crash) releases it. The rule lives here, not in the GUI.

Only the foundation required for safe orchestration is implemented. There is no
queue: a second request is rejected with an explicit :class:`GpuBusyError` that
names the current owner, which the UI surfaces to the user.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count

from PySide6.QtCore import QObject, Signal


class JobState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
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


class GpuBusyError(RuntimeError):
    """Raised when a job requests the GPU while another job owns it."""

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
    """Tracks engine jobs and enforces exclusive GPU ownership."""

    job_changed = Signal(object)  # JobSnapshot
    gpu_ownership_changed = Signal(object)  # str | None

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._lock = GpuLock()
        self._mutex = threading.RLock()
        self._jobs: dict[str, JobSnapshot] = {}
        self._ids = count(1)

    # -- queries ---------------------------------------------------------

    @property
    def gpu_owner(self) -> str | None:
        return self._lock.owner

    def snapshot(self, job_id: str) -> JobSnapshot | None:
        with self._mutex:
            return self._jobs.get(job_id)

    def jobs(self) -> list[JobSnapshot]:
        with self._mutex:
            return sorted(self._jobs.values(), key=lambda job: job.created_at)

    def active_job(self) -> JobSnapshot | None:
        with self._mutex:
            for job in self._jobs.values():
                if job.is_active:
                    return job
        return None

    def can_accept(self) -> bool:
        return not self._lock.is_locked

    # -- lifecycle -------------------------------------------------------

    def submit(self, engine: str, model_ref: str = "") -> JobSnapshot:
        """Start tracking a new job, or raise :class:`GpuBusyError`."""
        engine = engine.strip().lower()
        if not engine:
            raise ValueError("engine must be a non-empty name")

        with self._mutex:
            job_id = f"{engine}-{next(self._ids)}"
            if not self._lock.try_acquire(job_id):
                owner_id = self._lock.owner or ""
                owner = self._jobs.get(owner_id)
                raise GpuBusyError(
                    owner_job_id=owner_id,
                    owner_engine=owner.engine if owner else "another",
                    requested_engine=engine,
                )

            job = JobSnapshot(
                job_id=job_id,
                engine=engine,
                model_ref=model_ref,
                state=JobState.STARTING,
                created_at=self._clock(),
            )
            self._jobs[job_id] = job
            self._emit_gpu_owner()
        self.job_changed.emit(job)
        return job

    def mark_running(self, job_id: str, *, pid: int | None = None) -> JobSnapshot | None:
        def transition(job: JobSnapshot) -> JobSnapshot:
            if job.state is not JobState.STARTING:
                return job
            message = f"pid {pid}" if pid is not None else job.message
            return replace(job, state=JobState.RUNNING, started_at=self._clock(), message=message)

        return self._transition(job_id, transition)

    def request_cancel(self, job_id: str) -> bool:
        """Ask a running job to stop. The GPU stays reserved until it exits."""

        def transition(job: JobSnapshot) -> JobSnapshot:
            if not job.is_active:
                return job
            return replace(job, state=JobState.CANCELLING)

        before = self.snapshot(job_id)
        if before is None or not before.is_active:
            return False
        self._transition(job_id, transition)
        return True

    def mark_succeeded(self, job_id: str, *, message: str = "") -> JobSnapshot | None:
        return self._finish(job_id, JobState.SUCCEEDED, message=message, exit_code=0)

    def mark_failed(
        self, job_id: str, *, message: str, exit_code: int | None = None
    ) -> JobSnapshot | None:
        return self._finish(job_id, JobState.FAILED, message=message, exit_code=exit_code)

    def mark_cancelled(
        self, job_id: str, *, message: str = "Cancelled by user."
    ) -> JobSnapshot | None:
        return self._finish(job_id, JobState.CANCELLED, message=message)

    def handle_worker_exit(self, job_id: str, exit_code: int) -> JobSnapshot | None:
        """Reconcile worker process death with the job state and the GPU lock.

        Safe to call for jobs that already reached a terminal state and safe to
        call twice: the lock is only released by its current owner.
        """
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is not None and job.is_active:
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
            # Terminal or unknown job: still make sure the lock is not stranded.
            if self._lock.release(job_id):
                self._emit_gpu_owner()
            return job

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
        exit_code: int | None,
        require_lock_owner: bool = True,
    ) -> JobSnapshot | None:
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is None:
                self._lock.release(job_id)
                return None
            if job.state.is_terminal:
                # Prevent a stale terminal transition from moving a job backwards.
                self._lock.release(job_id)
                return job
            updated = replace(
                job,
                state=state,
                message=message,
                exit_code=exit_code,
                finished_at=self._clock(),
            )
            self._jobs[job_id] = updated
            released = self._lock.release(job_id)
            if require_lock_owner and not released:
                message = (message + " " if message else "") + (
                    "(warning: the GPU lock was not owned by this job)"
                )
                updated = replace(updated, message=message)
                self._jobs[job_id] = updated
        if released or not require_lock_owner:
            self._emit_gpu_owner()
        self.job_changed.emit(updated)
        return updated

    def _emit_gpu_owner(self) -> None:
        self.gpu_ownership_changed.emit(self._lock.owner)

    # -- testing / diagnostics ------------------------------------------

    def terminal_jobs(self) -> Iterable[JobSnapshot]:
        return (job for job in self.jobs() if job.state.is_terminal)

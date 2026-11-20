"""GPU-exclusive job manager behaviour."""

from __future__ import annotations

import pytest

from workers.job_manager import (
    GpuBusyError,
    GpuLock,
    JobManager,
    JobState,
)


def test_gpu_lock_is_single_owner() -> None:
    lock = GpuLock()
    assert lock.try_acquire("a") is True
    assert lock.try_acquire("b") is False
    assert lock.owner == "a"
    assert lock.release("b") is False
    assert lock.owner == "a"
    assert lock.release("a") is True
    assert lock.is_locked is False


def test_submit_grants_exclusive_gpu(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth", "Qwen/Qwen2.5-0.5B")
    assert job.state is JobState.STARTING
    assert manager.gpu_owner == job.job_id
    assert manager.can_accept() is False


def test_second_engine_is_rejected_with_explicit_reason(qapp) -> None:
    manager = JobManager()
    first = manager.submit("unsloth")
    with pytest.raises(GpuBusyError) as caught:
        manager.submit("heretic", "some/model")
    assert caught.value.owner_job_id == first.job_id
    assert caught.value.owner_engine == "unsloth"
    assert caught.value.requested_engine == "heretic"
    assert manager.gpu_owner == first.job_id


def test_success_releases_the_gpu(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    manager.mark_running(job.job_id, pid=1234)
    finished = manager.mark_succeeded(job.job_id, message="adapter saved")
    assert finished.state is JobState.SUCCEEDED
    assert finished.exit_code == 0
    assert manager.gpu_owner is None
    assert manager.active_job() is None
    assert manager.can_accept() is True


def test_failure_releases_the_gpu(qapp) -> None:
    manager = JobManager()
    job = manager.submit("heretic")
    finished = manager.mark_failed(job.job_id, message="CUDA out of memory")
    assert finished.state is JobState.FAILED
    assert finished.message == "CUDA out of memory"
    assert manager.gpu_owner is None


def test_cancellation_keeps_lock_until_worker_exits(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    assert manager.request_cancel(job.job_id) is True
    assert manager.snapshot(job.job_id).state is JobState.CANCELLING
    assert manager.gpu_owner == job.job_id

    finished = manager.handle_worker_exit(job.job_id, 1)
    assert finished.state is JobState.CANCELLED
    assert manager.gpu_owner is None


def test_worker_crash_releases_lock_and_fails_job(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    finished = manager.handle_worker_exit(job.job_id, -1073741819)
    assert finished.state is JobState.FAILED
    assert finished.exit_code == -1073741819
    assert "exited unexpectedly" in finished.message
    assert manager.gpu_owner is None


def test_clean_exit_without_completion_is_not_success(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    finished = manager.handle_worker_exit(job.job_id, 0)
    assert finished.state is JobState.FAILED
    assert "without reporting completion" in finished.message
    assert manager.gpu_owner is None


def test_worker_exit_for_unknown_job_does_not_raise(qapp) -> None:
    manager = JobManager()
    assert manager.handle_worker_exit("does-not-exist", 3) is None
    assert manager.gpu_owner is None


def test_repeated_terminal_transitions_do_not_change_state(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    manager.mark_failed(job.job_id, message="first")
    again = manager.mark_succeeded(job.job_id, message="late success")
    assert again.state is JobState.FAILED
    assert again.message == "first"


def test_cancel_after_completion_is_rejected(qapp) -> None:
    manager = JobManager()
    job = manager.submit("unsloth")
    manager.mark_succeeded(job.job_id)
    assert manager.request_cancel(job.job_id) is False


def test_signals_report_job_and_ownership_transitions(qapp) -> None:
    manager = JobManager()
    states: list[JobState] = []
    owners: list[str | None] = []
    manager.job_changed.connect(lambda job: states.append(job.state))
    manager.gpu_ownership_changed.connect(owners.append)

    job = manager.submit("unsloth")
    manager.mark_running(job.job_id, pid=42)
    manager.mark_succeeded(job.job_id)

    assert states == [JobState.STARTING, JobState.RUNNING, JobState.SUCCEEDED]
    assert owners == [job.job_id, None]


def test_engine_name_is_required(qapp) -> None:
    manager = JobManager()
    with pytest.raises(ValueError):
        manager.submit("   ")

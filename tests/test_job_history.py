"""FIFO job queue and persisted terminal job history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workers.job_manager import JobManager, JobSnapshot, JobState


@pytest.fixture
def make_manager(tmp_path: Path):
    """Build isolated managers that share one history file under ``tmp_path``."""

    def _make(file_name: str = "jobs.json", **kwargs) -> JobManager:
        return JobManager(history_file=tmp_path / file_name, **kwargs)

    return _make


def test_second_and_third_jobs_queue_in_fifo_order_and_start_in_order(qapp, make_manager) -> None:
    manager = make_manager()
    starts: list[JobSnapshot] = []
    manager.job_start_requested.connect(starts.append)

    running = manager.submit("unsloth", "model-0")
    first_queued = manager.submit("heretic", "model-1")
    second_queued = manager.submit("unsloth", "model-2")

    assert running.state is JobState.STARTING
    assert first_queued.state is JobState.QUEUED
    assert second_queued.state is JobState.QUEUED
    assert [job.job_id for job in manager.queued_jobs()] == [
        first_queued.job_id,
        second_queued.job_id,
    ]
    # Queued submissions notify the tab that a worker must be registered.
    assert [job.job_id for job in starts] == [first_queued.job_id, second_queued.job_id]
    assert manager.gpu_owner == running.job_id

    manager.mark_succeeded(running.job_id)
    assert manager.gpu_owner == first_queued.job_id
    assert [job.job_id for job in manager.queued_jobs()] == [second_queued.job_id]
    assert manager.snapshot(first_queued.job_id).state is JobState.STARTING

    manager.mark_succeeded(first_queued.job_id)
    assert manager.gpu_owner == second_queued.job_id
    assert manager.snapshot(second_queued.job_id).state is JobState.STARTING
    assert manager.queued_jobs() == []

    assert [job.job_id for job in starts] == [
        first_queued.job_id,
        second_queued.job_id,
        first_queued.job_id,
        second_queued.job_id,
    ]
    assert [job.state for job in starts] == [
        JobState.QUEUED,
        JobState.QUEUED,
        JobState.STARTING,
        JobState.STARTING,
    ]


@pytest.mark.parametrize(
    "finish",
    [
        lambda manager, job_id: manager.mark_succeeded(job_id),
        lambda manager, job_id: manager.mark_failed(job_id, message="boom"),
        lambda manager, job_id: manager.mark_cancelled(job_id),
        lambda manager, job_id: manager.handle_worker_exit(job_id, -1073741819),
    ],
)
def test_every_terminal_path_auto_starts_the_oldest_queued_job(qapp, make_manager, finish) -> None:
    manager = make_manager()
    running = manager.submit("unsloth")
    queued = manager.submit("heretic")

    finish(manager, running.job_id)

    assert manager.snapshot(running.job_id).state.is_terminal
    assert manager.snapshot(queued.job_id).state is JobState.STARTING
    assert manager.gpu_owner == queued.job_id
    assert manager.queued_jobs() == []


def test_cancel_while_queued_removes_it_without_touching_the_gpu_lock(qapp, make_manager) -> None:
    manager = make_manager()
    running = manager.submit("unsloth")
    cancelled_job = manager.submit("heretic")
    survivor = manager.submit("unsloth")

    assert manager.request_cancel(cancelled_job.job_id) is True
    cancelled = manager.snapshot(cancelled_job.job_id)
    assert cancelled is not None
    assert cancelled.state is JobState.CANCELLED
    assert manager.gpu_owner == running.job_id
    assert [job.job_id for job in manager.queued_jobs()] == [survivor.job_id]
    assert [job.job_id for job in manager.history()] == [cancelled_job.job_id]

    # The cancelled job is skipped when the current owner finishes.
    manager.mark_succeeded(running.job_id)
    assert manager.gpu_owner == survivor.job_id
    assert manager.snapshot(cancelled_job.job_id).state is JobState.CANCELLED

    # Cancelling a queued job twice is a no-op.
    assert manager.request_cancel(cancelled_job.job_id) is False


def test_stale_terminal_notifications_do_not_promote_twice(qapp, make_manager) -> None:
    manager = make_manager()
    starts: list[JobSnapshot] = []
    manager.job_start_requested.connect(starts.append)

    running = manager.submit("unsloth")
    first_queued = manager.submit("heretic")
    second_queued = manager.submit("unsloth")

    manager.mark_succeeded(running.job_id)
    # Late duplicates from the finished worker must not hand the GPU again.
    manager.handle_worker_exit(running.job_id, 0)
    manager.mark_failed(running.job_id, message="late duplicate")

    assert manager.gpu_owner == first_queued.job_id
    assert [job.job_id for job in starts] == [
        first_queued.job_id,
        second_queued.job_id,
        first_queued.job_id,
    ]
    assert [job.job_id for job in manager.history()] == [running.job_id]

    # Each later completion promotes exactly one remaining job.
    manager.mark_succeeded(first_queued.job_id)
    assert manager.gpu_owner == second_queued.job_id
    assert starts[-1].job_id == second_queued.job_id
    assert starts[-1].state is JobState.STARTING


def test_history_persists_across_manager_restarts(qapp, make_manager) -> None:
    original = make_manager()
    succeeded = original.submit("unsloth", "model-a")
    original.mark_succeeded(succeeded.job_id, message="adapter saved")
    failed = original.submit("heretic", "model-b")
    original.mark_failed(failed.job_id, message="CUDA out of memory", exit_code=2)
    cancelled = original.submit("unsloth", "model-c")
    original.mark_cancelled(cancelled.job_id)

    restarted = make_manager()
    records = restarted.history()
    assert [record.job_id for record in records] == [
        succeeded.job_id,
        failed.job_id,
        cancelled.job_id,
    ]
    assert records[0].state is JobState.SUCCEEDED
    assert records[0].model_ref == "model-a"
    assert records[0].message == "adapter saved"
    assert records[0].finished_at is not None
    assert records[1].state is JobState.FAILED
    assert records[1].exit_code == 2
    assert records[2].state is JobState.CANCELLED

    # Persisted history is separate from the in-memory session view.
    assert list(restarted.terminal_jobs()) == []
    assert restarted.snapshot(succeeded.job_id) is None
    assert restarted.gpu_owner is None

    # A third manager chaining off the same store keeps appending in order.
    third = make_manager()
    assert [record.job_id for record in third.history()] == [
        succeeded.job_id,
        failed.job_id,
        cancelled.job_id,
    ]


def test_corrupt_and_incomplete_history_entries_are_ignored(qapp, tmp_path: Path) -> None:
    valid = JobSnapshot(
        job_id="unsloth-1",
        engine="unsloth",
        model_ref="a/model",
        state=JobState.SUCCEEDED,
        created_at=1.0,
        finished_at=2.0,
        exit_code=0,
        message="done",
    )
    history_file = tmp_path / "jobs.json"
    history_file.write_text(
        json.dumps(
            [
                "not an object",
                None,
                {"job_id": "no-state"},
                {"job_id": "not-terminal", "engine": "x", "state": "running", "created_at": 1.0},
                {"job_id": "bad-time", "engine": "x", "state": "failed", "created_at": "yesterday"},
                {"job_id": "bool-time", "engine": "x", "state": "failed", "created_at": True},
                valid.to_dict(),
            ]
        ),
        encoding="utf-8",
    )

    manager = JobManager(history_file=history_file)
    assert [record.job_id for record in manager.history()] == ["unsloth-1"]
    assert manager.history()[0].message == "done"


def test_non_list_history_payload_is_ignored(qapp, tmp_path: Path) -> None:
    history_file = tmp_path / "jobs.json"
    history_file.write_text('{"jobs": []}', encoding="utf-8")
    assert JobManager(history_file=history_file).history() == []

    history_file.write_text("not json at all", encoding="utf-8")
    assert JobManager(history_file=history_file).history() == []


def test_history_is_bounded_and_survives_restart(qapp, make_manager) -> None:
    manager = make_manager(history_limit=3)
    job_ids = []
    for _ in range(5):
        job = manager.submit("unsloth")
        job_ids.append(job.job_id)
        manager.mark_succeeded(job.job_id)

    assert [record.job_id for record in manager.history()] == job_ids[-3:]
    assert [record.job_id for record in make_manager(history_limit=3).history()] == job_ids[-3:]


def test_queued_job_failure_is_persisted_like_any_terminal_job(qapp, make_manager) -> None:
    manager = make_manager()
    running = manager.submit("unsloth")
    queued = manager.submit("heretic")
    manager.mark_failed(queued.job_id, message="cancelled before launch")

    assert manager.gpu_owner == running.job_id
    assert [record.job_id for record in manager.history()] == [queued.job_id]
    assert manager.snapshot(queued.job_id).state is JobState.FAILED

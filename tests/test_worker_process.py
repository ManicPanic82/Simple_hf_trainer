"""End-to-end worker process behaviour over the JSON-lines protocol."""

from __future__ import annotations

import sys
from pathlib import Path

from workers.protocol import MessageType
from workers.worker_process import WorkerProcess

SCRIPT = Path(__file__).resolve().parent / "support" / "fake_worker.py"


def test_worker_handles_valid_and_malformed_output(qapp, wait_for) -> None:
    worker = WorkerProcess(sys.executable, SCRIPT)
    messages = []
    errors = []
    exits: list[int] = []
    worker.message_received.connect(messages.append)
    worker.protocol_error.connect(errors.append)
    worker.exited.connect(exits.append)

    worker.start()
    try:
        assert wait_for(lambda: any(m.type is MessageType.HELLO for m in messages))
        assert wait_for(lambda: bool(errors))
        assert errors[0].code == "invalid_json"
        assert worker.running is True
        assert worker.pid > 0

        assert worker.request_cancel("job-42") is True
        assert wait_for(lambda: bool(exits), timeout_ms=10_000)

        assert any(m.type is MessageType.JOB_COMPLETE for m in messages)
        assert all(m.job_id == "job-42" for m in messages if m.type is not MessageType.HELLO)
        assert exits == [0]
    finally:
        worker.stop()


def test_send_to_stopped_worker_returns_false(qapp) -> None:
    worker = WorkerProcess(sys.executable, SCRIPT)
    assert worker.send_message(MessageType.CANCEL, job_id="job-1") is False
    assert worker.running is False


def test_missing_interpreter_reports_start_failure(qapp, wait_for, tmp_path: Path) -> None:
    worker = WorkerProcess(tmp_path / "does-not-exist" / "python.exe", SCRIPT)
    failures: list[str] = []
    worker.failed_to_start.connect(failures.append)

    worker.start()
    assert wait_for(lambda: bool(failures), timeout_ms=10_000)
    assert "could not be started" in failures[0]
    worker.stop()


def test_stop_terminates_a_running_worker(qapp, wait_for) -> None:
    worker = WorkerProcess(sys.executable, SCRIPT)
    messages = []
    worker.message_received.connect(messages.append)

    worker.start()
    assert wait_for(lambda: bool(messages))
    worker.stop(grace_ms=500)
    assert worker.running is False


def test_worker_runs_in_the_requested_directory(qapp, wait_for, tmp_path: Path) -> None:
    worker = WorkerProcess(sys.executable, SCRIPT, cwd=tmp_path)
    messages = []
    worker.message_received.connect(messages.append)

    worker.start()
    try:
        assert wait_for(lambda: bool(messages), timeout_ms=10_000)
        reported = Path(messages[0].payload["cwd"]).resolve()
        assert reported == tmp_path.resolve()
    finally:
        worker.stop()


def test_missing_working_directory_is_reported(qapp, wait_for, tmp_path: Path) -> None:
    worker = WorkerProcess(sys.executable, SCRIPT, cwd=tmp_path / "missing")
    failures: list[str] = []
    worker.failed_to_start.connect(failures.append)

    worker.start()
    assert wait_for(lambda: bool(failures), timeout_ms=10_000)
    worker.stop()


def test_cancel_fallback_terminates_unresponsive_worker(qapp, wait_for) -> None:
    script = Path(__file__).resolve().parent / "support" / "unresponsive_worker.py"
    worker = WorkerProcess(sys.executable, script)
    exits: list[int] = []
    worker.exited.connect(exits.append)

    worker.start()
    try:
        assert wait_for(lambda: worker.running, timeout_ms=5_000)
        assert worker.request_cancel("job-99", grace_ms=100) is True
        assert wait_for(lambda: bool(exits), timeout_ms=8_000)
        assert worker.running is False
    finally:
        worker.stop()

"""Live Heretic console: streaming, history retention, and log export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.engine_registry import HERETIC, EngineState, EngineStatus
from ui.console import ConsoleWidget
from ui.heretic_tab import HereticTab
from workers.heretic_worker import OutputInterceptor
from workers.protocol import MessageType, WorkerMessage
from workers.worker_process import split_complete_lines


@pytest.fixture
def console(qapp) -> ConsoleWidget:
    widget = ConsoleWidget(max_visible_blocks=200)
    widget.start_session("Test run", "unit")
    return widget


# ---------------------------------------------------------------------------
# Console widget
# ---------------------------------------------------------------------------


def test_append_tags_levels_and_preserves_text(console: ConsoleWidget) -> None:
    console.append("plain line")
    console.append("careful", level="warning")
    console.append("broken", level="error")
    console.append("note", level="system")

    text = console.visible_text()
    assert "[WARN] careful" in text
    assert "[ERROR] broken" in text
    assert "plain line" in text
    assert "note" in text
    # Every rendered line is timestamped.
    for line in text.splitlines():
        assert line.startswith("[")


def test_append_splits_multiline_output(console: ConsoleWidget) -> None:
    console.append("first\nsecond\nthird")
    history = [entry.text for entry in console.history]
    assert history[-3:] == ["first", "second", "third"]


def test_clear_display_keeps_history(console: ConsoleWidget) -> None:
    console.append("retained")
    assert "retained" in console.visible_text()

    console.clear_display()

    assert console.visible_text() == ""
    assert any(entry.text == "retained" for entry in console.history)
    assert "retained" in console.full_text()


def test_save_log_writes_full_history_after_clear(console: ConsoleWidget, tmp_path: Path) -> None:
    console.append("line one")
    console.append("line two", level="warning")
    console.clear_display()

    target = console.save_log_to(tmp_path / "heretic.log")

    written = target.read_text(encoding="utf-8")
    assert "line one" in written
    assert "[WARN] line two" in written
    assert "Session 1: Test run" in written


def test_sessions_are_distinct_and_retained(console: ConsoleWidget) -> None:
    console.append("idle output")
    console.start_session("Second run")
    console.append("run output")

    assert len(console.sessions) == 2
    assert console.sessions[0].entries
    assert console.sessions[1].entries
    full = console.full_text()
    assert "idle output" in full
    assert "run output" in full


def test_visible_blocks_are_bounded_but_history_is_not(qapp) -> None:
    widget = ConsoleWidget(max_visible_blocks=200)
    widget.start_session("Long run")
    for index in range(600):
        widget.append(f"line {index}")

    assert len(widget.visible_text().splitlines()) <= 200
    # The complete history survives regardless of the visible block limit.
    assert len(widget.history) == 602  # header + 2 separator lines + 600 lines
    assert "line 0" in widget.full_text()


def test_copy_without_selection_copies_full_history(console: ConsoleWidget) -> None:
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QApplication

    console.append("copied line")
    console.text_edit.moveCursor(QTextCursor.MoveOperation.Start)
    console.text_edit.textCursor().clearSelection()
    console.copy()

    assert "copied line" in QApplication.clipboard().text()


def test_save_log_dialog_emits_saved_signal(
    console: ConsoleWidget, tmp_path: Path, monkeypatch
) -> None:
    console.append("payload")
    received: list[str] = []
    console.saved.connect(received.append)
    target = tmp_path / "out.log"
    monkeypatch.setattr(
        "ui.console.QFileDialog.getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "Log files (*.log *.txt)")),
    )

    saved = console.save_log()

    assert saved == target
    assert target.is_file()
    assert "payload" in target.read_text(encoding="utf-8")
    assert received == [str(target)]


# ---------------------------------------------------------------------------
# Worker partial-line handling
# ---------------------------------------------------------------------------


def test_split_complete_lines_buffers_partial_tail() -> None:
    lines, remainder = split_complete_lines(b"", b"alpha\nbet")
    assert lines == ["alpha"]
    assert remainder == b"bet"

    lines, remainder = split_complete_lines(remainder, b"a\r\ngamma\n")
    assert lines == ["beta", "gamma"]
    assert remainder == b""


def test_split_complete_lines_handles_multibyte_split() -> None:
    payload = "🦥 Heretic\n".encode()
    lines, remainder = split_complete_lines(b"", payload[:2])
    assert lines == []
    assert remainder == payload[:2]

    lines, remainder = split_complete_lines(remainder, payload[2:])
    assert lines == ["🦥 Heretic"]
    assert remainder == b""


def test_output_interceptor_assembles_partial_writes(monkeypatch) -> None:
    import workers.heretic_worker as worker

    captured: list[dict] = []

    class Stream:
        def write(self, text: str) -> int:
            captured.extend(json.loads(line) for line in text.splitlines())
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(worker, "_PROTOCOL_STREAM", Stream())

    interceptor = OutputInterceptor("job-1", 3, None)
    interceptor.write("Loading mo")
    interceptor.write("del...\nRunning trial 2 of 3\n")

    messages = [entry for entry in captured if entry["type"] == "log"]
    assert [entry["payload"]["message"] for entry in messages] == [
        "Loading model...",
        "Running trial 2 of 3",
    ]
    assert all(entry["payload"]["stream"] == "stdout" for entry in messages)
    progress = [entry for entry in captured if entry["type"] == "progress"]
    assert progress and progress[-1]["payload"]["step"] == 2


def test_output_interceptor_flush_emits_buffered_tail(monkeypatch) -> None:
    import workers.heretic_worker as worker

    captured: list[dict] = []

    class Stream:
        def write(self, text: str) -> int:
            captured.extend(json.loads(line) for line in text.splitlines())
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(worker, "_PROTOCOL_STREAM", Stream())

    interceptor = OutputInterceptor("job-2", 1, None)
    interceptor.write("no trailing newline")
    assert [entry["type"] for entry in captured] == []

    interceptor.flush()

    logs = [entry for entry in captured if entry["type"] == "log"]
    assert logs[-1]["payload"]["message"] == "no trailing newline"


# ---------------------------------------------------------------------------
# Heretic tab wiring
# ---------------------------------------------------------------------------


def _tab(tmp_path: Path) -> HereticTab:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    storage = MagicMock()
    storage.validate.return_value = SimpleNamespace(path=outputs, issues=[], ok=True, exists=True)
    storage.path_for.return_value = outputs
    state = MagicMock()
    state.engines = {}
    state.models = []
    state.hardware = None
    registry = MagicMock()
    registry.status.return_value = EngineStatus(
        HERETIC, EngineState.AVAILABLE, None, "heretic test", "2.1.0"
    )
    return HereticTab(
        state,
        storage,
        job_manager=MagicMock(),
        registry=registry,
        model_service=MagicMock(),
    )


def test_heretic_tab_console_streams_protocol_logs(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.LOG,
            payload={"message": "stdout line", "level": "info", "stream": "stdout"},
        )
    )
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.LOG,
            payload={"message": "stderr line", "level": "warning", "stream": "stderr"},
        )
    )

    text = tab.console.visible_text()
    assert "[stdout] stdout line" in text
    assert "[stderr] stderr line" in text
    assert "[WARN]" in text


def test_heretic_tab_clear_display_does_not_stop_worker(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    worker = MagicMock()
    worker.running = True
    tab._active_worker = worker
    tab._active_job_id = "heretic-1"
    tab.console.append("engine output")

    tab.console.clear_button.click()

    assert tab.console.visible_text() == ""
    assert "engine output" in tab.console.full_text()
    worker.request_cancel.assert_not_called()
    tab._job_manager.request_cancel.assert_not_called()


def test_heretic_tab_console_survives_worker_exit(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab._active_job_id = "heretic-1"
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.LOG,
            payload={"message": "before exit", "level": "info", "stream": "stdout"},
        )
    )
    tab._on_worker_exited(1)

    assert "before exit" in tab.console.full_text()
    assert tab.console.visible_text()  # still rendered after the worker exits

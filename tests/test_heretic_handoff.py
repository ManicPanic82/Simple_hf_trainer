"""Heretic worker log wiring and Heretic -> Models hand-off."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.engine_registry import HERETIC, EngineState, EngineStatus
from ui.heretic_tab import HereticTab
from workers.protocol import MessageType, WorkerMessage


def _make_tab(tmp_path: Path, *, job_manager, model_service, state) -> HereticTab:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    storage = MagicMock()
    storage.validate.return_value = SimpleNamespace(path=outputs, issues=[], ok=True, exists=True)
    storage.path_for.return_value = outputs
    registry = MagicMock()
    registry.status.return_value = EngineStatus(
        HERETIC, EngineState.AVAILABLE, None, "heretic test", "test"
    )
    state.engines = {}
    state.models = []
    state.hardware = None
    return HereticTab(
        state,
        storage,
        job_manager=job_manager,
        registry=registry,
        model_service=model_service,
    )


def test_successful_heretic_job_publishes_scanned_models(qapp, tmp_path: Path) -> None:
    state = MagicMock()
    model_service = MagicMock()
    job_manager = MagicMock()
    scanned = [SimpleNamespace(name="abliterated-model", kind="abliterated")]
    model_service.scan_local_models.return_value = scanned
    tab = _make_tab(tmp_path, job_manager=job_manager, model_service=model_service, state=state)

    job_id = "heretic-1"
    tab._active_job_id = job_id
    out = tmp_path / "outputs" / "heretic_heretic-1" / "model"
    for message in (
        WorkerMessage(type=MessageType.HELLO, payload={"pid": 99}),
        WorkerMessage(type=MessageType.READY, payload={"engine": "heretic"}),
        WorkerMessage(type=MessageType.RESULT, payload={"output_dir": str(out)}),
        WorkerMessage(
            type=MessageType.JOB_COMPLETE,
            payload={"success": True, "message": "done"},
            job_id=job_id,
        ),
    ):
        tab._on_worker_message(message)
    tab._on_worker_exited(0)

    model_service.scan_local_models.assert_called_once_with()
    state.publish_models.assert_called_once_with(scanned)
    assert tab.last_model_row.value.text() == str(out)


def test_worker_log_and_protocol_errors_are_visible(qapp, tmp_path: Path) -> None:
    state = MagicMock()
    tab = _make_tab(
        tmp_path,
        job_manager=MagicMock(),
        model_service=MagicMock(),
        state=state,
    )
    tab._on_worker_log_line("🦥 Heretic: loading model")
    tab._on_worker_protocol_error("invalid_json: bad line")

    text = tab.log_view.toPlainText()
    assert "[engine stderr] 🦥 Heretic: loading model" in text
    assert "[protocol warning] invalid_json: bad line" in text

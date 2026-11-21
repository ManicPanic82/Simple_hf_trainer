"""Tabs hold queued jobs and launch them when the job manager promotes them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.engine_registry import EngineState, EngineStatus
from ui.heretic_tab import HereticTab
from ui.training_tab import TrainingTab
from workers.job_manager import JobState


def _base(tmp_path: Path):
    state = MagicMock()
    state.engines = {}
    state.models = []
    state.hardware = None
    storage = MagicMock()
    storage.path_for.return_value = tmp_path / "outputs"
    storage.path_for.return_value.mkdir(parents=True, exist_ok=True)
    storage.validate.return_value = SimpleNamespace(
        path=tmp_path / "outputs", issues=[], ok=True, exists=True
    )
    return state, storage


def test_training_queued_job_launches_on_promotion(qapp, tmp_path: Path) -> None:
    state, storage = _base(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text":"hi"}\n', encoding="utf-8")
    registry = MagicMock()
    registry.status.return_value = EngineStatus(
        "unsloth", EngineState.AVAILABLE, None, "unsloth test", "test"
    )
    manager = MagicMock()
    manager.submit.return_value = SimpleNamespace(job_id="queued-1", state=JobState.QUEUED)
    tab = TrainingTab(
        state,
        storage,
        job_manager=manager,
        registry=registry,
        model_service=MagicMock(),
        dataset_service=MagicMock(),
    )
    tab.model_combo.addItem("a-model", "a-model")
    tab.model_combo.setCurrentIndex(0)
    tab.dataset_combo.addItem(str(dataset), str(dataset))
    tab.dataset_combo.setCurrentIndex(0)

    with patch("ui.training_tab.WorkerProcess") as worker_cls:
        tab._start_training()
        worker_cls.assert_not_called()
        assert "queued-1" in tab._queued_jobs

        tab._on_job_start_requested(SimpleNamespace(job_id="queued-1", state=JobState.STARTING))
        worker_cls.assert_called_once()
        worker_cls.return_value.start.assert_called_once_with()
        assert "queued-1" not in tab._queued_jobs


def test_heretic_queued_job_launches_on_promotion(qapp, tmp_path: Path) -> None:
    state, storage = _base(tmp_path)
    registry = MagicMock()
    registry.status.return_value = EngineStatus(
        "heretic", EngineState.AVAILABLE, None, "heretic test", "test"
    )
    manager = MagicMock()
    manager.submit.return_value = SimpleNamespace(job_id="queued-2", state=JobState.QUEUED)
    tab = HereticTab(
        state,
        storage,
        job_manager=manager,
        registry=registry,
        model_service=MagicMock(),
    )
    tab.model_combo.addItem("a-model", "a-model")
    tab.model_combo.setCurrentIndex(0)

    with patch("ui.heretic_tab.WorkerProcess") as worker_cls:
        tab._start_heretic()
        worker_cls.assert_not_called()
        assert "queued-2" in tab._queued_jobs

        tab._on_job_start_requested(SimpleNamespace(job_id="queued-2", state=JobState.STARTING))
        worker_cls.assert_called_once()
        worker_cls.return_value.start.assert_called_once_with()
        assert "queued-2" not in tab._queued_jobs

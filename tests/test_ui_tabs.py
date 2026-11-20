"""Tab-level UI behaviour and settings dialog validation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog

from app.state import AppState
from services.engine_registry import EngineState, EngineStatus
from services.settings_service import (
    AppSettings,
    EngineSettings,
    PathSettings,
    SettingsService,
)
from services.storage_service import StorageService
from ui.datasets_tab import DatasetsTab
from ui.heretic_tab import HereticTab
from ui.models_tab import ModelInfoWorker, ModelsTab
from ui.settings_dialog import SettingsDialog
from ui.training_tab import TrainingTab
from workers.protocol import MessageType, WorkerMessage


def settings_with(tmp_path: Path, **paths: Path) -> AppSettings:
    return AppSettings(paths=PathSettings(**paths))


def test_models_tab_shows_clear_storage_problem(qapp, tmp_path: Path) -> None:
    blocked = tmp_path / "models"
    blocked.write_text("a file", encoding="utf-8")
    settings = settings_with(
        tmp_path,
        models_dir=blocked,
        datasets_dir=tmp_path / "d",
        outputs_dir=tmp_path / "o",
        hf_cache_dir=tmp_path / "c",
    )
    tab = ModelsTab(AppState(), StorageService(settings))

    assert tab.models_path.pill.text() == "Problem"
    assert tab.search_button.isEnabled() is False
    assert "not a directory" in tab.models_path.note()


def test_models_tab_shows_ready_state_for_existing_folder(qapp, tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    settings = settings_with(
        tmp_path,
        models_dir=models,
        datasets_dir=tmp_path / "d",
        outputs_dir=tmp_path / "o",
        hf_cache_dir=tmp_path / "c",
    )
    tab = ModelsTab(AppState(), StorageService(settings))
    assert tab.models_path.pill.text() == "Ready"
    assert tab.status_pill.text() == "Model library ready"


def test_datasets_tab_reports_workspace_readiness(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "cache",
    )
    tab = DatasetsTab(AppState(), StorageService(settings))

    assert tab.status_pill.text() == "Dataset workspace ready"


def test_engine_actions_wait_for_initial_detection(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "cache",
    )
    state = AppState()
    training = TrainingTab(state, StorageService(settings))
    heretic = HereticTab(state, StorageService(settings))

    assert training.status_pill.text() == "Checking Unsloth"
    assert heretic.status_pill.text() == "Checking Heretic"
    assert not training.start_button.isEnabled()
    assert not heretic.run_button.isEnabled()


def test_engine_cards_reflect_detection(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "m",
        datasets_dir=tmp_path / "d",
        outputs_dir=tmp_path / "o",
        hf_cache_dir=tmp_path / "c",
    )
    state = AppState()
    training = TrainingTab(state, StorageService(settings))
    heretic = HereticTab(state, StorageService(settings))

    state.publish_engines(
        {
            "unsloth": EngineStatus(
                "unsloth", EngineState.INTERPRETER_MISSING, None, "No interpreter configured."
            ),
            "heretic": EngineStatus(
                "heretic", EngineState.AVAILABLE, Path("C:/h/python.exe"), "heretic 2.0", "2.0"
            ),
        }
    )

    assert training.engine_card.pill.text() == "Environment not set up"
    assert heretic.engine_card.pill.text() == "Engine available"
    assert heretic.engine_card.version_row.value.text() == "2.0"
    assert training.status_pill.text() == "Unsloth not set up"
    assert heretic.status_pill.text() == "Heretic engine available"


def test_settings_dialog_rejects_relative_path_without_saving(qapp, tmp_path: Path) -> None:
    service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=Path(__file__).parents[1] / "config" / "defaults.json",
    )
    current = service.load()
    dialog = SettingsDialog(current, service)
    dialog.models_edit.set_value("relative/models")
    dialog.save_button.click()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.status_pill.text() == "Fix the highlighted problems"
    assert dialog.message.isVisible() is False or dialog.message.text()


def test_settings_dialog_saves_valid_settings(qapp, tmp_path: Path) -> None:
    service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=Path(__file__).parents[1] / "config" / "defaults.json",
    )
    current = service.load()
    target = tmp_path / "custom-models"
    dialog = SettingsDialog(current, service)
    dialog.models_edit.set_value(str(target))
    dialog.unsloth_edit.set_value("")
    dialog.poll_spin.setValue(1500)
    dialog.save_button.click()

    assert dialog.result() == QDialog.DialogCode.Accepted
    saved = dialog.result_settings()
    assert saved is not None
    assert saved.paths.models_dir == target
    assert saved.ui.poll_interval_ms == 1500
    assert target.is_dir()
    assert service.load().ui.poll_interval_ms == 1500


def test_settings_dialog_keeps_engine_paths(qapp, tmp_path: Path) -> None:
    service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=Path(__file__).parents[1] / "config" / "defaults.json",
    )
    current = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "m",
            datasets_dir=tmp_path / "d",
            outputs_dir=tmp_path / "o",
            hf_cache_dir=tmp_path / "c",
        ),
        engines=EngineSettings(unsloth_python=str(tmp_path / "u.exe")),
    )
    for folder in (tmp_path / "m", tmp_path / "d", tmp_path / "o", tmp_path / "c"):
        folder.mkdir()

    dialog = SettingsDialog(current, service)
    dialog.save_button.click()

    saved = dialog.result_settings()
    assert saved is not None
    assert saved.engines.unsloth_python == str(tmp_path / "u.exe")


def test_model_info_lookup_runs_in_background(qapp, wait_for) -> None:
    class StubModelService:
        def get_hub_model_info(self, repo_id: str) -> dict:
            return {"repo_id": repo_id, "total_size_mb": 12.5, "files": [], "pipeline_tag": "text-generation"}

    worker = ModelInfoWorker(StubModelService(), "org/model")
    results = []
    worker.info_ready.connect(lambda repo_id, info: results.append((repo_id, info)))
    worker.start()
    try:
        assert wait_for(lambda: bool(results), timeout_ms=5_000)
        assert results[0][0] == "org/model"
        assert results[0][1]["total_size_mb"] == 12.5
    finally:
        worker.wait(5_000)


def test_training_job_keeps_gpu_lock_until_process_exit(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "cache",
    )
    tab = TrainingTab(AppState(), StorageService(settings))
    job = tab._job_manager.submit("unsloth", "org/model")
    tab._active_job_id = job.job_id

    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.JOB_COMPLETE,
            job_id=job.job_id,
            payload={"success": True},
        )
    )
    assert tab._job_manager.gpu_owner == job.job_id

    tab._on_worker_exited(0)
    assert tab._job_manager.gpu_owner is None


def test_training_setup_failure_releases_gpu_lock(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "cache",
    )
    tab = TrainingTab(AppState(), StorageService(settings))
    job = tab._job_manager.submit("unsloth", "org/model")
    tab._active_job_id = job.job_id

    tab._handle_setup_failure(job.job_id, OSError("disk unavailable"))

    assert tab._job_manager.gpu_owner is None
    assert tab._job_manager.snapshot(job.job_id).state.value == "failed"
    assert tab._active_job_id is None
    assert tab.start_button.isEnabled()
    assert not tab.cancel_button.isEnabled()


def test_heretic_setup_failure_releases_gpu_lock(qapp, tmp_path: Path) -> None:
    settings = settings_with(
        tmp_path,
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        outputs_dir=tmp_path / "outputs",
        hf_cache_dir=tmp_path / "cache",
    )
    tab = HereticTab(AppState(), StorageService(settings))
    job = tab._job_manager.submit("heretic", "org/model")
    tab._active_job_id = job.job_id

    tab._handle_setup_failure(job.job_id, OSError("disk unavailable"))

    assert tab._job_manager.gpu_owner is None
    assert tab._job_manager.snapshot(job.job_id).state.value == "failed"
    assert tab._active_job_id is None
    assert tab.run_button.isEnabled()
    assert not tab.cancel_button.isEnabled()

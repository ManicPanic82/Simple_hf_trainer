"""Acceptance: settings and libraries survive a fresh application start."""

from __future__ import annotations

import json
from pathlib import Path

from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.settings_service import (
    AppSettings,
    EngineSettings,
    PathSettings,
    SettingsService,
    UiSettings,
)
from services.storage_service import StorageService
from workers.job_manager import JobManager

DEFAULTS = Path(__file__).resolve().parents[1] / "config" / "defaults.json"


def test_settings_models_and_datasets_survive_a_fresh_start(tmp_path: Path, qapp) -> None:
    """A new context with a new service set rediscovers the same on-disk state."""
    folders = {
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "outputs_dir": tmp_path / "outputs",
        "hf_cache_dir": tmp_path / "hf-cache",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    model_dir = folders["models_dir"] / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    (folders["datasets_dir"] / "prepared.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")

    settings_file = tmp_path / "settings.json"
    original = AppSettings(
        paths=PathSettings(**folders),
        engines=EngineSettings(unsloth_python=str(tmp_path / "unsloth" / "python.exe")),
        ui=UiSettings(poll_interval_ms=1500),
    )
    SettingsService(settings_file=settings_file, defaults_file=DEFAULTS).save(original)

    # Simulate a restart: brand-new services reading the same files.
    loaded = SettingsService(settings_file=settings_file, defaults_file=DEFAULTS).load()
    assert loaded.paths.outputs_dir == folders["outputs_dir"]
    assert loaded.engines.unsloth_python == str(tmp_path / "unsloth" / "python.exe")
    assert loaded.ui.poll_interval_ms == 1500

    storage = StorageService(loaded)
    assert any(r.path == model_dir for r in ModelService(storage).scan_local_models())
    assert any(
        r.path == folders["datasets_dir"] / "prepared.jsonl"
        for r in DatasetService(storage).scan_local_datasets()
    )


def test_job_history_persists_but_active_session_is_not_restored(qapp, tmp_path: Path) -> None:
    """Terminal history survives restart; active worker state never does."""
    history_file = tmp_path / "jobs.json"
    manager = JobManager(history_file=history_file)
    assert manager.history() == []
    assert list(manager.terminal_jobs()) == []
    assert manager.snapshot("any-old-job") is None

    job = manager.submit("unsloth", model_ref="a-model")
    manager.mark_succeeded(job.job_id, message="adapter saved")
    assert len(list(manager.terminal_jobs())) == 1
    assert [record.job_id for record in manager.history()] == [job.job_id]

    # A fresh manager reloads only the persisted terminal summary.
    fresh = JobManager(history_file=history_file)
    assert [record.job_id for record in fresh.history()] == [job.job_id]
    assert fresh.history()[0].message == "adapter saved"
    assert list(fresh.terminal_jobs()) == []
    assert fresh.snapshot(job.job_id) is None
    assert fresh.gpu_owner is None

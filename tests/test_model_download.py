"""Tests for cooperative per-file model downloads and library refresh wiring.

All Hugging Face interactions are mocked: these tests must never touch the
network.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.state import AppState
from services.model_service import DownloadCancelled, ModelService
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService
from ui.models_tab import ModelDownloadWorker, ModelsTab

FILES = ["config.json", "model.safetensors", "tokenizer.json"]


def make_service(tmp_path: Path) -> ModelService:
    for name in ("models", "datasets", "outputs", "cache"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    return ModelService(StorageService(settings))


def install_hub_mocks(monkeypatch, files: list[str], on_download=None) -> list[str]:
    """Patch HfApi/hf_hub_download/snapshot_download; return downloaded names."""
    downloaded: list[str] = []

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = list(files)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: mock_api)

    def fake_download(*, repo_id, filename, repo_type, local_dir, token):
        downloaded.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub", encoding="utf-8")
        if on_download is not None:
            on_download(filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    def forbid_snapshot(*args, **kwargs):
        raise AssertionError("snapshot_download must not be used for cancellable downloads")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbid_snapshot)
    return downloaded


def test_download_model_downloads_each_file_with_honest_progress(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    downloaded = install_hub_mocks(monkeypatch, FILES)

    progress: list[tuple[float, str]] = []
    destination = tmp_path / "models" / "org--model"
    result = service.download_model(
        "org/model",
        destination=destination,
        progress_callback=lambda fraction, message: progress.append((fraction, message)),
    )

    assert result == destination
    assert downloaded == FILES
    for filename in FILES:
        assert (destination / filename).is_file()

    fractions = [round(fraction, 6) for fraction, _ in progress]
    assert set(fractions) == {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}
    assert fractions == sorted(fractions), "progress must be monotonic"
    # Fractions come from completed files / total files, never a made-up 0.1.
    assert 0.1 not in fractions
    assert any("1/3" in message for _, message in progress)
    assert any("2/3" in message for _, message in progress)


def test_download_cancel_stops_between_files_and_keeps_partial_files(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    state = {"calls": 0}

    def cancel_when_first_file_done() -> bool:
        return state["calls"] >= 1

    def on_download(_filename: str) -> None:
        state["calls"] += 1

    downloaded = install_hub_mocks(monkeypatch, FILES, on_download=on_download)
    progress: list[tuple[float, str]] = []
    destination = tmp_path / "models" / "org--model"

    with pytest.raises(DownloadCancelled) as exc_info:
        service.download_model(
            "org/model",
            destination=destination,
            progress_callback=lambda fraction, message: progress.append((fraction, message)),
            cancel=cancel_when_first_file_done,
        )

    assert downloaded == [FILES[0]]
    assert (destination / FILES[0]).is_file()
    assert not (destination / FILES[1]).exists()
    assert exc_info.value.downloaded == 1
    assert exc_info.value.total == len(FILES)
    assert "between files" in str(exc_info.value)
    assert any("Cancelling after 1/3 files" in message for _, message in progress)


def test_download_accepts_threading_event_for_cancellation(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    event = threading.Event()
    downloaded = install_hub_mocks(monkeypatch, FILES, on_download=lambda _name: event.set())

    with pytest.raises(DownloadCancelled) as exc_info:
        service.download_model("org/model", destination=tmp_path / "models" / "m", cancel=event)

    assert downloaded == [FILES[0]]
    assert exc_info.value.downloaded == 1


def test_download_model_keeps_signature_compatible(tmp_path, monkeypatch):
    """The original positional args still work without a cancel callback."""
    service = make_service(tmp_path)
    downloaded = install_hub_mocks(monkeypatch, FILES)
    destination = service.download_model("org/model", tmp_path / "models" / "m", None)
    assert downloaded == FILES
    assert destination.is_dir()


def test_models_tab_refreshes_asynchronously_after_model_changes(qapp, tmp_path, wait_for):
    for name in ("models", "datasets", "outputs", "cache"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )

    class StubService:
        def __init__(self) -> None:
            self.calls = 0

        def scan_local_models(self):
            self.calls += 1
            return []

    state = AppState()
    service = StubService()
    tab = ModelsTab(
        state, StorageService(settings), model_service=service, scan_on_init=False
    )

    assert tab._models_changed_connected is True
    state.publish_models([])
    assert wait_for(lambda: service.calls == 1 and tab._scan_worker is None)
    assert tab.library_table.rowCount() == 0

    # The tab's own publish must not trigger a recursive second scan.
    tab._scan_library(publish=True)
    assert wait_for(lambda: service.calls == 2 and tab._scan_worker is None)
    assert tab.library_table.rowCount() == 0


def test_cancel_button_requests_worker_cancellation(qapp, tmp_path, monkeypatch):
    for name in ("models", "datasets", "outputs", "cache"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    state = AppState()
    tab = ModelsTab(state, StorageService(settings))
    worker = ModelDownloadWorker(tab._service, "org/model", tab)
    tab._download_worker = worker
    tab.cancel_download_button.setEnabled(True)
    monkeypatch.setattr(worker, "isRunning", lambda: True)

    tab.cancel_download_button.click()

    assert worker._cancel_event.is_set()
    assert not tab.cancel_download_button.isEnabled()
    assert "Cancelling" in tab.download_status_label.text()



def test_download_rejects_empty_hub_repository(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    install_hub_mocks(monkeypatch, [])
    destination = tmp_path / "models" / "empty-model"

    with pytest.raises(ValueError, match="contains no files"):
        service.download_model("org/empty", destination=destination)

    assert not destination.exists()


def test_download_rejects_repository_without_model_artifacts(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    install_hub_mocks(monkeypatch, ["README.md", "tokenizer.json"])
    destination = tmp_path / "models" / "not-a-model"

    with pytest.raises(ValueError, match="recognised model configuration"):
        service.download_model("org/not-a-model", destination=destination)

    assert (destination / "README.md").is_file()
    assert (destination / "tokenizer.json").is_file()

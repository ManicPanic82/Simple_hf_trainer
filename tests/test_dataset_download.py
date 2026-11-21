"""Tests for Hub dataset download/import (service + UI worker).

All Hugging Face Hub network calls are mocked; these tests never download data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services.dataset_service import DatasetService
from services.settings_service import AppSettings
from services.storage_service import StorageService
from ui.datasets_tab import DatasetDownloadWorker


def _service(tmp_path: Path) -> DatasetService:
    return DatasetService(StorageService(AppSettings()))


def test_download_dataset_calls_snapshot_download_and_returns_path(tmp_path, monkeypatch) -> None:
    snapshot = MagicMock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    target = tmp_path / "nested" / "datasets" / "org--data"
    progress_events: list[tuple[float, str]] = []
    result = _service(tmp_path).download_dataset(
        "org/data",
        target,
        token="hf_test_token",
        progress=lambda frac, msg: progress_events.append((frac, msg)),
    )

    assert result == target
    assert target.is_dir()
    snapshot.assert_called_once_with(
        repo_id="org/data",
        local_dir=str(target),
        repo_type="dataset",
        token="hf_test_token",
    )
    assert [frac for frac, _ in progress_events] == [0.1, 1.0]


def test_download_dataset_uses_credentials_token_when_not_supplied(tmp_path, monkeypatch) -> None:
    snapshot = MagicMock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    credentials = MagicMock()
    credentials.available = True
    credentials.get_token.return_value = "hf_from_store"
    service = DatasetService(StorageService(AppSettings()), credentials)

    service.download_dataset("org/data", tmp_path / "target")

    assert snapshot.call_args.kwargs["token"] == "hf_from_store"


def test_download_dataset_validates_arguments(tmp_path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="repo_id"):
        service.download_dataset("   ", tmp_path / "target")
    with pytest.raises(ValueError, match="target directory"):
        service.download_dataset("org/data", "")


def test_download_dataset_raises_on_error(tmp_path, monkeypatch) -> None:
    snapshot = MagicMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)

    with pytest.raises(RuntimeError, match="Failed to download dataset org/data: offline"):
        _service(tmp_path).download_dataset("org/data", tmp_path / "target")


def _run_worker(qapp, wait_for, service: DatasetService, repo_id: str, target: Path):
    worker = DatasetDownloadWorker(service, repo_id, target)
    results: list[Path] = []
    errors: list[str] = []
    worker.download_finished.connect(results.append)
    worker.download_failed.connect(errors.append)
    worker.start()
    assert wait_for(lambda: bool(results) or bool(errors), timeout_ms=5_000)
    worker.wait(5_000)
    return results, errors


def test_download_worker_emits_result_signal(qapp, wait_for, tmp_path) -> None:
    class StubService:
        def download_dataset(self, repo_id, target_dir, progress=None):
            if progress:
                progress(0.5, "halfway")
            return target_dir

    target = tmp_path / "org--data"
    results, errors = _run_worker(qapp, wait_for, StubService(), "org/data", target)

    assert errors == []
    assert results == [target]


def test_download_worker_emits_error_signal(qapp, wait_for, tmp_path) -> None:
    class FailingService:
        def download_dataset(self, repo_id, target_dir, progress=None):
            raise RuntimeError("boom")

    results, errors = _run_worker(qapp, wait_for, FailingService(), "org/data", tmp_path / "target")

    assert results == []
    assert len(errors) == 1
    assert errors[0] == "boom"

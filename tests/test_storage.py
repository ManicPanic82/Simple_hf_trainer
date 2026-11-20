"""Storage path validation and safe directory creation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.settings_service import AppSettings, PathSettings
from services.storage_service import Severity, StorageService


def settings_with(**folders: Path) -> AppSettings:
    return AppSettings(paths=PathSettings(**folders))


def test_existing_directories_validate(app_settings: AppSettings) -> None:
    validation = StorageService(app_settings).validate_all()
    assert validation.ok
    assert all(result.exists for result in validation.results)
    assert validation.issues == []


def test_missing_directories_are_created(tmp_path: Path) -> None:
    folders = {
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "outputs_dir": tmp_path / "outputs",
        "hf_cache_dir": tmp_path / "cache",
    }
    service = StorageService(settings_with(**folders))

    before = service.validate_all()
    assert before.ok
    assert any(issue.severity is Severity.WARNING for issue in before.issues)

    after = service.ensure_directories()
    assert after.ok
    for path in folders.values():
        assert path.is_dir()
    assert len(after.created) == 4


def test_file_in_place_of_directory_is_an_error(tmp_path: Path) -> None:
    blocked = tmp_path / "models"
    blocked.write_text("i am a file", encoding="utf-8")
    service = StorageService(
        settings_with(
            models_dir=blocked,
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )

    result = service.validate("models_dir")
    assert result.ok is False
    assert any("not a directory" in issue.message for issue in result.issues)

    created = service.ensure_directories()
    assert created.ok is False
    assert blocked.read_text(encoding="utf-8") == "i am a file"


def test_relative_path_is_rejected() -> None:
    service = StorageService(
        settings_with(
            models_dir=Path("models"),
            datasets_dir=Path("datasets"),
            outputs_dir=Path("outputs"),
            hf_cache_dir=Path("cache"),
        )
    )
    result = service.validate("models_dir")
    assert result.ok is False
    assert any("absolute" in issue.message for issue in result.issues)


def test_unknown_label_is_rejected(app_settings: AppSettings) -> None:
    with pytest.raises(KeyError):
        StorageService(app_settings).path_for("downloads")


def test_unwritable_parent_is_reported(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "nested" / "models"
    service = StorageService(
        settings_with(
            models_dir=missing,
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    monkeypatch.setattr(StorageService, "_can_write", staticmethod(lambda _path: False))

    result = service.validate("models_dir")
    assert result.ok is False
    assert any("not writable" in issue.message for issue in result.issues)


def test_creation_failure_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "models"
    service = StorageService(
        settings_with(
            models_dir=missing,
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )

    real_mkdir = Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self == missing:
            raise PermissionError("denied by test")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    created = service.ensure_directories()
    assert created.ok is False
    assert any("Could not create directory" in issue.message for issue in created.issues)


def test_existing_data_is_never_deleted(tmp_path: Path) -> None:
    keep = tmp_path / "models" / "important.bin"
    keep.parent.mkdir()
    keep.write_bytes(b"payload")

    service = StorageService(
        settings_with(
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    service.ensure_directories()

    assert keep.read_bytes() == b"payload"


def test_settings_json_is_not_created_by_validation(tmp_path: Path, monkeypatch) -> None:
    settings_file = tmp_path / "settings.json"
    from services.settings_service import SettingsService

    service = SettingsService(
        settings_file=settings_file,
        defaults_file=Path(__file__).parents[1] / "config/defaults.json",
    )
    loaded = service.load()
    assert settings_file.is_file()
    StorageService(loaded).validate_all()
    json.loads(settings_file.read_text(encoding="utf-8"))

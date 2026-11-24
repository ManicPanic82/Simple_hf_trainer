"""Settings persistence, validation, and corruption handling."""

from __future__ import annotations

import json
import re
from pathlib import Path

from services.settings_service import (
    DEFAULT_POLL_INTERVAL_MS,
    AppSettings,
    EngineSettings,
    PathSettings,
    SettingsService,
    UiSettings,
    expand_vars,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = PROJECT_ROOT / "config" / "defaults.json"


def make_service(tmp_path: Path) -> SettingsService:
    return SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=DEFAULTS,
    )


def test_defaults_are_created_and_used(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_HF_TRAINER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HF_HOME", raising=False)

    service = make_service(tmp_path)
    settings = service.load()

    assert (tmp_path / "settings.json").is_file()
    assert any("defaults were created" in issue for issue in service.issues)
    assert settings.paths.models_dir == tmp_path / "home" / "models"
    assert settings.paths.datasets_dir == tmp_path / "home" / "datasets"
    assert settings.paths.outputs_dir == tmp_path / "home" / "outputs"
    assert settings.paths.hf_cache_dir == tmp_path / "home" / "hf-cache"
    assert settings.ui.poll_interval_ms == DEFAULT_POLL_INTERVAL_MS
    assert settings.schema_version == 1


def test_hf_home_environment_is_honoured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_HF_TRAINER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    settings = make_service(tmp_path).load()
    assert settings.paths.hf_cache_dir == tmp_path / "cache"


def test_roundtrip_persistence(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.load()

    wanted = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "m",
            datasets_dir=tmp_path / "d",
            outputs_dir=tmp_path / "o",
            hf_cache_dir=tmp_path / "c",
        ),
        engines=EngineSettings(
            unsloth_python=str(tmp_path / "unsloth.exe"),
            heretic_python=str(tmp_path / "heretic.exe"),
        ),
        ui=UiSettings(poll_interval_ms=3000),
    )
    service.save(wanted)
    assert not (tmp_path / "settings.json.tmp").exists()

    reloaded = service.load()
    assert service.issues == []
    assert reloaded.paths == wanted.paths
    assert reloaded.engines == wanted.engines
    assert reloaded.ui == wanted.ui


def test_corrupt_json_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_HF_TRAINER_HOME", str(tmp_path / "home"))
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")

    service = make_service(tmp_path)
    settings = service.load()

    assert settings.paths.models_dir == tmp_path / "home" / "models"
    assert any("could not be read" in issue for issue in service.issues)


def test_relative_path_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_HF_TRAINER_HOME", str(tmp_path / "home"))
    (tmp_path / "settings.json").write_text(
        json.dumps({"paths": {"models_dir": "models"}}), encoding="utf-8"
    )

    service = make_service(tmp_path)
    settings = service.load()

    assert settings.paths.models_dir == tmp_path / "home" / "models"
    assert any("absolute" in issue for issue in service.issues)


def test_wrong_types_fall_back_per_field(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SIMPLE_HF_TRAINER_HOME", str(tmp_path / "home"))
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "paths": {"models_dir": 42},
                "engines": {"unsloth_python": 7},
                "ui": {"poll_interval_ms": "fast"},
            }
        ),
        encoding="utf-8",
    )

    service = make_service(tmp_path)
    settings = service.load()

    assert settings.paths.models_dir == tmp_path / "home" / "models"
    assert settings.engines.unsloth_python == ""
    assert settings.ui.poll_interval_ms == DEFAULT_POLL_INTERVAL_MS
    assert len([issue for issue in service.issues if "must be" in issue]) == 3


def test_out_of_range_poll_interval_is_clamped(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(
        json.dumps({"ui": {"poll_interval_ms": 10}}), encoding="utf-8"
    )
    service = make_service(tmp_path)
    settings = service.load()
    assert settings.ui.poll_interval_ms == DEFAULT_POLL_INTERVAL_MS
    assert any("outside" in issue for issue in service.issues)


def test_unknown_sections_are_reported_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(json.dumps({"mystery": {"a": 1}}), encoding="utf-8")
    service = make_service(tmp_path)
    service.load()
    assert any("Ignored unknown settings section" in issue for issue in service.issues)


def test_schema_version_mismatch_is_reported(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    service = make_service(tmp_path)
    service.load()
    assert any("schema_version" in issue for issue in service.issues)


def test_saved_settings_never_contain_credentials(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.save(AppSettings())
    text = (tmp_path / "settings.json").read_text(encoding="utf-8").lower()
    assert "token" not in text
    assert "secret" not in text
    assert "password" not in text
    assert re.search(r"hf_[a-z0-9]{8,}", text) is None


def test_expand_vars_handles_unknown_variables() -> None:
    assert expand_vars("${DEFINITELY_NOT_SET}/x", {}) == "${DEFINITELY_NOT_SET}/x"
    assert Path(expand_vars("${HOME}/models", {"USERPROFILE": r"C:\Users\Test"})) == Path(
        r"C:\Users\Test\models"
    )

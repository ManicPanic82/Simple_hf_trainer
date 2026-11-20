"""Engine interpreter detection without importing the engines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.engine_registry import (
    EngineRegistry,
    EngineState,
    EngineStatus,
    parse_probe_output,
)
from services.hardware_monitor import CommandResult
from services.settings_service import AppSettings, EngineSettings


def fake_python(tmp_path: Path, name: str = "python.exe") -> Path:
    path = tmp_path / name
    path.write_text("not a real interpreter", encoding="utf-8")
    return path


def registry_with(
    tmp_path: Path, runner, *, unsloth: str = "", heretic: str = ""
) -> EngineRegistry:
    settings = AppSettings(engines=EngineSettings(unsloth_python=unsloth, heretic_python=heretic))
    return EngineRegistry(settings, repository_root=tmp_path / "repo", runner=runner)


def test_missing_interpreter_is_reported_clearly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SIMPLE_HF_TRAINER_UNSLOTH_PYTHON", raising=False)
    registry = registry_with(tmp_path, runner=pytest.fail)
    status = registry.probe("unsloth")
    assert status.state is EngineState.INTERPRETER_MISSING
    assert "Checked:" in status.detail
    assert status.is_available is False


def test_available_engine_reports_version(tmp_path: Path) -> None:
    interpreter = fake_python(tmp_path)
    calls: list[list[str]] = []

    def runner(args, timeout):
        calls.append(list(args))
        return CommandResult(0, json.dumps({"found": True, "version": "2026.11.7"}), "")

    registry = registry_with(tmp_path, runner, unsloth=str(interpreter))
    status = registry.probe("unsloth")

    assert status.state is EngineState.AVAILABLE
    assert status.version == "2026.11.7"
    assert status.interpreter == interpreter
    assert calls[0][0] == str(interpreter)
    assert calls[0][1] == "-c"


def test_package_present_without_version_metadata(tmp_path: Path) -> None:
    def runner(args, timeout):
        return CommandResult(0, json.dumps({"found": True, "version": None}), "")

    registry = registry_with(tmp_path, runner, unsloth=str(fake_python(tmp_path)))
    status = registry.probe("unsloth")
    assert status.state is EngineState.AVAILABLE
    assert status.version is None
    assert "is installed" in status.detail


def test_package_missing_is_not_installed(tmp_path: Path) -> None:
    def runner(args, timeout):
        return CommandResult(0, json.dumps({"found": False, "version": None}), "")

    registry = registry_with(tmp_path, runner, heretic=str(fake_python(tmp_path)))
    status = registry.probe("heretic")
    assert status.state is EngineState.NOT_INSTALLED
    assert "not installed" in status.detail


def test_probe_failure_is_an_error_not_a_crash(tmp_path: Path) -> None:
    def runner(args, timeout):
        return CommandResult(1, "", "ModuleNotFoundError: boom")

    registry = registry_with(tmp_path, runner, unsloth=str(fake_python(tmp_path)))
    status = registry.probe("unsloth")
    assert status.state is EngineState.ERROR
    assert "boom" in status.detail


def test_unreadable_probe_output_is_an_error(tmp_path: Path) -> None:
    def runner(args, timeout):
        return CommandResult(0, "hello there", "")

    registry = registry_with(tmp_path, runner, unsloth=str(fake_python(tmp_path)))
    status = registry.probe("unsloth")
    assert status.state is EngineState.ERROR
    assert "Unreadable probe output" in status.detail


def test_interpreter_that_cannot_start(tmp_path: Path) -> None:
    interpreter = fake_python(tmp_path)

    def runner(args, timeout):
        raise FileNotFoundError("gone")

    registry = registry_with(tmp_path, runner, unsloth=str(interpreter))
    status = registry.probe("unsloth")
    assert status.state is EngineState.INTERPRETER_MISSING
    assert "could not be started" in status.detail


def test_standard_layout_is_discovered_automatically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SIMPLE_HF_TRAINER_HERETIC_PYTHON", raising=False)
    repo_root = tmp_path / "repo"
    expected = repo_root / "environments" / "heretic" / "Scripts" / "python.exe"
    expected.parent.mkdir(parents=True)
    expected.write_text("dummy", encoding="utf-8")

    def runner(args, timeout):
        return CommandResult(0, json.dumps({"found": True, "version": "2.0"}), "")

    settings = AppSettings()
    registry = EngineRegistry(settings, repository_root=repo_root, runner=runner)
    status = registry.probe("heretic")
    assert status.interpreter == expected
    assert status.state is EngineState.AVAILABLE


def test_refresh_reports_all_engines_and_emits(qapp, tmp_path: Path) -> None:
    def runner(args, timeout):
        return CommandResult(0, json.dumps({"found": False, "version": None}), "")

    api = fake_python(tmp_path, "unsloth.exe")
    registry = registry_with(tmp_path, runner, unsloth=str(api), heretic=str(api))
    seen: list[dict[str, EngineStatus]] = []
    registry.statuses_changed.connect(seen.append)

    statuses = registry.refresh()

    assert set(statuses) == {"unsloth", "heretic"}
    assert seen and set(seen[-1]) == {"unsloth", "heretic"}
    assert registry.status("unsloth").state is EngineState.NOT_INSTALLED


def test_untouched_engine_is_unchecked(tmp_path: Path) -> None:
    registry = registry_with(tmp_path, runner=pytest.fail)
    assert registry.status("unsloth").state is EngineState.UNCHECKED


def test_parse_probe_output_validates_shape() -> None:
    with pytest.raises(ValueError):
        parse_probe_output("")
    with pytest.raises(ValueError):
        parse_probe_output(json.dumps({"version": "1"}))
    with pytest.raises(ValueError):
        parse_probe_output(json.dumps({"found": "yes"}))

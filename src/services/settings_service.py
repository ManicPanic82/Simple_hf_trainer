"""Persistent application settings.

Settings are ordinary local JSON. Secrets are *not* part of this schema; Hugging
Face tokens live in the Windows Credential Manager (see
``services.credentials``). Loading is defensive: a corrupt, partial, or
hand-edited file degrades to defaults with an explicit issue list instead of
crashing or silently discarding user intent.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app import paths

SCHEMA_VERSION = 1
DEFAULT_POLL_INTERVAL_MS = 2000
MIN_POLL_INTERVAL_MS = 500
MAX_POLL_INTERVAL_MS = 30_000

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class PathSettings:
    """User-configurable storage locations (absolute paths)."""

    models_dir: Path
    datasets_dir: Path
    outputs_dir: Path
    hf_cache_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "models_dir": str(self.models_dir),
            "datasets_dir": str(self.datasets_dir),
            "outputs_dir": str(self.outputs_dir),
            "hf_cache_dir": str(self.hf_cache_dir),
        }

    def items(self) -> list[tuple[str, Path]]:
        return [
            ("models_dir", self.models_dir),
            ("datasets_dir", self.datasets_dir),
            ("outputs_dir", self.outputs_dir),
            ("hf_cache_dir", self.hf_cache_dir),
        ]


@dataclass(frozen=True)
class EngineSettings:
    """Interpreter paths for the isolated engine environments (may be blank)."""

    unsloth_python: str = ""
    heretic_python: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "unsloth_python": self.unsloth_python,
            "heretic_python": self.heretic_python,
        }


@dataclass(frozen=True)
class UiSettings:
    poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS

    def as_dict(self) -> dict[str, int]:
        return {"poll_interval_ms": self.poll_interval_ms}


@dataclass(frozen=True)
class AppSettings:
    schema_version: int = SCHEMA_VERSION
    paths: PathSettings = field(default_factory=lambda: default_paths())
    engines: EngineSettings = field(default_factory=EngineSettings)
    ui: UiSettings = field(default_factory=UiSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "paths": self.paths.as_dict(),
            "engines": self.engines.as_dict(),
            "ui": self.ui.as_dict(),
        }


def default_paths(
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PathSettings:
    environ = os.environ if environ is None else environ
    base = Path(home) if home is not None else paths.default_home()
    hf_home = environ.get("HF_HOME")
    cache = Path(hf_home) if hf_home else base / "hf-cache"
    return PathSettings(
        models_dir=base / "models",
        datasets_dir=base / "datasets",
        outputs_dir=base / "outputs",
        hf_cache_dir=cache,
    )


def _app_home(environ: Mapping[str, str]) -> Path:
    override = environ.get(paths.HOME_ENV_VAR)
    if override:
        return Path(override)
    return Path(environ.get("USERPROFILE") or Path.home()) / paths.APP_DIR_NAME


def expand_vars(value: str, environ: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` and ``%VAR%`` forms; unknown variables are preserved."""
    environ = os.environ if environ is None else environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "HOME":
            return environ.get("USERPROFILE") or str(Path.home())
        if name == "APP_HOME":
            return str(_app_home(environ))
        if name == "HF_HOME":
            return environ.get("HF_HOME") or str(_app_home(environ) / "hf-cache")
        return environ.get(name, match.group(0))

    expanded = _VAR_PATTERN.sub(replace, value)
    return os.path.expandvars(expanded)


class SettingsService:
    """Loads, validates, and atomically persists :class:`AppSettings`."""

    def __init__(
        self,
        settings_file: Path | None = None,
        defaults_file: Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._settings_file = Path(settings_file) if settings_file else paths.settings_path()
        self._defaults_file = (
            Path(defaults_file)
            if defaults_file
            else paths.repository_root() / "config/defaults.json"
        )
        self._environ = dict(os.environ) if environ is None else environ
        self.issues: list[str] = []

    @property
    def settings_file(self) -> Path:
        return self._settings_file

    # -- loading ---------------------------------------------------------

    def load(self) -> AppSettings:
        self.issues = []
        raw_defaults = self._read_defaults()
        file_data: dict[str, Any] = {}

        if self._settings_file.exists():
            try:
                text = self._settings_file.read_text(encoding="utf-8")
                parsed = json.loads(text)
            except (OSError, ValueError) as error:
                self.issues.append(f"Settings file could not be read ({error}); using defaults.")
            else:
                if isinstance(parsed, dict):
                    file_data = parsed
                else:
                    self.issues.append("Settings file is not a JSON object; using defaults.")
        else:
            self.issues.append("No settings file yet; defaults were created.")
            try:
                self._write_json(self._settings_file, raw_defaults)
            except OSError as error:
                self.issues.append(f"Could not create settings file: {error}")

        settings = self._merge(raw_defaults, file_data)
        return settings

    def _read_defaults(self) -> dict[str, Any]:
        default_paths_dict = default_paths(environ=self._environ).as_dict()
        builtin: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "paths": default_paths_dict,
            "engines": EngineSettings().as_dict(),
            "ui": UiSettings().as_dict(),
        }
        if not self._defaults_file.exists():
            self.issues.append(
                f"Defaults file missing at {self._defaults_file}; built-in defaults used."
            )
            return builtin
        try:
            parsed = json.loads(self._defaults_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.issues.append(
                f"Defaults file could not be read ({error}); built-in defaults used."
            )
            return builtin
        if not isinstance(parsed, dict):
            self.issues.append("Defaults file is not a JSON object; built-in defaults used.")
            return builtin

        merged = _deep_merge(builtin, parsed)
        # Defaults use placeholders; resolve them with the current environment.
        for key, value in list(merged["paths"].items()):
            if isinstance(value, str):
                expanded = expand_vars(value, self._environ)
                merged["paths"][key] = str(Path(expanded)) if expanded else expanded
        return merged

    def _merge(self, defaults: dict[str, Any], data: dict[str, Any]) -> AppSettings:
        known_sections = {"schema_version", "paths", "engines", "ui"}
        for key in data:
            if key not in known_sections:
                self.issues.append(f"Ignored unknown settings section '{key}'.")

        version = data.get("schema_version", defaults.get("schema_version", SCHEMA_VERSION))
        if not isinstance(version, int) or isinstance(version, bool):
            self.issues.append("schema_version is not an integer; assuming current version.")
            version = SCHEMA_VERSION
        elif version != SCHEMA_VERSION:
            self.issues.append(
                f"Settings schema_version {version} differs from {SCHEMA_VERSION}; "
                "known keys were kept without migration."
            )

        file_paths = data.get("paths") if isinstance(data.get("paths"), dict) else {}
        default_paths_section = defaults.get("paths", {})
        resolved: dict[str, Path] = {}
        for key in ("models_dir", "datasets_dir", "outputs_dir", "hf_cache_dir"):
            fallback = Path(str(default_paths_section.get(key, "")))
            raw = file_paths.get(key)
            if raw is None:
                resolved[key] = fallback
                continue
            if not isinstance(raw, str) or not raw.strip():
                self.issues.append(f"paths.{key} must be a non-empty string; default used.")
                resolved[key] = fallback
                continue
            expanded = expand_vars(raw, self._environ)
            candidate = Path(expanded)
            if not candidate.is_absolute():
                self.issues.append(f"paths.{key} must be an absolute path; default used.")
                resolved[key] = fallback
                continue
            resolved[key] = candidate

        file_engines = data.get("engines") if isinstance(data.get("engines"), dict) else {}
        engine_values: dict[str, str] = {}
        for key in ("unsloth_python", "heretic_python"):
            raw = file_engines.get(key, "")
            if raw is None:
                raw = ""
            if not isinstance(raw, str):
                self.issues.append(f"engines.{key} must be a string; ignored.")
                raw = ""
            engine_values[key] = raw.strip()

        file_ui = data.get("ui") if isinstance(data.get("ui"), dict) else {}
        poll = file_ui.get("poll_interval_ms", DEFAULT_POLL_INTERVAL_MS)
        if isinstance(poll, bool) or not isinstance(poll, int):
            self.issues.append("ui.poll_interval_ms must be an integer; default used.")
            poll = DEFAULT_POLL_INTERVAL_MS
        elif not MIN_POLL_INTERVAL_MS <= poll <= MAX_POLL_INTERVAL_MS:
            self.issues.append(
                "ui.poll_interval_ms outside "
                f"{MIN_POLL_INTERVAL_MS}-{MAX_POLL_INTERVAL_MS}; default used."
            )
            poll = DEFAULT_POLL_INTERVAL_MS

        return AppSettings(
            schema_version=SCHEMA_VERSION,
            paths=PathSettings(**resolved),
            engines=EngineSettings(
                unsloth_python=engine_values["unsloth_python"],
                heretic_python=engine_values["heretic_python"],
            ),
            ui=UiSettings(poll_interval_ms=poll),
        )

    # -- saving ----------------------------------------------------------

    def save(self, settings: AppSettings) -> None:
        """Persist settings atomically. Raises ``OSError`` if the write fails."""
        self._write_json(self._settings_file, settings.to_dict())

    @staticmethod
    def _write_json(target: Path, payload: dict[str, Any]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result

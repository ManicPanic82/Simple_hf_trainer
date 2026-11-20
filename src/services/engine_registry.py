"""Detection of the optional ML engine environments.

The application environment deliberately does not contain Unsloth or Heretic.
Their interpreters are configured separately and probed out-of-process, so a
broken or missing engine can never prevent the GUI from starting.

Detection is metadata-only: it asks the engine interpreter whether the package
is importable and which version is installed. It does not import torch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app import paths
from services.hardware_monitor import CommandRunner, run_command
from services.settings_service import AppSettings

PROBE_TIMEOUT_S = 30.0
UNSLOTH = "unsloth"
HERETIC = "heretic"
ENGINE_IDS = (UNSLOTH, HERETIC)
_ENGINE_MODULES = {UNSLOTH: "unsloth", HERETIC: "heretic"}
# The importable module name is not always the distribution name (heretic-llm).
_ENGINE_DISTRIBUTIONS = {UNSLOTH: ("unsloth",), HERETIC: ("heretic-llm", "heretic")}
_ENV_VARS = {
    UNSLOTH: "SIMPLE_HF_TRAINER_UNSLOTH_PYTHON",
    HERETIC: "SIMPLE_HF_TRAINER_HERETIC_PYTHON",
}
_DISPLAY_NAMES = {UNSLOTH: "Unsloth", HERETIC: "Heretic"}

PROBE_SCRIPT = """
import importlib.util
import importlib.metadata as metadata
import json
import sys

module = sys.argv[1]
distributions = [name for name in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if name]
try:
    found = importlib.util.find_spec(module) is not None
except Exception as error:
    print(json.dumps({"found": False, "version": None, "error": repr(error)}))
    raise SystemExit(0)

version = None
if found:
    for name in [module, *distributions]:
        try:
            version = metadata.version(name)
            break
        except Exception:
            continue
print(json.dumps({"found": found, "version": version}))
"""


class EngineState(StrEnum):
    UNCHECKED = "unchecked"
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    INTERPRETER_MISSING = "interpreter_missing"
    ERROR = "error"


@dataclass(frozen=True)
class EngineStatus:
    engine: str
    state: EngineState
    interpreter: Path | None = None
    detail: str = ""
    version: str | None = None

    @property
    def display_name(self) -> str:
        return _DISPLAY_NAMES.get(self.engine, self.engine.title())

    @property
    def is_available(self) -> bool:
        return self.state is EngineState.AVAILABLE

    def badge(self) -> str:
        return {
            EngineState.AVAILABLE: "available",
            EngineState.NOT_INSTALLED: "not installed",
            EngineState.INTERPRETER_MISSING: "not set up",
            EngineState.ERROR: "check failed",
            EngineState.UNCHECKED: "not checked",
        }[self.state]


@dataclass(frozen=True)
class _ProbeResult:
    found: bool
    version: str | None
    error: str | None = None


def parse_probe_output(stdout: str) -> _ProbeResult:
    """Parse the JSON line printed by :data:`PROBE_SCRIPT`."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("probe produced no output")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict) or "found" not in payload:
        raise ValueError("probe output is missing the 'found' field")
    found = payload.get("found")
    if not isinstance(found, bool):
        raise ValueError("probe 'found' field is not a boolean")
    version = payload.get("version")
    error = payload.get("error")
    return _ProbeResult(
        found=found,
        version=version if isinstance(version, str) else None,
        error=error if isinstance(error, str) else None,
    )


class EngineRegistry(QObject):
    """Resolves and probes the isolated engine interpreters."""

    statuses_changed = Signal(object)  # dict[str, EngineStatus]
    probe_started = Signal()

    def __init__(
        self,
        settings: AppSettings,
        *,
        repository_root: Path | None = None,
        runner: CommandRunner = run_command,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._repository_root = (
            Path(repository_root) if repository_root else paths.repository_root()
        )
        self._runner = runner
        self._statuses: dict[str, EngineStatus] = {
            engine: EngineStatus(engine, EngineState.UNCHECKED) for engine in ENGINE_IDS
        }
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pending = False

    # -- interpreter resolution -----------------------------------------

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._statuses = {
            engine: EngineStatus(engine, EngineState.UNCHECKED) for engine in ENGINE_IDS
        }

    def configured_interpreter(self, engine: str) -> str:
        if engine == UNSLOTH:
            return self._settings.engines.unsloth_python
        return self._settings.engines.heretic_python

    def default_interpreter(self, engine: str) -> Path:
        return self._repository_root / "environments" / engine / "Scripts" / "python.exe"

    def interpreter_candidates(self, engine: str) -> list[Path]:
        candidates: list[Path] = []
        configured = self.configured_interpreter(engine).strip()
        if configured:
            candidates.append(Path(configured))
        env_value = os.environ.get(_ENV_VARS[engine], "").strip()
        if env_value:
            candidates.append(Path(env_value))
        candidates.append(self.default_interpreter(engine))
        return candidates

    def resolve_interpreter(self, engine: str) -> tuple[Path | None, str]:
        candidates = self.interpreter_candidates(engine)
        for candidate in candidates:
            if candidate.is_file():
                return candidate, ""
        tried = ", ".join(str(candidate) for candidate in candidates)
        return None, f"No engine interpreter found. Checked: {tried}"

    # -- probing ---------------------------------------------------------

    def status(self, engine: str) -> EngineStatus:
        return self._statuses.get(engine, EngineStatus(engine, EngineState.UNCHECKED))

    def statuses(self) -> dict[str, EngineStatus]:
        return dict(self._statuses)

    def probe(self, engine: str) -> EngineStatus:
        interpreter, detail = self.resolve_interpreter(engine)
        if interpreter is None:
            status = EngineStatus(engine, EngineState.INTERPRETER_MISSING, None, detail)
        else:
            status = self._probe_package(engine, interpreter)
        self._statuses[engine] = status
        return status

    def _probe_package(self, engine: str, interpreter: Path) -> EngineStatus:
        module = _ENGINE_MODULES[engine]
        distributions = ",".join(_ENGINE_DISTRIBUTIONS[engine])
        try:
            result = self._runner(
                [str(interpreter), "-c", PROBE_SCRIPT, module, distributions], PROBE_TIMEOUT_S
            )
        except FileNotFoundError:
            return EngineStatus(
                engine,
                EngineState.INTERPRETER_MISSING,
                interpreter,
                f"Engine interpreter could not be started: {interpreter}",
            )
        except Exception as error:  # noqa: BLE001 - never propagate to the GUI
            return EngineStatus(
                engine, EngineState.ERROR, interpreter, f"Engine probe failed: {error}"
            )

        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            reason = message[-1] if message else f"exit code {result.returncode}"
            return EngineStatus(
                engine, EngineState.ERROR, interpreter, f"Engine probe failed: {reason}"[:300]
            )

        try:
            probe = parse_probe_output(result.stdout)
        except (ValueError, json.JSONDecodeError) as error:
            return EngineStatus(
                engine, EngineState.ERROR, interpreter, f"Unreadable probe output: {error}"
            )

        if probe.error:
            return EngineStatus(
                engine,
                EngineState.ERROR,
                interpreter,
                f"Engine import check failed: {probe.error}"[:300],
            )
        if not probe.found:
            return EngineStatus(
                engine,
                EngineState.NOT_INSTALLED,
                interpreter,
                f"{module} is not installed in this interpreter.",
            )
        return EngineStatus(
            engine,
            EngineState.AVAILABLE,
            interpreter,
            f"{module} {probe.version}" if probe.version else f"{module} is installed.",
            version=probe.version,
        )

    def refresh(self) -> dict[str, EngineStatus]:
        for engine in ENGINE_IDS:
            self.probe(engine)
        self.statuses_changed.emit(self.statuses())
        return self.statuses()

    def refresh_async(self) -> None:
        if self._pending:
            return
        self._pending = True
        self.probe_started.emit()
        self._pool.start(_RefreshTask(self))

    def wait_for_idle(self, timeout_ms: int = 5000) -> None:
        self._pool.waitForDone(timeout_ms)


class _RefreshTask(QRunnable):
    def __init__(self, registry: EngineRegistry) -> None:
        super().__init__()
        self._registry = registry

    def run(self) -> None:  # pragma: no cover - covered through the GUI
        try:
            self._registry.refresh()
        finally:
            self._registry._pending = False

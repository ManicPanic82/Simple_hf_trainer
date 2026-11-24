"""Hardware monitoring with graceful degradation.

The GUI must never crash because a monitoring library, driver, or CLI tool is
missing. Every probe returns a typed snapshot where unavailable values are
``None`` and the reason is carried in a short error string.

GPU facts are read from ``nvidia-smi`` (present with any modern NVIDIA driver)
rather than from an optional Python package. System RAM is read from ``psutil``
when it is installed.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

NVML_QUERY = "name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu"
GPU_ERROR_LIMIT = 240


class HardwareProbeError(RuntimeError):
    """Raised when a probe cannot produce a value."""


@dataclass(frozen=True)
class GpuStats:
    name: str
    driver_version: str | None = None
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class HardwareSnapshot:
    gpu: GpuStats | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None
    gpu_error: str | None = None
    ram_error: str | None = None
    captured_at: float = 0.0

    @property
    def gpu_available(self) -> bool:
        return self.gpu is not None and self.gpu_error is None

    @property
    def vram_percent(self) -> float | None:
        gpu = self.gpu
        if not gpu or not gpu.memory_total_mb or gpu.memory_used_mb is None:
            return None
        return max(0.0, min(100.0, gpu.memory_used_mb / gpu.memory_total_mb * 100.0))

    @property
    def ram_percent(self) -> float | None:
        if not self.ram_total_bytes or self.ram_used_bytes is None:
            return None
        return max(0.0, min(100.0, self.ram_used_bytes / self.ram_total_bytes * 100.0))


def _number(raw: str) -> float | None:
    """Parse ``nvidia-smi`` numeric fields, tolerating N/A style placeholders."""
    text = raw.strip()
    if not text or text.upper().startswith("N/A") or text.startswith("["):
        return None
    for suffix in ("MiB", "%", "C", "MB"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str) -> GpuStats:
    """Parse ``--format=csv,noheader,nounits`` output for one or more GPUs.

    Only the first GPU row is used: the application targets a single GPU.
    """
    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        raise HardwareProbeError("nvidia-smi returned no GPU information.")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) < 2 or not fields[0]:
        raise HardwareProbeError("nvidia-smi output was not in the expected CSV format.")

    def optional(index: int) -> float | None:
        return _number(fields[index]) if index < len(fields) else None

    return GpuStats(
        name=fields[0],
        driver_version=fields[1] or None,
        memory_used_mb=optional(2),
        memory_total_mb=optional(3),
        utilization_percent=optional(4),
        temperature_c=optional(5),
    )


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], float], CommandResult]


def run_command(args: list[str], timeout: float) -> CommandResult:
    """Run a short-lived console command without flashing a console window."""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=creation_flags,
    )
    return CommandResult(completed.returncode, completed.stdout or "", completed.stderr or "")


class GpuProbe(Protocol):
    def read(self) -> GpuStats: ...


class RamProbe(Protocol):
    def read(self) -> tuple[int, int]: ...


class NvidiaSmiProbe:
    """Reads GPU utilisation and memory from the NVIDIA driver CLI."""

    def __init__(
        self,
        executable: str = "nvidia-smi",
        *,
        timeout: float = 2.5,
        runner: CommandRunner = run_command,
    ) -> None:
        self._executable = executable
        self._timeout = timeout
        self._runner = runner

    def read(self) -> GpuStats:
        executable = shutil.which(self._executable) or self._executable
        args = [
            executable,
            f"--query-gpu={NVML_QUERY}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = self._runner(args, self._timeout)
        except FileNotFoundError as error:
            raise HardwareProbeError(
                "nvidia-smi was not found. Install or repair the NVIDIA driver."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise HardwareProbeError("nvidia-smi timed out; GPU state is unknown.") from error
        except OSError as error:
            raise HardwareProbeError(f"nvidia-smi could not be started: {error}") from error

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[0] if detail else "unknown error"
            raise HardwareProbeError(f"nvidia-smi failed: {message}")
        return parse_nvidia_smi_csv(result.stdout)


class PsutilRamProbe:
    """Reads system RAM usage through psutil when it is available."""

    def read(self) -> tuple[int, int]:
        try:
            import psutil
        except ImportError as error:
            raise HardwareProbeError(
                "psutil is not installed; RAM usage is unavailable."
            ) from error
        memory = psutil.virtual_memory()
        return int(memory.used), int(memory.total)


def collect_snapshot(
    gpu_probe: GpuProbe | None = None,
    ram_probe: RamProbe | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> HardwareSnapshot:
    """Collect one snapshot, converting probe failures into error strings."""
    gpu: GpuStats | None = None
    gpu_error: str | None = None
    try:
        gpu = (gpu_probe or NvidiaSmiProbe()).read()
    except HardwareProbeError as error:
        gpu_error = str(error)[:GPU_ERROR_LIMIT]
    except Exception as error:  # noqa: BLE001 - monitoring must never crash the UI
        gpu_error = f"Unexpected GPU probe failure: {type(error).__name__}: {error}"[
            :GPU_ERROR_LIMIT
        ]

    ram_used: int | None = None
    ram_total: int | None = None
    ram_error: str | None = None
    try:
        ram_used, ram_total = (ram_probe or PsutilRamProbe()).read()
    except HardwareProbeError as error:
        ram_error = str(error)[:GPU_ERROR_LIMIT]
    except Exception as error:  # noqa: BLE001
        ram_error = f"Unexpected RAM probe failure: {type(error).__name__}: {error}"[
            :GPU_ERROR_LIMIT
        ]

    return HardwareSnapshot(
        gpu=gpu,
        ram_used_bytes=ram_used,
        ram_total_bytes=ram_total,
        gpu_error=gpu_error,
        ram_error=ram_error,
        captured_at=clock(),
    )


class _SnapshotSignals(QObject):
    snapshot_ready = Signal(object)


class _SnapshotTask(QRunnable):
    def __init__(self, signals: _SnapshotSignals, gpu_probe, ram_probe) -> None:
        super().__init__()
        self._signals = signals
        self._gpu_probe = gpu_probe
        self._ram_probe = ram_probe

    def run(self) -> None:  # pragma: no cover - exercised through the GUI
        snapshot = collect_snapshot(self._gpu_probe, self._ram_probe)
        self._signals.snapshot_ready.emit(snapshot)


class HardwareMonitor(QObject):
    """Polls hardware state off the GUI thread.

    ``refresh()`` is a no-op while a probe is still running, so a slow or
    hanging driver cannot build up a backlog of work.
    """

    snapshot_ready = Signal(object)
    probe_started = Signal()

    def __init__(
        self,
        gpu_probe: GpuProbe | None = None,
        ram_probe: RamProbe | None = None,
        *,
        interval_ms: int = 2000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._gpu_probe = gpu_probe
        self._ram_probe = ram_probe
        self._interval_ms = max(500, int(interval_ms))
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _SnapshotSignals(self)
        self._signals.snapshot_ready.connect(self._on_snapshot)
        self._timer: QTimer | None = None
        self._pending = False
        self.last_snapshot: HardwareSnapshot | None = None

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = max(500, int(interval_ms))
        if self._timer is not None:
            self._timer.setInterval(self._interval_ms)

    def start(self) -> None:
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(self._interval_ms)
            self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        self._pool.waitForDone(3000)

    @property
    def running(self) -> bool:
        return self._timer is not None and self._timer.isActive()

    def refresh(self) -> None:
        if self._pending:
            return
        self._pending = True
        self.probe_started.emit()
        self._pool.start(_SnapshotTask(self._signals, self._gpu_probe, self._ram_probe))

    def _on_snapshot(self, snapshot: HardwareSnapshot) -> None:
        self._pending = False
        self.last_snapshot = snapshot
        self.snapshot_ready.emit(snapshot)

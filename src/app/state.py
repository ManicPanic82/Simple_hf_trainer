"""Shared observable application state for the GUI.

Services publish data here; widgets subscribe. No Qt widget ever talks to a
service directly, which keeps the UI testable and the data flow one-way.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from services.hardware_monitor import HardwareSnapshot
from workers.job_manager import JobSnapshot


@dataclass(frozen=True)
class Notice:
    """A user-facing message shown in the main window notice area."""

    level: str  # "info" | "warning" | "error"
    message: str


class AppState(QObject):
    """Single source of truth for cross-tab status."""

    hardware_changed = Signal(object)  # HardwareSnapshot
    engines_changed = Signal(object)  # dict[str, EngineStatus]
    job_changed = Signal(object)  # JobSnapshot
    notice = Signal(object)  # Notice
    models_changed = Signal(object)  # list[LocalModelRecord]
    datasets_changed = Signal(object)  # list[LocalDatasetRecord]
    metric_changed = Signal(object)  # dict[str, object]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.hardware: HardwareSnapshot | None = None
        self.engines: dict[str, object] = {}
        self.job: JobSnapshot | None = None
        self.models: list[object] = []
        self.datasets: list[object] = []
        self.metric_history: list[dict[str, object]] = []
        self.hardware_history: list[HardwareSnapshot] = []

    def publish_hardware(self, snapshot: HardwareSnapshot) -> None:
        self.hardware = snapshot
        self.hardware_history.append(snapshot)
        del self.hardware_history[:-1200]
        self.hardware_changed.emit(snapshot)

    def publish_metric(self, name: str, step: int, value: float, job_id: str | None = None) -> None:
        """Publish a genuine numeric metric from a worker and retain bounded history."""
        item: dict[str, object] = {
            "name": str(name), "step": int(step), "value": float(value), "job_id": job_id
        }
        self.metric_history.append(item)
        del self.metric_history[:-10000]
        self.metric_changed.emit(item)

    def publish_engines(self, statuses: dict[str, object]) -> None:
        self.engines = statuses
        self.engines_changed.emit(statuses)

    def publish_job(self, snapshot: JobSnapshot) -> None:
        self.job = snapshot
        self.job_changed.emit(snapshot)

    def publish_models(self, models: list[object]) -> None:
        self.models = models
        self.models_changed.emit(models)

    def publish_datasets(self, datasets: list[object]) -> None:
        self.datasets = datasets
        self.datasets_changed.emit(datasets)

    def notify(self, level: str, message: str) -> None:
        self.notice.emit(Notice(level=level, message=message))

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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.hardware: HardwareSnapshot | None = None
        self.engines: dict[str, object] = {}
        self.job: JobSnapshot | None = None

    def publish_hardware(self, snapshot: HardwareSnapshot) -> None:
        self.hardware = snapshot
        self.hardware_changed.emit(snapshot)

    def publish_engines(self, statuses: dict[str, object]) -> None:
        self.engines = statuses
        self.engines_changed.emit(statuses)

    def publish_job(self, snapshot: JobSnapshot) -> None:
        self.job = snapshot
        self.job_changed.emit(snapshot)

    def notify(self, level: str, message: str) -> None:
        self.notice.emit(Notice(level=level, message=message))

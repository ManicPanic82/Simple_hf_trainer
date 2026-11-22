"""Lightweight metrics collection service for training runs."""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, Signal


class MetricsRecord:
    def __init__(
        self, job_id: str | None, name: str, value: float, step: int | None = None
    ) -> None:
        self.job_id = job_id
        self.name = name
        self.value = value
        self.step = step
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "value": self.value,
            "step": self.step,
            "timestamp": self.timestamp,
        }


class MetricsService(QObject):
    metrics_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[MetricsRecord] = []

    def capture_from_worker_message(self, message) -> None:
        try:
            job_id = getattr(message, "job_id", None)
            payload = getattr(message, "payload", {}) or {}
            name = payload.get("name")
            value = payload.get("value")
            if name is None or value is None:
                return
            step = payload.get("step")
            record = MetricsRecord(job_id=job_id, name=str(name), value=float(value), step=step)
            self._records.append(record)
            self.metrics_ready.emit(record.to_dict())
        except Exception:
            return

    def capture(self, job_id: str | None, name: str, value: float, step: int | None = None) -> None:
        record = MetricsRecord(job_id=job_id, name=name, value=value, step=step)
        self._records.append(record)
        self.metrics_ready.emit(record.to_dict())

    def get_records(self, job_id=None, name=None) -> list[dict[str, Any]]:
        out = []
        for r in self._records:
            if job_id is not None and r.job_id != job_id:
                continue
            if name is not None and r.name != name:
                continue
            out.append(r.to_dict())
        return out

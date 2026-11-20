"""Compact shared status strip (not a tab) pinned to the bottom of the window."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from services.hardware_monitor import HardwareSnapshot
from ui.components import Pill
from workers.job_manager import JobSnapshot


def _gb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "—"
    return f"{size_bytes / 1024**3:.1f} GB"


def _mb(size_mb: float | None) -> str:
    if size_mb is None:
        return "—"
    return f"{size_mb / 1024:.1f} GB"


class StatusStrip(QFrame):
    """GPU, VRAM, RAM, job, and GPU-lock state at a glance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        self.setMinimumHeight(34)
        self.setMaximumHeight(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(14)

        self.gpu_value = self._value_label("Detecting GPU…")
        layout.addWidget(self._caption("GPU"))
        layout.addWidget(self.gpu_value)
        layout.addWidget(self._separator())

        self.util_value = self._value_label("—")
        layout.addWidget(self._caption("UTIL"))
        layout.addWidget(self.util_value)

        self.vram_value = self._value_label("—")
        self.vram_bar = self._bar()
        layout.addWidget(self._caption("VRAM"))
        layout.addWidget(self.vram_value)
        layout.addWidget(self.vram_bar)
        layout.addWidget(self._separator())

        self.ram_value = self._value_label("—")
        self.ram_bar = self._bar()
        layout.addWidget(self._caption("RAM"))
        layout.addWidget(self.ram_value)
        layout.addWidget(self.ram_bar)

        layout.addStretch(1)

        self.job_value = self._value_label("Idle")
        layout.addWidget(self._caption("JOB"))
        layout.addWidget(self.job_value)

        self.engine_value = self._value_label("—")
        layout.addWidget(self._caption("ENGINE"))
        layout.addWidget(self.engine_value)
        layout.addWidget(self._separator())

        self.lock_pill = Pill("GPU free", "ok", self)
        layout.addWidget(self.lock_pill)

    # -- construction helpers ------------------------------------------

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("StripLabel")
        return label

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("StripValue")
        return label

    @staticmethod
    def _bar() -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedWidth(64)
        bar.setFixedHeight(6)
        return bar

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setObjectName("StripSep")
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFixedHeight(14)
        return line

    # -- updates --------------------------------------------------------

    def apply_snapshot(self, snapshot: HardwareSnapshot) -> None:
        gpu = snapshot.gpu
        if gpu is not None:
            self.gpu_value.setText(gpu.name)
            self.gpu_value.setToolTip(
                f"Driver {gpu.driver_version or 'unknown'}"
                + (f"\n{gpu.temperature_c} °C" if gpu.temperature_c is not None else "")
            )
            self.util_value.setText(
                f"{gpu.utilization_percent:.0f} %" if gpu.utilization_percent is not None else "—"
            )
            if gpu.memory_total_mb:
                self.vram_value.setText(f"{_mb(gpu.memory_used_mb)} / {_mb(gpu.memory_total_mb)}")
                self.vram_bar.setValue(int(snapshot.vram_percent or 0))
            else:
                self.vram_value.setText("—")
                self.vram_bar.setValue(0)
        else:
            self.gpu_value.setText("Unavailable")
            self.gpu_value.setToolTip(snapshot.gpu_error or "No GPU information available.")
            self.util_value.setText("—")
            self.vram_value.setText("—")
            self.vram_bar.setValue(0)

        if snapshot.ram_total_bytes:
            self.ram_value.setText(
                f"{_gb(snapshot.ram_used_bytes)} / {_gb(snapshot.ram_total_bytes)}"
            )
            self.ram_bar.setValue(int(snapshot.ram_percent or 0))
        else:
            self.ram_value.setText("—")
            self.ram_bar.setValue(0)
            self.ram_value.setToolTip(snapshot.ram_error or "")

    def apply_job(self, job: JobSnapshot | None) -> None:
        if job is None:
            self.job_value.setText("Idle")
            self.job_value.setToolTip("")
            self.engine_value.setText("—")
            return
        self.job_value.setText(job.state.value)
        self.engine_value.setText(job.engine)
        detail = job.message or job.model_ref or ""
        self.job_value.setToolTip(f"{job.job_id}: {detail}" if detail else job.job_id)

    def set_gpu_owner(self, owner_job_id: str | None) -> None:
        if owner_job_id:
            self.lock_pill.set_state("warn", f"GPU busy · {owner_job_id}")
            self.lock_pill.setToolTip(
                "Unsloth and Heretic run exclusively; a second engine cannot start "
                "until this job exits."
            )
        else:
            self.lock_pill.set_state("ok", "GPU free")
            self.lock_pill.setToolTip("No engine job currently owns the GPU.")

    def set_engine_availability(self, available: int, total: int) -> None:
        self.engine_value.setToolTip(f"{available} of {total} engine environments detected.")

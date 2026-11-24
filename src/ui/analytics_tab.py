"""Run analytics: live metric and hardware charts plus persisted job history."""

from __future__ import annotations

import math
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QPalette
from PySide6.QtWidgets import QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from app.state import AppState
from ui.tab_page import TabPage
from ui.training_dashboard import ArchitectureView
from workers.job_manager import JobManager


class HistoryChart(QWidget):
    """Dependency-free line chart for timestamped numeric history."""

    def __init__(self, series: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.series = series
        self.points: dict[str, list[tuple[float, float]]] = {key: [] for key in series}
        self.setMinimumHeight(175)

    def add_point(self, name: str, x: float, value: float) -> None:
        if name not in self.points or not math.isfinite(value):
            return
        self.points[name].append((x, value))
        del self.points[name][:-1200]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(42, 12, -12, -26)
        palette = self.palette()
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        colors = (palette.color(QPalette.ColorRole.Highlight),
                  palette.color(QPalette.ColorRole.Link),
                  palette.color(QPalette.ColorRole.Text))
        painter.setPen(QPen(muted, 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        all_points = [p for values in self.points.values() for p in values]
        if not all_points:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Waiting for real run data…")
            painter.end()
            return
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        if max_x == min_x:
            max_x += 1
        if math.isclose(min_y, max_y):
            padding = max(abs(max_y) * 0.1, 0.1)
        else:
            padding = (max_y - min_y) * 0.08
        min_y -= padding
        max_y += padding
        painter.setPen(muted)
        painter.drawText(3, rect.top() + 8, f"{max_y:.3g}")
        painter.drawText(3, rect.bottom(), f"{min_y:.3g}")
        for index, name in enumerate(self.series):
            values = self.points[name]
            if not values:
                continue
            painter.setPen(QPen(colors[index % len(colors)], 2))
            painter.drawText(rect.left() + index * 110, rect.top() + 12, name)
            previous = None
            for x_value, y_value in values:
                point = (
                    round(rect.left() + (x_value - min_x) / (max_x - min_x) * rect.width()),
                    round(rect.bottom() - (y_value - min_y) / (max_y - min_y) * rect.height()),
                )
                if previous is not None:
                    painter.drawLine(previous[0], previous[1], point[0], point[1])
                previous = point
        painter.end()


class AnalyticsTab(TabPage):
    """Aggregates actual worker metrics, hardware snapshots, and persisted jobs."""

    def __init__(self, state: AppState, job_manager: JobManager, parent=None) -> None:
        super().__init__(
            "Analytics",
            "Inspect real training metrics, hardware utilisation and previous job outcomes. "
            "Charts remain empty until the worker reports measurements.",
            parent,
        )
        self._state = state
        self._job_manager = job_manager
        self.overview = QLabel("No training metrics received yet.", self)
        self.overview.setObjectName("StripValue")
        self.add(self.overview)

        dataset_title = QLabel("Dataset overview", self)
        dataset_title.setObjectName("SectionLabel")
        self.add(dataset_title)
        self.dataset_summary = QLabel("Waiting for local dataset discovery…", self)
        self.dataset_summary.setObjectName("Muted")
        self.dataset_summary.setWordWrap(True)
        self.add(self.dataset_summary)
        state.datasets_changed.connect(self._on_datasets)
        if state.datasets:
            self._on_datasets(state.datasets)

        grid = QGridLayout()
        self.loss_chart = HistoryChart(("loss", "eval_loss"), self)
        self.lr_chart = HistoryChart(("learning_rate",), self)
        self.gpu_chart = HistoryChart(("GPU %", "VRAM %", "RAM %"), self)
        self.gradient_chart = HistoryChart(("grad_norm",), self)
        self.throughput_chart = HistoryChart(("tokens_per_second",), self)
        self.evaluation_chart = HistoryChart(
            ("eval_accuracy", "eval_f1", "eval_precision", "eval_recall"), self
        )
        for index, (title, chart) in enumerate((
            ("Training and evaluation loss", self.loss_chart),
            ("Learning-rate schedule", self.lr_chart),
            ("Hardware utilisation", self.gpu_chart),
            ("Gradient norm", self.gradient_chart),
            ("Training throughput", self.throughput_chart),
            ("Evaluation metrics", self.evaluation_chart),
        )):
            column = index % 2
            row = index // 2
            wrapper = QWidget(self)
            layout = QVBoxLayout(wrapper)
            layout.setContentsMargins(0, 0, 0, 0)
            heading = QLabel(title, wrapper)
            heading.setObjectName("SectionLabel")
            layout.addWidget(heading)
            layout.addWidget(chart)
            grid.addWidget(wrapper, row, column)
        self.add_layout(grid)

        architecture_title = QLabel("Loaded model architecture", self)
        architecture_title.setObjectName("SectionLabel")
        self.add(architecture_title)
        self.architecture = ArchitectureView(self)
        self.add(self.architecture)
        self.architecture_details = QLabel("Architecture details appear after a model is loaded.", self)
        self.architecture_details.setObjectName("Muted")
        self.architecture_details.setWordWrap(True)
        self.add(self.architecture_details)
        state.architecture_changed.connect(self.set_architecture)
        if state.architecture:
            self.set_architecture(state.architecture)

        history_title = QLabel("Run history", self)
        history_title.setObjectName("SectionLabel")
        self.add(history_title)
        self.history_table = QTableWidget(0, 6, self)
        self.history_table.setHorizontalHeaderLabels(
            ["Engine", "Model", "Status", "Started", "Duration", "Message"]
        )
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setMinimumHeight(190)
        self.add(self.history_table)

        state.metric_changed.connect(self._on_metric)
        state.hardware_changed.connect(self._on_hardware)
        state.job_changed.connect(lambda _job: self.refresh_history())
        for item in getattr(state, "metric_history", []):
            self._on_metric(item)
        for item in getattr(state, "hardware_history", []):
            self._on_hardware(item)
        self.refresh_history()
        self.set_status("Live analytics")

    def _on_datasets(self, records: list) -> None:
        count = len(records)
        rows = [record.row_count for record in records if record.row_count is not None]
        total_rows = sum(rows) if rows else None
        total_size = sum(float(record.size_mb) for record in records)
        prepared = sum(1 for record in records if record.is_prepared)
        formats: dict[str, int] = {}
        for record in records:
            label = str(record.format).upper()
            formats[label] = formats.get(label, 0) + 1
        row_text = f"{total_rows:,} known rows" if total_rows is not None else "row counts unavailable"
        format_text = ", ".join(f"{name}: {amount}" for name, amount in sorted(formats.items()))
        self.dataset_summary.setText(
            f"{count} local datasets · {row_text} · {total_size:.1f} MB · "
            f"{prepared} prepared · {format_text or 'no format data'}"
        )

    def set_architecture(self, metadata: dict) -> None:
        self.architecture.set_architecture(metadata)
        model_type = metadata.get("model_type") or "Transformer"
        layers = metadata.get("num_hidden_layers")
        self.architecture_details.setText(
            f"{model_type} · {layers or 'unknown'} layers · "
            f"hidden size {metadata.get('hidden_size', 'unknown')} · "
            f"attention heads {metadata.get('num_attention_heads', 'unknown')} · "
            f"LoRA rank {metadata.get('lora_rank', '—')} · "
            f"targets {metadata.get('target_modules', '—')}"
        )

    def _on_metric(self, item: dict) -> None:
        name = str(item.get("name", ""))
        step = int(item.get("step", 0))
        value = float(item.get("value", 0.0))
        if name in self.loss_chart.points:
            self.loss_chart.add_point(name, step, value)
        elif name == "learning_rate":
            self.lr_chart.add_point(name, step, value)
        elif name == "grad_norm":
            self.gradient_chart.add_point(name, step, value)
        elif name == "tokens_per_second":
            self.throughput_chart.add_point(name, step, value)
        elif name in self.evaluation_chart.points:
            self.evaluation_chart.add_point(name, step, value)
        self.overview.setText(
            f"Latest: step {step} · {name} {value:.5g} · "
            f"job {item.get('job_id') or 'current'}"
        )

    def _on_hardware(self, snapshot) -> None:
        stamp = float(getattr(snapshot, "captured_at", 0.0) or 0.0)
        gpu = getattr(snapshot, "gpu", None)
        if gpu is not None and gpu.utilization_percent is not None:
            self.gpu_chart.add_point("GPU %", stamp, float(gpu.utilization_percent))
        if getattr(snapshot, "vram_percent", None) is not None:
            self.gpu_chart.add_point("VRAM %", stamp, float(snapshot.vram_percent))
        if getattr(snapshot, "ram_percent", None) is not None:
            self.gpu_chart.add_point("RAM %", stamp, float(snapshot.ram_percent))

    def refresh(self) -> None:
        self.refresh_history()

    def refresh_history(self) -> None:
        jobs = list(self._job_manager.history())
        jobs.extend(job for job in self._job_manager.jobs() if job.is_active)
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        self.history_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            started = job.started_at or job.created_at
            finished = job.finished_at
            duration = max(0, int((finished or datetime.now().timestamp()) - started))
            values = (
                job.engine,
                job.model_ref,
                job.state.value,
                datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
                f"{duration // 60}m {duration % 60}s",
                job.message,
            )
            for column, value in enumerate(values):
                self.history_table.setItem(row, column, QTableWidgetItem(str(value)))


__all__ = ["AnalyticsTab", "HistoryChart"]

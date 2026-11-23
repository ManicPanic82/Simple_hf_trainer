"""Live training dashboard widgets: model architecture and real metric history."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QPalette
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget


class MetricChart(QWidget):
    """Small dependency-free chart that plots only metrics received from the worker."""

    def __init__(self, title: str, series: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.series = series
        self.history: dict[str, list[tuple[int, float]]] = {name: [] for name in series}
        self.setMinimumHeight(190)
        self.setMinimumWidth(260)
        self.setToolTip("Values are plotted from the active training worker's reported metrics.")

    def add_point(self, name: str, step: int, value: float) -> None:
        if name not in self.history or not math.isfinite(value):
            return
        points = self.history[name]
        if points and points[-1][0] == step:
            points[-1] = (step, value)
        else:
            points.append((step, value))
        if len(points) > 2000:
            del points[:-2000]
        self.update()

    def clear(self) -> None:
        for points in self.history.values():
            points.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(42, 12, -14, -30)
        palette = self.palette()
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        foreground = palette.color(QPalette.ColorRole.Text)
        painter.setPen(QPen(muted, 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        all_points = [point for points in self.history.values() for point in points]
        if not all_points:
            painter.setPen(muted)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Waiting for worker metrics…")
            painter.end()
            return
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        if max_x == min_x:
            max_x += 1
        if math.isclose(max_y, min_y):
            padding = max(abs(max_y) * 0.1, 0.1)
            min_y -= padding
            max_y += padding
        else:
            padding = (max_y - min_y) * 0.08
            min_y -= padding
            max_y += padding
        painter.setPen(muted)
        painter.drawText(4, rect.top() + 8, f"{max_y:.3g}")
        painter.drawText(4, rect.bottom(), f"{min_y:.3g}")
        painter.drawText(rect.left(), self.height() - 7, str(min_x))
        painter.drawText(rect.right() - 35, self.height() - 7, str(max_x))
        colors = [palette.color(QPalette.ColorRole.Highlight), foreground,
                  palette.color(QPalette.ColorRole.Link)]
        for index, name in enumerate(self.series):
            points = self.history[name]
            if not points:
                continue
            pen = QPen(colors[index % len(colors)], 2)
            painter.setPen(pen)
            previous = None
            for step, value in points:
                x = rect.left() + (step - min_x) / (max_x - min_x) * rect.width()
                y = rect.bottom() - (value - min_y) / (max_y - min_y) * rect.height()
                current = (round(x), round(y))
                if previous is not None:
                    painter.drawLine(previous[0], previous[1], current[0], current[1])
                previous = current
        painter.end()


class ArchitectureView(QWidget):
    """Schematic transformer stack populated from the loaded model's config."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metadata: dict = {}
        self.setMinimumHeight(210)
        self.setMinimumWidth(260)

    def set_architecture(self, metadata: dict) -> None:
        self.metadata = {
            key: value for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        self.update()

    def clear(self) -> None:
        self.metadata.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        muted = palette.color(QPalette.ColorRole.PlaceholderText)
        text = palette.color(QPalette.ColorRole.Text)
        accent = palette.color(QPalette.ColorRole.Highlight)
        if not self.metadata:
            painter.setPen(muted)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Architecture appears after the model loads")
            painter.end()
            return
        width = min(174, self.width() - 30)
        x = (self.width() - width) // 2
        y = 8
        blocks = [
            ("Token embeddings", "vocab_size"),
            ("Transformer stack", "num_hidden_layers"),
            ("Attention + MLP", "num_attention_heads"),
            ("LoRA adapters", "lora_rank"),
            ("Output / LM head", "hidden_size"),
        ]
        height = 29
        for index, (label, key) in enumerate(blocks):
            painter.setPen(QPen(accent if index in (1, 3) else muted, 1))
            painter.drawRoundedRect(x, y, width, height, 5, 5)
            value = self.metadata.get(key)
            detail = f" · {value}" if value is not None else ""
            painter.setPen(text)
            painter.drawText(x + 8, y + 19, f"{label}{detail}")
            if index < len(blocks) - 1:
                painter.setPen(QPen(muted, 1))
                painter.drawLine(x + width // 2, y + height, x + width // 2, y + height + 9)
            y += height + 9
        painter.end()


class TrainingDashboard(QWidget):
    """Architecture, loss/LR plots and live run statistics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._started_at: float | None = None
        self._progress_fraction: float | None = None
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update_clock)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        self.stats = QGridLayout()
        self.stats.setHorizontalSpacing(18)
        self.stats.setVerticalSpacing(6)
        self.stat_labels: dict[str, QLabel] = {}
        for index, (key, title) in enumerate((
            ("epoch", "Epoch"), ("learning_rate", "Learning rate"),
            ("grad_norm", "Gradient norm"), ("tokens_per_second", "Tokens/sec"),
            ("gpu_utilization", "GPU utilisation"), ("gpu_memory", "GPU memory"),
            ("ram", "System RAM"), ("elapsed", "Elapsed"), ("remaining", "Est. remaining"),
        )):
            label = QLabel(f"{title}: —", self)
            label.setObjectName("StripValue")
            self.stat_labels[key] = label
            self.stats.addWidget(label, index // 3, index % 3)
        root.addLayout(self.stats)

        charts = QGridLayout()
        charts.setSpacing(10)
        loss_column = QVBoxLayout()
        loss_title = QLabel("Loss history · lower is generally better", self)
        loss_title.setObjectName("SectionLabel")
        loss_column.addWidget(loss_title)
        self.loss_chart = MetricChart("Loss", ("loss", "eval_loss"), self)
        loss_column.addWidget(self.loss_chart)
        lr_column = QVBoxLayout()
        lr_title = QLabel("Learning-rate schedule", self)
        lr_title.setObjectName("SectionLabel")
        lr_column.addWidget(lr_title)
        self.lr_chart = MetricChart("Learning rate", ("learning_rate",), self)
        lr_column.addWidget(self.lr_chart)
        charts.addLayout(loss_column, 0, 0)
        charts.addLayout(lr_column, 0, 1)
        root.addLayout(charts)

        self.architecture_title = QLabel("Loaded model architecture", self)
        self.architecture_title.setObjectName("SectionLabel")
        root.addWidget(self.architecture_title)
        self.architecture = ArchitectureView(self)
        root.addWidget(self.architecture)
        self.architecture_details = QLabel(
            "Layer dimensions and adapter targets are reported by the loaded model.", self
        )
        self.architecture_details.setObjectName("Muted")
        self.architecture_details.setWordWrap(True)
        root.addWidget(self.architecture_details)

    def reset(self) -> None:
        self._clock.stop()
        self._started_at = None
        self._progress_fraction = None
        for label in self.stat_labels.values():
            title = label.text().split(":", 1)[0]
            label.setText(f"{title}: —")
        self.loss_chart.clear()
        self.lr_chart.clear()
        self.architecture.clear()
        self.architecture_title.setText("Loaded model architecture")
        self.architecture_details.setText(
            "Layer dimensions and adapter targets are reported by the loaded model."
        )

    def begin_run(self) -> None:
        self._started_at = time.monotonic()
        self._progress_fraction = 0.0
        self._clock.start()
        self._update_clock()

    def set_progress(self, fraction: float) -> None:
        self._progress_fraction = max(0.0, min(1.0, float(fraction)))
        self._update_clock()

    def finish_run(self) -> None:
        self._clock.stop()
        self.stat_labels["remaining"].setText("Est. remaining: Complete")

    def _update_clock(self) -> None:
        if self._started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self._started_at)
        self.stat_labels["elapsed"].setText(f"Elapsed: {self._format_duration(elapsed)}")
        fraction = self._progress_fraction
        remaining = elapsed * (1.0 - fraction) / fraction if fraction and fraction > 0 else None
        estimate = self._format_duration(remaining) if remaining is not None else "—"
        self.stat_labels["remaining"].setText(f"Est. remaining: {estimate}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def set_architecture(self, metadata: dict) -> None:
        self.architecture.set_architecture(metadata)
        model_type = metadata.get("model_type") or "Transformer"
        layers = metadata.get("num_hidden_layers")
        self.architecture_title.setText(
            f"Loaded architecture · {model_type}" + (f" · {layers} layers" if layers else "")
        )
        detail_keys = (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "max_position_embeddings",
            "lora_rank",
            "lora_alpha",
            "target_modules",
            "precision",
        )
        details = [
            f"{key.replace('_', ' ')}: {metadata[key]}"
            for key in detail_keys
            if metadata.get(key) is not None
        ]
        self.architecture_details.setText(
            "   ·   ".join(details) or "Architecture metadata received."
        )

    def add_metric(self, name: str, step: int, value: float) -> None:
        if not math.isfinite(value):
            return
        if name in self.loss_chart.series:
            self.loss_chart.add_point(name, step, value)
        if name == "learning_rate":
            self.lr_chart.add_point(name, step, value)
        if name in self.stat_labels:
            if name in {"learning_rate", "grad_norm"}:
                display = f"{value:.3g}"
            elif name == "tokens_per_second":
                display = f"{value:.1f}"
            else:
                display = f"{value:.2f}"
            title = self.stat_labels[name].text().split(":", 1)[0]
            self.stat_labels[name].setText(f"{title}: {display}")

    def set_hardware(self, snapshot) -> None:
        gpu = getattr(snapshot, "gpu", None)
        if gpu is not None:
            if gpu.utilization_percent is not None:
                self.stat_labels["gpu_utilization"].setText(
                    f"GPU utilisation: {gpu.utilization_percent:.0f}%"
                )
            if gpu.memory_used_mb is not None and gpu.memory_total_mb:
                self.stat_labels["gpu_memory"].setText(
                    f"GPU memory: {gpu.memory_used_mb / 1024:.2f} / "
                    f"{gpu.memory_total_mb / 1024:.2f} GB"
                )
        used, total = snapshot.ram_used_bytes, snapshot.ram_total_bytes
        if used is not None and total:
            self.stat_labels["ram"].setText(
                f"System RAM: {used / (1024 ** 3):.1f} / {total / (1024 ** 3):.1f} GB"
            )

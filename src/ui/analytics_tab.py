"""Analytics tab: training visualisation, hardware monitoring, and run history."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from app.state import AppState
from services.engine_registry import EngineRegistry
from services.model_service import ModelService
from services.storage_service import StorageService
from ui.components import Card
from ui.console import Console
from ui.tab_page import TabPage
from workers.job_manager import JobManager


class AnalyticsTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        job_manager=None,
        metrics_service=None,
        registry: EngineRegistry | None = None,
        model_service: ModelService | None = None,
        dataset_service=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Analytics",
            "Monitor training performance, hardware metrics, evaluation results, "
            "and historical run comparisons.",
            parent=parent,
        )
        self._state = state
        self._storage = storage
        self._job_manager = job_manager
        self._metrics_service = metrics_service
        self._registry = registry
        self._model_service = model_service
        self._dataset_service = dataset_service
        self._active_job_id = None
        self._charts = {}

        # Overview
        overview_card = Card("Overview", "Select a job to see real-time metrics")
        overview_layout = QHBoxLayout()
        overview_layout.setSpacing(12)

        self.job_selector = QListWidget(overview_card)
        self.job_selector.setMinimumWidth(220)
        overview_layout.addWidget(self.job_selector, 1)

        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(4)
        metric_names = ("loss", "eval_loss", "lr", "tps", "gpu_util", "vram_used", "elapsed")
        metric_display = {
            "loss": "Training Loss:  --",
            "eval_loss": "Eval Loss:  --",
            "lr": "LR:  --",
            "tps": "TPS:  --",
            "gpu_util": "GPU Util:  --",
            "vram_used": "VRAM Used:  --",
            "elapsed": "Elapsed:  --",
        }
        self._metric_labels = {}
        for name in metric_names:
            lbl = QLabel(metric_display[name], overview_card)
            self._metric_labels[name] = lbl
            metrics_layout.addWidget(lbl)
        metrics_layout.addStretch(1)
        overview_layout.addLayout(metrics_layout, 1)

        self._loss_steps = []
        self._loss_values = []
        overview_card.add_layout(overview_layout)
        self.add(overview_card)

        # Charts + hardware
        charts_splitter = QSplitter(Qt.Orientation.Vertical)

        training_section = Card("Training Performance", "Live metrics visualisations")
        self.loss_plot_widget = pg.PlotWidget(title="Training Loss")
        self.loss_plot_widget.setLabel("bottom", "Step")
        self.loss_plot_widget.setLabel("left", "Loss")
        self.loss_plot_widget.setMinimumHeight(220)
        training_section.add(self.loss_plot_widget)
        charts_splitter.addWidget(training_section)

        hardware_section = Card("Hardware", "GPU and system monitoring")
        hardware_layout = QHBoxLayout()
        hardware_layout.setSpacing(12)

        gpu_info = QFrame(hardware_section)
        gpu_layout = QVBoxLayout(gpu_info)
        self.gpu_label = QLabel("GPU: --", gpu_info)
        self.gpu_util_label = QLabel("Util: --", gpu_info)
        self.gpu_vram_label = QLabel("VRAM: --", gpu_info)
        gpu_layout.addWidget(self.gpu_label)
        gpu_layout.addWidget(self.gpu_util_label)
        gpu_layout.addWidget(self.gpu_vram_label)
        hardware_layout.addWidget(gpu_info, 1)

        sys_info = QFrame(hardware_section)
        sys_layout = QVBoxLayout(sys_info)
        self.sys_ram_label = QLabel("RAM: --", sys_info)
        sys_layout.addWidget(self.sys_ram_label)
        sys_layout.addStretch(1)
        hardware_layout.addWidget(sys_info, 1)

        hardware_section.add_layout(hardware_layout)
        charts_splitter.addWidget(hardware_section)

        self.add(charts_splitter)

        # History
        history_card = Card("Run History", "Browse and compare past training runs")
        self.history_list = QListWidget(history_card)
        self.history_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.history_list.setMinimumHeight(120)
        history_card.add(self.history_list, 1)

        history_buttons = QHBoxLayout()
        self.refresh_history_btn = QPushButton("Refresh")
        self.export_history_btn = QPushButton("Export Logs")
        history_buttons.addWidget(self.refresh_history_btn)
        history_buttons.addStretch(1)
        history_buttons.addWidget(self.export_history_btn)
        history_card.add_layout(history_buttons)
        self.add(history_card)

        # Console
        console_card = Card("Console", "Live training logs")
        self.console_widget = Console(parent=console_card)
        self.console_widget.setMinimumHeight(160)
        console_card.add(self.console_widget, 1)
        self.add(console_card)

        self.finish()

    def _build_ui(self) -> None:
        return

    def _connect_signals(self) -> None:
        if self._metrics_service is not None and hasattr(self._metrics_service, "metrics_ready"):
            try:
                self._metrics_service.metrics_ready.connect(self._on_metrics_ready)
            except Exception:
                pass
        try:
            self.job_selector.currentRowChanged.connect(self._on_row_changed)
        except Exception:
            pass
        try:
            self.history_list.itemDoubleClicked.connect(self._on_history_item_double_clicked)
        except Exception:
            pass
        try:
            self.refresh_history_btn.clicked.connect(self.refresh_history)
        except Exception:
            pass
        try:
            self.export_history_btn.clicked.connect(
                lambda: self._export_run_logs(self._active_job_id)
            )
        except Exception:
            pass

    def _on_metrics_ready(self, record=None, *args, **kwargs) -> None:
        if not isinstance(record, dict):
            # support being called with a positional record or no args
            if args and isinstance(args[0], dict):
                record = args[0]
            else:
                return
        job_id = record.get("job_id")
        name = record.get("name")
        value = record.get("value")
        step = record.get("step")
        if name == "loss" and value is not None:
            try:
                s = int(step) if step is not None else len(self._loss_steps) + 1
            except (TypeError, ValueError):
                s = len(self._loss_steps) + 1
            self._loss_steps.append(s)
            self._loss_values.append(float(value))
            try:
                self.loss_plot_widget.plot(self._loss_steps, self._loss_values, clear=True)
            except Exception:
                try:
                    self.loss_plot_widget.clear()
                    self.loss_plot_widget.plot(self._loss_steps, self._loss_values)
                except Exception:
                    pass
            if "loss" in self._metric_labels:
                self._metric_labels["loss"].setText(f"Training Loss:  {float(value):.4f}")
        if name in self._metric_labels and name != "loss":
            self._metric_labels[name].setText(f"{name}:  {value}")

    def refresh_charts_for_job(self, job_id=None) -> None:
        self._loss_steps = []
        self._loss_values = []
        if self._metrics_service is None:
            return
        try:
            records = self._metrics_service.get_records(job_id=job_id, name="loss")
        except Exception:
            records = []
        for rec in records:
            step = rec.get("step")
            value = rec.get("value")
            if value is None:
                continue
            try:
                s = int(step) if step is not None else len(self._loss_steps) + 1
            except (TypeError, ValueError):
                s = len(self._loss_steps) + 1
            self._loss_steps.append(s)
            self._loss_values.append(float(value))
        try:
            self.loss_plot_widget.clear()
            if self._loss_steps:
                self.loss_plot_widget.plot(self._loss_steps, self._loss_values)
        except Exception:
            pass

    def refresh(self) -> None:
        self.refresh_charts_for_job(self._active_job_id)

    def _on_job_selected(self) -> None:
        if self.job_selector is None:
            return
        try:
            item = self.job_selector.currentItem()
            self._active_job_id = item.text() if item is not None else None
        except Exception:
            self._active_job_id = None
        self.refresh_charts_for_job(self._active_job_id)

    def _on_row_changed(self, new_row: int) -> None:
        self._on_job_selected()

    def _on_history_item_double_clicked(self, item) -> None:
        return

    def refresh_history(self) -> None:
        if self.history_list is None or self._job_manager is None:
            return
        try:
            self.history_list.clear()
        except Exception:
            pass

    def _export_run_logs(self, job_id=None) -> None:
        return

    def _on_console_log(self, line: str) -> None:
        if self.console_widget is not None:
            try:
                self.console_widget.append(str(line))
            except Exception:
                pass

    def refresh_charts_for_job(self, job_id=None) -> None:
        return

    def _get_training_loss(self, job_id=None):
        return []

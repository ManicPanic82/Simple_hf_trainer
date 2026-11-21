"""Main application window: header, five workflow tabs, and shared status strip."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.state import Notice
from services.engine_registry import EngineState, EngineStatus
from ui.analytics_tab import AnalyticsTab
from ui.components import Pill
from ui.datasets_tab import DatasetsTab
from ui.heretic_tab import HereticTab
from ui.models_tab import ModelsTab
from ui.settings_dialog import SettingsDialog
from ui.status_strip import StatusStrip
from ui.training_tab import TrainingTab

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.application import AppContext

APP_VERSION = "0.1.0"
NOTICE_TIMEOUT_MS = 12_000
SHUTDOWN_POLL_MS = 100
SHUTDOWN_TIMEOUT_MS = 15_000


class NoticeBanner(QFrame):
    """Transient message strip below the header."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Notice")
        self.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 7, 10, 7)
        layout.setSpacing(10)

        self.label = QLabel("", self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)

        close = QPushButton("Dismiss", self)
        close.setFlat(True)
        close.clicked.connect(self.hide)
        layout.addWidget(close)
        self._close_button = close
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_notice(self, notice: Notice) -> None:
        self.setProperty("level", notice.level)
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.label.setText(notice.message)
        self.setToolTip(notice.message)
        self.setVisible(True)
        self._timer.start(NOTICE_TIMEOUT_MS)


class MainWindow(QMainWindow):
    """The single application window. Five tabs, in fixed order."""

    TAB_NAMES = ("Models", "Datasets", "Training", "Analytics", "Heretic")

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.state = context.state

        self.setWindowTitle("Simple HF Trainer")
        self.resize(1200, 820)
        self.setMinimumSize(960, 680)

        self._build_menu()
        self._build_layout()
        self._connect_signals()
        self._report_settings_issues()
        self.refresh_all()

    # -- construction ----------------------------------------------------

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        settings_action = file_menu.addAction("Settings…")
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_settings)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("About Simple HF Trainer")
        about_action.triggered.connect(self._show_about)

    def _build_layout(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame(central)
        header.setObjectName("Header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(14)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("Simple HF Trainer", header)
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "Local model discovery · datasets · Unsloth fine-tuning · Heretic modification",
            header,
        )
        subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)
        header_layout.addStretch(1)

        self.unsloth_pill = Pill("Unsloth · checking", "off", header)
        self.heretic_pill = Pill("Heretic · checking", "off", header)
        header_layout.addWidget(self.unsloth_pill)
        header_layout.addWidget(self.heretic_pill)

        self.settings_button = QPushButton("Settings", header)
        self.settings_button.clicked.connect(self.open_settings)
        header_layout.addWidget(self.settings_button)

        root.addWidget(header)

        self.banner = NoticeBanner(central)
        root.addWidget(self.banner)

        self.tabs = QTabWidget(central)
        self.tabs.setObjectName("MainTabs")
        self.tabs.setDocumentMode(True)

        self.models_tab = ModelsTab(
            self.state,
            self.context.storage,
            self.context.model_service,
        )
        self.datasets_tab = DatasetsTab(
            self.state,
            self.context.storage,
            self.context.dataset_service,
        )
        self.training_tab = TrainingTab(
            self.state,
            self.context.storage,
            self.context.job_manager,
            self.context.registry,
            self.context.model_service,
            self.context.dataset_service,
        )
        # Analytics tab (fifth tab)
        self.analytics_tab = AnalyticsTab(
            self.state,
            self.context.storage,
            self.context.job_manager,
            None,  # metrics_service not yet implemented in AppContext
            self.context.registry,
            self.context.model_service,
            self.context.dataset_service,
        )
        self.heretic_tab = HereticTab(
            self.state,
            self.context.storage,
            self.context.job_manager,
            self.context.registry,
            self.context.model_service,
            credentials=self.context.credentials,
        )

        for tab, name in zip(
            (
                self.models_tab,
                self.datasets_tab,
                self.training_tab,
                self.analytics_tab,
                self.heretic_tab,
            ),
            self.TAB_NAMES,
            strict=True,
        ):
            self.tabs.addTab(tab, name)

        root.addWidget(self.tabs, 1)

        self.status_strip = StatusStrip(central)
        root.addWidget(self.status_strip)

        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.state.hardware_changed.connect(self.status_strip.apply_snapshot)
        self.state.job_changed.connect(self.status_strip.apply_job)
        self.state.job_changed.connect(self._on_job_changed)
        self.state.engines_changed.connect(self._on_engines_changed)
        self.state.notice.connect(self.banner.show_notice)
        self.context.job_manager.gpu_ownership_changed.connect(self.status_strip.set_gpu_owner)

        self.models_tab.use_model_in_training.connect(self._on_use_model_in_training)
        self.models_tab.use_model_in_heretic.connect(self._on_use_model_in_heretic)
        self.datasets_tab.use_dataset_in_training.connect(self._on_use_dataset_in_training)

    def _on_use_model_in_training(self, model_ref: str) -> None:
        """Navigate to Analytics tab for training monitoring."""
        self.training_tab.select_model(model_ref)
        # Analytics tab (index 3) is dedicated for training metrics and monitoring
        self.tabs.setCurrentIndex(3)

    def _on_use_dataset_in_training(self, dataset_path: str) -> None:
        """Navigate to Analytics tab for dataset analysis."""
        self.training_tab.select_dataset(dataset_path)
        # Analytics tab (index 3) is dedicated for training metrics and monitoring
        self.tabs.setCurrentIndex(3)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Begin background monitoring. Called once the window is shown."""
        self.context.monitor.start()
        self.context.registry.refresh_async()
        self.models_tab.scan_library()
        self.datasets_tab.scan_datasets()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not getattr(self, "_force_close", False) and self._workers_running():
            event.ignore()
            self._begin_shutdown()
            return
        self._finalize_close(event)

    def _workers_running(self) -> bool:
        """Whether either engine tab still reports a live worker process."""
        return any(
            bool(getattr(tab, "worker_running", False))
            for tab in (self.training_tab, self.heretic_tab)
        )

    def _begin_shutdown(self) -> None:
        """Start a bounded, non-blocking shutdown of any active workers."""
        # Repeated close attempts must not spin or create duplicate timers.
        if getattr(self, "_shutdown_pending", False):
            return
        self._shutdown_pending = True
        self._shutdown_escalated = False
        self._force_close = False

        self.banner.show_notice(
            Notice(
                "info",
                "Stopping active workers before closing. Please wait for the workers to exit.",
            )
        )

        for tab in (
            self.training_tab,
            self.heretic_tab,
            self.models_tab,
            self.datasets_tab,
            self.analytics_tab,
        ):
            request_shutdown = getattr(tab, "request_shutdown", None)
            if callable(request_shutdown):
                request_shutdown()

        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(SHUTDOWN_POLL_MS)
        self._shutdown_timer.timeout.connect(self._poll_worker_shutdown)
        self._shutdown_timer.start()

        self._shutdown_timeout_timer = QTimer(self)
        self._shutdown_timeout_timer.setSingleShot(True)
        self._shutdown_timeout_timer.setInterval(SHUTDOWN_TIMEOUT_MS)
        self._shutdown_timeout_timer.timeout.connect(self._escalate_shutdown)
        self._shutdown_timeout_timer.start()

    def _poll_worker_shutdown(self) -> None:
        """Close the window once every engine worker has actually exited."""
        if self._workers_running():
            return

        self._stop_shutdown_timers()
        self.close()

    def _escalate_shutdown(self) -> None:
        """Force-stop stubborn workers, then close regardless of the outcome."""
        if getattr(self, "_shutdown_escalated", False):
            return
        self._shutdown_escalated = True
        self._stop_shutdown_timers()

        self.banner.show_notice(
            Notice(
                "warning",
                "Workers did not exit in time. Forcing them to stop and closing now.",
            )
        )

        for tab in (
            self.training_tab,
            self.heretic_tab,
            self.models_tab,
            self.datasets_tab,
            self.analytics_tab,
        ):
            self._force_stop_worker(tab)
            request_shutdown = getattr(tab, "request_shutdown", None)
            if callable(request_shutdown):
                with suppress(Exception):
                    request_shutdown()

        # A forced worker may still report as running; bypass that check so the
        # window closes even when a forced stop fails or is unavailable.
        self._force_close = True
        self.close()

    @staticmethod
    def _force_stop_worker(tab) -> None:
        """Best-effort hard stop of a tab's worker; never raises to the caller."""
        worker = getattr(tab, "worker", None)
        if worker is None:
            worker = getattr(tab, "_active_worker", None)
        stop = getattr(worker, "stop", None)
        if not callable(stop):
            return
        # A failed forced stop must not block the window from closing.
        with suppress(Exception):
            stop()

    def _stop_shutdown_timers(self) -> None:
        for name in ("_shutdown_timer", "_shutdown_timeout_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()

    def _finalize_close(self, event) -> None:
        self._stop_shutdown_timers()
        self.context.monitor.stop()
        self.context.registry.wait_for_idle(3000)
        super().closeEvent(event)

    def _report_settings_issues(self) -> None:
        issues = self.context.settings_service.issues
        if not issues:
            return
        level = "info"
        if any(
            marker in issue.lower()
            for issue in issues
            for marker in ("could not", "not an", "must be", "ignored", "corrupt")
        ):
            level = "warning"
        self.banner.show_notice(Notice(level, " ".join(issues[:4])))

    def _on_job_changed(self, job) -> None:
        if job is None:
            return
        if job.state.is_terminal and job.message:
            level = "error" if job.state.value == "failed" else "info"
            self.banner.show_notice(Notice(level, f"{job.engine}: {job.message}"))

    def _on_engines_changed(self, statuses: dict[str, EngineStatus]) -> None:
        self._update_engine_pill(self.unsloth_pill, "Unsloth", statuses.get("unsloth"))
        self._update_engine_pill(self.heretic_pill, "Heretic", statuses.get("heretic"))
        available = sum(1 for status in statuses.values() if status.is_available)
        self.status_strip.set_engine_availability(available, len(statuses) or 2)

    @staticmethod
    def _update_engine_pill(pill: Pill, name: str, status: EngineStatus | None) -> None:
        if status is None or status.state is EngineState.UNCHECKED:
            pill.set_state("off", f"{name} · checking")
            pill.setToolTip("Engine environments are being detected.")
            return
        mapping = {
            EngineState.AVAILABLE: "ok",
            EngineState.NOT_INSTALLED: "warn",
            EngineState.INTERPRETER_MISSING: "warn",
            EngineState.ERROR: "error",
        }
        pill.set_state(mapping.get(status.state, "off"), f"{name} · {status.badge()}")
        pill.setToolTip(status.detail or "")

    # -- commands --------------------------------------------------------

    def refresh_all(self) -> None:
        for tab in (
            self.models_tab,
            self.datasets_tab,
            self.training_tab,
            self.analytics_tab,
            self.heretic_tab,
        ):
            tab.refresh()

    def open_settings(self) -> None:
        dialog = SettingsDialog(
            self.context.settings,
            self.context.settings_service,
            credentials=self.context.credentials,
            parent=self,
        )
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        updated = dialog.result_settings()
        if updated is None:
            return

        self.context.settings = updated
        self.context.storage.update_settings(updated)
        self.context.registry.update_settings(updated)
        self.context.monitor.set_interval(updated.ui.poll_interval_ms)
        self.state.publish_engines(self.context.registry.statuses())
        self.context.registry.refresh_async()
        self.refresh_all()
        self.state.notify("info", "Settings saved.")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Simple HF Trainer",
            f"<b>Simple HF Trainer {APP_VERSION}</b><br><br>"
            "Phase 1 desktop foundation: model, dataset, training, and Heretic workflows "
            "are being built on top of verified engine compatibility.<br><br>"
            "Unsloth and Heretic are optional, installed in their own isolated Python "
            "environments. Features that are not implemented are labelled as such.",
        )

    # -- test hooks ------------------------------------------------------

    def tab_names(self) -> list[str]:
        return [self.tabs.tabText(index) for index in range(self.tabs.count())]


__all__ = ["MainWindow", "NoticeBanner", "APP_VERSION"]

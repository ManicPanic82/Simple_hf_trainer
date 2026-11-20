"""Application bootstrap: services, main window, and the Qt event loop."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QMessageBox

from app import paths
from app.logging_setup import configure_logging
from app.state import AppState
from services.credentials import CredentialStore, create_credential_store
from services.dataset_service import DatasetService
from services.engine_registry import EngineRegistry
from services.hardware_monitor import HardwareMonitor
from services.model_service import ModelService
from services.settings_service import AppSettings, SettingsService
from services.storage_service import StorageService
from ui.main_window import APP_VERSION, MainWindow
from ui.theme import apply_theme
from workers.job_manager import JobManager

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Everything the window needs, created once at startup."""

    settings_service: SettingsService
    settings: AppSettings
    storage: StorageService
    state: AppState
    monitor: HardwareMonitor
    registry: EngineRegistry
    job_manager: JobManager
    credentials: CredentialStore
    model_service: ModelService
    dataset_service: DatasetService
    window: MainWindow | None = None


def create_application(argv: list[str] | None = None) -> QApplication:
    """Return the existing QApplication or create one."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    app = QApplication(list(sys.argv if argv is None else argv))
    app.setApplicationName(paths.APP_NAME)
    app.setOrganizationName("Simple HF Trainer")
    app.setApplicationVersion(APP_VERSION)
    return app


def build_context(
    settings_service: SettingsService | None = None,
    *,
    settings: AppSettings | None = None,
    parent=None,
) -> AppContext:
    """Create services and wire the observable state.

    ``settings`` bypasses loading from disk and is used by tests.
    """
    service = settings_service or SettingsService()
    resolved = settings if settings is not None else service.load()

    storage = StorageService(resolved)
    state = AppState(parent)
    monitor = HardwareMonitor(interval_ms=resolved.ui.poll_interval_ms, parent=parent)
    registry = EngineRegistry(resolved, parent=parent)
    jobs = JobManager(parent=parent)

    credentials = create_credential_store()
    model_service = ModelService(storage, credentials)
    dataset_service = DatasetService(storage, credentials)

    context = AppContext(
        settings_service=service,
        settings=resolved,
        storage=storage,
        state=state,
        monitor=monitor,
        registry=registry,
        job_manager=jobs,
        credentials=credentials,
        model_service=model_service,
        dataset_service=dataset_service,
    )

    monitor.snapshot_ready.connect(state.publish_hardware)
    registry.statuses_changed.connect(state.publish_engines)
    jobs.job_changed.connect(state.publish_job)
    return context


def create_window(context: AppContext) -> MainWindow:
    window = MainWindow(context)
    context.window = window
    return window


def run(argv: list[str] | None = None) -> int:
    """Start the desktop application. Returns the Qt exit code."""
    log_path = configure_logging(paths.log_file())
    app = create_application(list(sys.argv if argv is None else argv))
    apply_theme(app)

    try:
        context = build_context()
        window = create_window(context)
        window.show()
        window.start()
    except Exception:  # noqa: BLE001 - last-resort startup guard
        logger.exception("Fatal error during application startup")
        QMessageBox.critical(
            None,
            "Simple HF Trainer",
            "The application could not start. See the log file for details:\n"
            f"{log_path or paths.log_file()}",
        )
        return 1

    logger.info("Application started (log: %s)", log_path)
    logger.info("Python %s", sys.version.split()[0])
    return app.exec()

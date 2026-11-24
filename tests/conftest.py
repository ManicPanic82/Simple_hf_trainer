"""Shared pytest configuration for the Simple HF Trainer test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from services.settings_service import (  # noqa: E402
    AppSettings,
    EngineSettings,
    PathSettings,
    UiSettings,
)


@pytest.fixture(scope="session")
def qapp():
    from ui.theme import apply_theme

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication([])
    apply_theme(app)
    return app


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    """Settings pointing at real, existing directories under ``tmp_path``."""
    folders = {
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "outputs_dir": tmp_path / "outputs",
        "hf_cache_dir": tmp_path / "hf-cache",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        paths=PathSettings(**folders),
        engines=EngineSettings(),
        ui=UiSettings(poll_interval_ms=1000),
    )


@pytest.fixture
def context(qapp, app_settings: AppSettings, tmp_path: Path):
    """AppContext with isolated settings and no running background work."""
    from app.application import build_context
    from services.settings_service import SettingsService

    service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=PROJECT_ROOT / "config" / "defaults.json",
    )
    ctx = build_context(service, settings=app_settings)
    yield ctx
    ctx.monitor.stop()
    ctx.registry.wait_for_idle(2000)


@pytest.fixture
def wait_for():
    """Wait for ``predicate`` while running the Qt event loop."""

    def _wait(predicate, timeout_ms: int = 5000, interval_ms: int = 10) -> bool:
        if predicate():
            return True
        loop = QEventLoop()
        result = {"done": False}
        timer = QTimer()
        timer.setInterval(interval_ms)

        def check() -> None:
            if predicate():
                result["done"] = True
                loop.quit()

        timer.timeout.connect(check)
        QTimer.singleShot(timeout_ms, loop.quit)
        timer.start()
        loop.exec()
        timer.stop()
        return result["done"] or bool(predicate())

    return _wait

"""Shared "open or create this storage folder" action."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from app.state import AppState
from services.storage_service import StorageService


def open_storage_path(state: AppState, storage: StorageService, label: str) -> None:
    """Create the configured directory (if missing) and open it in Explorer.

    Never deletes or moves anything. Failures are reported as a notice.
    """
    path = storage.path_for(label)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        state.notify("error", f"Could not create {path}: {error}")
        return
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        state.notify("warning", f"Could not open {path} in Explorer.")

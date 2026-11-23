"""Models tab: Hugging Face discovery, model downloads, and local model library."""

from __future__ import annotations

import threading
from contextlib import suppress
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.state import AppState
from services.model_service import DownloadCancelled, HubModelRecord, LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage


class ModelSearchWorker(QThread):
    """Background worker for searching Hugging Face Hub."""

    results_ready = Signal(list)
    search_failed = Signal(str)

    def __init__(self, service: ModelService, query: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.query = query

    def run(self):
        try:
            records = self.service.search_hub(self.query)
            self.results_ready.emit(records)
        except Exception as error:
            self.search_failed.emit(str(error))


class LocalModelScanWorker(QThread):
    """Discover local models without blocking the Qt event loop."""

    results_ready = Signal(list)
    scan_failed = Signal(str)

    def __init__(self, service: ModelService, parent=None):
        super().__init__(parent)
        self.service = service

    def run(self) -> None:
        try:
            self.results_ready.emit(self.service.scan_local_models())
        except Exception as error:
            self.scan_failed.emit(str(error))


class ModelInfoWorker(QThread):
    """Fetch selected model metadata without blocking the Qt event loop."""

    info_ready = Signal(str, dict)
    info_failed = Signal(str, str)

    def __init__(self, service: ModelService, repo_id: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.repo_id = repo_id

    def run(self) -> None:
        try:
            self.info_ready.emit(self.repo_id, self.service.get_hub_model_info(self.repo_id))
        except Exception as error:
            self.info_failed.emit(self.repo_id, str(error))


class ModelDownloadWorker(QThread):
    """Background worker for a cooperative, per-file model download."""

    progress = Signal(float, str)
    download_finished = Signal(Path)
    download_failed = Signal(str)
    download_cancelled = Signal(str, Path, int, int)

    def __init__(self, service: ModelService, repo_id: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.repo_id = repo_id
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation; the service checks this between files."""
        self._cancel_event.set()

    def run(self):
        try:
            target = self.service.download_model(
                self.repo_id,
                progress_callback=lambda frac, msg: self.progress.emit(frac, msg),
                cancel=self._cancel_event,
            )
            self.download_finished.emit(target)
        except DownloadCancelled as error:
            self.download_cancelled.emit(
                str(error), error.destination, error.downloaded, error.total
            )
        except Exception as error:
            self.download_failed.emit(str(error))


class ModelsTab(TabPage):
    use_model_in_training = Signal(str)
    use_model_in_heretic = Signal(str)

    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        model_service: ModelService | None = None,
        parent: QWidget | None = None,
        *,
        scan_on_init: bool = True,
    ) -> None:
        super().__init__(
            "Models",
            "Find, download, and inspect Hugging Face models, and reuse them in "
            "Training or Heretic.",
            parent,
        )
        self._state = state
        self._storage = storage
        self._service = model_service or ModelService(storage)
        self._search_worker: ModelSearchWorker | None = None
        self._scan_worker: LocalModelScanWorker | None = None
        self._scan_pending = False
        self._scan_pending_publish = False
        self._scan_publish = False
        self._download_worker: ModelDownloadWorker | None = None
        self._info_workers: list[ModelInfoWorker] = []
        self._selected_hub_model: HubModelRecord | None = None
        self._selected_local_model: LocalModelRecord | None = None
        self._local_models: list[LocalModelRecord] = []
        self._publishing_models = False

        # --- 1. Hub Search Card ---
        search = Card(
            "Search Hugging Face Hub",
            "Discover compatible models on the Hub, inspect their parameters and file sizes, "
            "and download them directly to your local library.",
        )
        self.search_status_pill = Pill("Ready", "info", search)
        search.add_header_widget(self.search_status_pill)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_input = QLineEdit(search)
        self.search_input.setPlaceholderText(
            "Search models (e.g. Qwen2.5-0.5B, Llama-3.2-1B, Gemma)..."
        )
        self.search_input.returnPressed.connect(self._start_search)
        search_row.addWidget(self.search_input, 1)

        self.search_button = QPushButton("Search", search)
        self.search_button.clicked.connect(self._start_search)
        search_row.addWidget(self.search_button)
        search.add_layout(search_row)

        self.results_table = QTableWidget(0, 4, search)
        self.results_table.setHorizontalHeaderLabels(["Model ID", "Author", "Downloads", "Likes"])
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_hub_model_selected)
        self.results_table.setMaximumHeight(160)
        search.add(self.results_table)

        details_box = QVBoxLayout()
        details_box.setSpacing(6)
        self.hub_model_row = FieldRow("Selected", "None selected", caption_width=104)
        self.hub_files_row = FieldRow("Files / Size", "—", caption_width=104)
        details_box.addWidget(self.hub_model_row)
        details_box.addWidget(self.hub_files_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.download_button = QPushButton("Download Model", search)
        self.download_button.setObjectName("Primary")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._start_download)
        action_row.addWidget(self.download_button)

        self.cancel_download_button = QPushButton("Cancel", search)
        self.cancel_download_button.setEnabled(False)
        self.cancel_download_button.clicked.connect(self._cancel_download)
        action_row.addWidget(self.cancel_download_button)

        self.download_progress = QProgressBar(search)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        action_row.addWidget(self.download_progress, 1)

        self.download_status_label = QLabel("", search)
        self.download_status_label.setObjectName("Muted")
        action_row.addWidget(self.download_status_label)
        details_box.addLayout(action_row)

        search.add_layout(details_box)
        self.add(search)

        # --- 2. Local Library Card ---
        library = Card(
            "Local model library",
            "Downloaded models, fine-tuned LoRA adapters, and Heretic modified models "
            "ready for training or abliteration.",
        )
        self.library_pill = Pill("0 models", "off", library)
        library.add_header_widget(self.library_pill)

        self.models_path = StoragePathRow("models_dir", "Models folder", library)
        self.models_path.open_requested.connect(self._open_storage)
        library.add(self.models_path)

        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)
        self.scan_button = QPushButton("Scan Library", library)
        self.scan_button.clicked.connect(self.scan_library)
        scan_row.addWidget(self.scan_button)
        scan_row.addStretch(1)
        library.add_layout(scan_row)

        self.library_table = QTableWidget(0, 5, library)
        self.library_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Format", "Size", "6 GB VRAM"]
        )
        self.library_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.library_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.library_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.library_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.library_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.library_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.library_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.library_table.itemSelectionChanged.connect(self._on_local_model_selected)
        self.library_table.setMaximumHeight(180)
        library.add(self.library_table)

        reuse_row = QHBoxLayout()
        reuse_row.setSpacing(10)
        self.use_training_btn = QPushButton("Use in Training", library)
        self.use_training_btn.setEnabled(False)
        self.use_training_btn.clicked.connect(self._on_use_in_training)
        reuse_row.addWidget(self.use_training_btn)

        self.use_heretic_btn = QPushButton("Use in Heretic", library)
        self.use_heretic_btn.setEnabled(False)
        self.use_heretic_btn.clicked.connect(self._on_use_in_heretic)
        reuse_row.addWidget(self.use_heretic_btn)
        reuse_row.addStretch(1)
        library.add_layout(reuse_row)

        self.add(library)

        # --- 3. Format and Compatibility Checks Card ---
        formats = Card(
            "Format and compatibility checks",
            "Hardware assessment for the reference NVIDIA GeForce GTX 1660 (6 GB VRAM, Compute 7.5).",
        )
        self.compat_model_row = FieldRow("Model", "None selected", caption_width=120)
        self.compat_arch_row = FieldRow("Architecture", "—", caption_width=120)
        self.compat_assessment_row = FieldRow(
            "Assessment", "Select a model above to assess VRAM fit", caption_width=120
        )
        formats.add(self.compat_model_row)
        formats.add(self.compat_arch_row)
        formats.add(self.compat_assessment_row)
        self.add(formats)

        self.finish()
        self.refresh()
        if scan_on_init:
            self.scan_library()

        # Training/Heretic publish their outputs through shared state. Rescan
        # from disk when that happens; connect after the startup scan so this
        # does not trigger a second, duplicate scan during construction.
        if not getattr(self, "_models_changed_connected", False):
            self._state.models_changed.connect(self.scan_local_models)
            self._models_changed_connected = True

    def _open_storage(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def refresh(self) -> None:
        validation = self._storage.validate("models_dir")
        errors = [issue for issue in validation.issues if issue.severity is Severity.ERROR]
        note = "\n".join(str(issue) for issue in validation.issues)
        self.models_path.set_path(
            validation.path,
            ok=validation.ok,
            exists=validation.exists,
            note=note,
            severity="error" if errors else "warn",
        )
        can_use = validation.ok
        self.search_button.setEnabled(can_use)
        self.search_input.setEnabled(can_use)
        self.set_status(
            "Model library ready" if can_use else "Model storage needs attention",
            "ok" if can_use else ("error" if errors else "warn"),
        )

    # --- Hub search and download ---

    def _start_search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return
        self.search_button.setEnabled(False)
        self.search_status_pill.set_state("warn", "Searching…")
        self._search_worker = ModelSearchWorker(self._service, query, self)
        self._search_worker.results_ready.connect(self._on_search_results)
        self._search_worker.search_failed.connect(self._on_search_failed)
        self._search_worker.start()

    def _on_search_results(self, records: list[HubModelRecord]) -> None:
        self.search_button.setEnabled(True)
        self.results_table.setRowCount(len(records))
        self._search_records = records
        for row, rec in enumerate(records):
            id_item = QTableWidgetItem(rec.repo_id)
            id_item.setData(Qt.ItemDataRole.UserRole, rec)
            self.results_table.setItem(row, 0, id_item)
            self.results_table.setItem(row, 1, QTableWidgetItem(rec.author))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{rec.downloads:,}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{rec.likes:,}"))

        self.search_status_pill.set_state("ok" if records else "off", f"{len(records)} found")

    def _on_search_failed(self, error: str) -> None:
        self.search_button.setEnabled(True)
        self.search_status_pill.set_state("error", "Search error")
        self._state.notify("error", f"Hub search failed: {error}")

    def _on_hub_model_selected(self) -> None:
        selected = self.results_table.selectedItems()
        if not selected:
            self._selected_hub_model = None
            self.download_button.setEnabled(False)
            return

        row = selected[0].row()
        item = self.results_table.item(row, 0)
        rec: HubModelRecord = item.data(Qt.ItemDataRole.UserRole)
        self._selected_hub_model = rec
        self.hub_model_row.set_value(rec.repo_id)
        self.download_button.setEnabled(True)
        self.download_status_label.setText("")

        self.hub_files_row.set_value("Loading model metadata…")
        self.compat_model_row.set_value(rec.repo_id)
        self.compat_arch_row.set_value("Checking Hub metadata…")
        self.compat_assessment_row.set_value("Estimating memory requirements…")

        worker = ModelInfoWorker(self._service, rec.repo_id, self)
        self._info_workers.append(worker)
        worker.info_ready.connect(self._on_hub_model_info)
        worker.info_failed.connect(self._on_hub_model_info_failed)
        worker.finished.connect(lambda worker=worker: self._discard_info_worker(worker))
        worker.start()

    def _discard_info_worker(self, worker: ModelInfoWorker) -> None:
        if worker in self._info_workers:
            self._info_workers.remove(worker)

    def _on_hub_model_info(self, repo_id: str, info: dict) -> None:
        if self._selected_hub_model is None or self._selected_hub_model.repo_id != repo_id:
            return
        size_mb = info.get("total_size_mb", 0.0)
        files = info.get("files", [])
        files_count = len(files)
        self.hub_files_row.set_value(f"{files_count} files ({size_mb:.1f} MB)")
        # The count alone hides what will actually be downloaded; expose the
        # repository file names on hover without cluttering the layout.
        names = [f.get("filename", "") for f in files if isinstance(f, dict)]
        self.hub_files_row.value.setToolTip(
            "\n".join(names[:40]) + ("\n…" if len(names) > 40 else "")
            if names
            else "No file list available from the Hub"
        )
        self.compat_arch_row.set_value(info.get("pipeline_tag") or "Unknown")
        self.compat_assessment_row.set_value(
            info.get("vram_detail") or "Runtime memory estimate unavailable"
        )

    def _on_hub_model_info_failed(self, repo_id: str, error: str) -> None:
        if self._selected_hub_model is None or self._selected_hub_model.repo_id != repo_id:
            return
        self.hub_files_row.set_value("Metadata unavailable")
        self.compat_arch_row.set_value("—")
        self.compat_assessment_row.set_value(f"Could not retrieve model metadata: {error}")

    def _start_download(self) -> None:
        if not self._selected_hub_model:
            return
        repo_id = self._selected_hub_model.repo_id
        self.download_button.setEnabled(False)
        self.cancel_download_button.setEnabled(True)
        self.cancel_download_button.setText("Cancel")
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status_label.setText(f"Downloading {repo_id}...")

        self._download_worker = ModelDownloadWorker(self._service, repo_id, self)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.download_finished.connect(self._on_download_finished)
        self._download_worker.download_failed.connect(self._on_download_failed)
        self._download_worker.download_cancelled.connect(self._on_download_cancelled)
        self._download_worker.start()

    def _cancel_download(self) -> None:
        worker = self._download_worker
        if worker is None or not worker.isRunning():
            return
        self.cancel_download_button.setEnabled(False)
        self.cancel_download_button.setText("Cancelling…")
        self.download_status_label.setText("Cancelling — stopping after the current file...")
        worker.cancel()

    def _on_download_progress(self, fraction: float, message: str) -> None:
        # ``fraction`` is completed files / total files, so this bar reflects
        # real file counts rather than an invented byte/percentage estimate.
        self.download_progress.setValue(max(0, min(100, int(fraction * 100))))
        self.download_status_label.setText(message)

    def _reset_download_controls(self) -> None:
        self.download_button.setEnabled(True)
        self.cancel_download_button.setEnabled(False)
        self.cancel_download_button.setText("Cancel")

    def _on_download_finished(self, destination: Path) -> None:
        self._reset_download_controls()
        self.download_progress.setValue(100)
        self.download_status_label.setText("Download complete")
        self._state.notify("info", f"Model downloaded to {destination.name}")
        self.scan_library()

    def _on_download_failed(self, error: str) -> None:
        self._reset_download_controls()
        self.download_progress.setVisible(False)
        self.download_status_label.setText("Download failed")
        self._state.notify("error", f"Model download failed: {error}")

    def _on_download_cancelled(
        self, reason: str, destination: Path, downloaded: int, total: int
    ) -> None:
        self._reset_download_controls()
        self.download_progress.setValue(int(100 * downloaded / total) if total else 0)
        self.download_status_label.setText(
            f"Cancelled after {downloaded}/{total} files; "
            f"partial files kept in {destination.name} (no mid-file cancellation)."
        )
        self._state.notify("warning", f"Download cancelled: {reason}")

    # --- Local library ---

    def scan_library(self) -> None:
        """Start a background scan and publish its result when it completes."""
        self._scan_library(publish=True)

    def scan_local_models(self) -> None:
        """Refresh after model changes without recursively publishing again."""
        if self._publishing_models:
            return
        self._scan_library(publish=False)

    def _scan_library(self, *, publish: bool) -> None:
        worker = self._scan_worker
        if worker is not None and worker.isRunning():
            self._scan_pending = True
            self._scan_pending_publish = self._scan_pending_publish or publish
            return

        self._scan_publish = publish
        self.scan_button.setEnabled(False)
        self.library_pill.set_state("warn", "Scanning…")
        self._scan_worker = LocalModelScanWorker(self._service, self)
        self._scan_worker.results_ready.connect(self._on_local_scan_results)
        self._scan_worker.scan_failed.connect(self._on_local_scan_failed)
        self._scan_worker.finished.connect(self._on_local_scan_finished)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.start()

    def _on_local_scan_results(self, records: list[LocalModelRecord]) -> None:
        self._local_models = records
        self.library_table.setRowCount(len(records))
        for row, rec in enumerate(records):
            name_item = QTableWidgetItem(rec.name)
            name_item.setData(Qt.ItemDataRole.UserRole, rec)
            self.library_table.setItem(row, 0, name_item)
            self.library_table.setItem(row, 1, QTableWidgetItem(rec.kind.capitalize()))
            self.library_table.setItem(row, 2, QTableWidgetItem(rec.format.upper()))
            self.library_table.setItem(row, 3, QTableWidgetItem(f"{rec.size_mb:.1f} MB"))
            self.library_table.setItem(row, 4, QTableWidgetItem(rec.status_detail))

        self.library_pill.set_state("ok" if records else "off", f"{len(records)} models")
        if self._scan_publish:
            self._publishing_models = True
            try:
                self._state.publish_models(records)
            finally:
                self._publishing_models = False

    def _on_local_scan_failed(self, error: str) -> None:
        self.library_pill.set_state("error", "Scan failed")
        self._state.notify("error", f"Local model scan failed: {error}")

    def _on_local_scan_finished(self) -> None:
        self.scan_button.setEnabled(True)
        self._scan_worker = None
        if self._scan_pending:
            publish = self._scan_pending_publish
            self._scan_pending = False
            self._scan_pending_publish = False
            self._scan_library(publish=publish)

    def _on_local_model_selected(self) -> None:
        selected = self.library_table.selectedItems()
        if not selected:
            self._selected_local_model = None
            self.use_training_btn.setEnabled(False)
            self.use_heretic_btn.setEnabled(False)
            return

        row = selected[0].row()
        item = self.library_table.item(row, 0)
        rec: LocalModelRecord = item.data(Qt.ItemDataRole.UserRole)
        self._selected_local_model = rec
        self.use_training_btn.setEnabled(True)
        self.use_heretic_btn.setEnabled(True)

        self.compat_model_row.set_value(f"{rec.name} ({rec.kind})")
        self.compat_arch_row.set_value(rec.architecture)
        self.compat_assessment_row.set_value(rec.status_detail)

    def _on_use_in_training(self) -> None:
        if self._selected_local_model:
            self.use_model_in_training.emit(str(self._selected_local_model.path))

    def _on_use_in_heretic(self) -> None:
        if self._selected_local_model:
            self.use_model_in_heretic.emit(str(self._selected_local_model.path))

    def request_shutdown(self) -> None:
        """Stop background Hub threads when the application closes."""
        workers: list[QThread] = []
        download = self._download_worker
        if download is not None:
            cancel = getattr(download, "cancel", None)
            if callable(cancel):
                with suppress(Exception):
                    cancel()
            workers.append(download)
        if self._search_worker is not None:
            workers.append(self._search_worker)
        if self._scan_worker is not None:
            workers.append(self._scan_worker)
        workers.extend(self._info_workers)
        for worker in workers:
            with suppress(Exception):
                worker.requestInterruption()
                worker.quit()
        for worker in workers:
            with suppress(Exception):
                worker.wait(500)

"""Datasets tab: discovery, preview, structure validation, and training preparation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
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
from services.dataset_service import (
    DatasetPreview,
    DatasetService,
    HubDatasetRecord,
    LocalDatasetRecord,
)
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage

# Data file types that can represent a dataset split/file on disk.
_DATA_FILE_SUFFIXES = frozenset({".jsonl", ".json", ".csv", ".parquet", ".arrow"})


class DatasetSearchWorker(QThread):
    results_ready = Signal(list)
    search_failed = Signal(str)

    def __init__(self, service: DatasetService, query: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.query = query

    def run(self):
        try:
            records = self.service.search_hub(self.query)
            self.results_ready.emit(records)
        except Exception as error:
            self.search_failed.emit(str(error))


class LocalDatasetScanWorker(QThread):
    """Discover local datasets without blocking the Qt event loop."""

    results_ready = Signal(list)
    scan_failed = Signal(str)

    def __init__(self, service: DatasetService, parent=None):
        super().__init__(parent)
        self.service = service

    def run(self) -> None:
        try:
            self.results_ready.emit(self.service.scan_local_datasets())
        except Exception as error:
            self.scan_failed.emit(str(error))


class DatasetDownloadWorker(QThread):
    """Background worker that imports a Hub dataset snapshot into the local library."""

    progress = Signal(float, str)
    download_finished = Signal(Path)
    download_failed = Signal(str)

    def __init__(
        self,
        service: DatasetService,
        repo_id: str,
        target_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.repo_id = repo_id
        self.target_dir = target_dir

    def run(self):
        try:
            target = self.service.download_dataset(
                self.repo_id,
                self.target_dir,
                progress=lambda frac, msg: self.progress.emit(frac, msg),
            )
            self.download_finished.emit(target)
        except Exception as error:
            self.download_failed.emit(str(error))


class DatasetsTab(TabPage):
    use_dataset_in_training = Signal(str)

    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        dataset_service: DatasetService | None = None,
        parent: QWidget | None = None,
        *,
        scan_on_init: bool = True,
    ) -> None:
        super().__init__(
            "Datasets",
            "Find or import training data, preview what it contains, and prepare it for "
            "a fine-tuning run.",
            parent,
        )
        self._state = state
        self._storage = storage
        self._service = dataset_service or DatasetService(storage)
        self._search_worker: DatasetSearchWorker | None = None
        self._scan_worker: LocalDatasetScanWorker | None = None
        self._scan_pending = False
        self._download_worker: DatasetDownloadWorker | None = None
        self._selected_hub_dataset: HubDatasetRecord | None = None
        self._selected_local_dataset: LocalDatasetRecord | None = None
        self._local_datasets: list[LocalDatasetRecord] = []
        self._selected_split_path: Path | None = None
        self._selected_split_valid = False

        # --- 1. Hub Search Card ---
        search = Card(
            "Search Hugging Face datasets",
            "Search public instruction and conversational datasets on the Hugging Face Hub.",
        )
        self.search_status_pill = Pill("Ready", "info", search)
        search.add_header_widget(self.search_status_pill)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_input = QLineEdit(search)
        self.search_input.setPlaceholderText(
            "Search datasets (e.g. alpaca, dolly, oasst, python)..."
        )
        self.search_input.returnPressed.connect(self._start_search)
        search_row.addWidget(self.search_input, 1)

        self.search_button = QPushButton("Search", search)
        self.search_button.clicked.connect(self._start_search)
        search_row.addWidget(self.search_button)
        search.add_layout(search_row)

        self.results_table = QTableWidget(0, 4, search)
        self.results_table.setHorizontalHeaderLabels(["Dataset ID", "Author", "Downloads", "Likes"])
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
        self.results_table.itemSelectionChanged.connect(self._on_hub_dataset_selected)
        self.results_table.setMaximumHeight(150)
        search.add(self.results_table)

        download_row = QHBoxLayout()
        download_row.setSpacing(8)
        self.download_button = QPushButton("Download / Import", search)
        self.download_button.setObjectName("Primary")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._start_download)
        download_row.addWidget(self.download_button)

        self.download_progress = QProgressBar(search)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        download_row.addWidget(self.download_progress, 1)

        self.download_status_label = QLabel("", search)
        self.download_status_label.setObjectName("Muted")
        download_row.addWidget(self.download_status_label)
        search.add_layout(download_row)
        self.add(search)

        # --- 2. Local Datasets and Preparation Card ---
        local = Card(
            "Local datasets and preparation",
            "Import and prepare training-ready JSONL files formatted with instruction/response templates.",
        )
        self.local_pill = Pill("0 datasets", "off", local)
        local.add_header_widget(self.local_pill)

        self.datasets_path = StoragePathRow("datasets_dir", "Datasets folder", local)
        self.datasets_path.open_requested.connect(self._open_datasets)
        local.add(self.datasets_path)

        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)
        self.scan_button = QPushButton("Scan Datasets", local)
        self.scan_button.clicked.connect(self.scan_datasets)
        scan_row.addWidget(self.scan_button)
        scan_row.addStretch(1)
        local.add_layout(scan_row)

        self.local_table = QTableWidget(0, 5, local)
        self.local_table.setHorizontalHeaderLabels(["Name", "Format", "Rows", "Size", "Status"])
        self.local_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.local_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.local_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.local_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.local_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.local_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.local_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.local_table.itemSelectionChanged.connect(self._on_local_dataset_selected)
        self.local_table.setMaximumHeight(160)
        local.add(self.local_table)

        # Preparation box
        prep_layout = QVBoxLayout()
        prep_layout.setSpacing(8)
        prep_label = QLabel("Format conversion (Alpaca instruction template)", local)
        prep_label.setObjectName("SectionLabel")
        prep_layout.addWidget(prep_label)

        grid = QGridLayout()
        grid.setSpacing(8)

        grid.addWidget(QLabel("Instruction column:", local), 0, 0)
        self.inst_combo = QComboBox(local)
        self.inst_combo.setEditable(True)
        grid.addWidget(self.inst_combo, 0, 1)

        grid.addWidget(QLabel("Input column (optional):", local), 0, 2)
        self.inp_combo = QComboBox(local)
        self.inp_combo.setEditable(True)
        grid.addWidget(self.inp_combo, 0, 3)

        grid.addWidget(QLabel("Output column:", local), 1, 0)
        self.out_combo = QComboBox(local)
        self.out_combo.setEditable(True)
        grid.addWidget(self.out_combo, 1, 1)

        self.prepare_button = QPushButton("Prepare for Training", local)
        self.prepare_button.setObjectName("Primary")
        self.prepare_button.setEnabled(False)
        self.prepare_button.clicked.connect(self._prepare_dataset)
        grid.addWidget(self.prepare_button, 1, 3)

        prep_layout.addLayout(grid)
        local.add_layout(prep_layout)
        self.add(local)

        # --- 3. Preview and Validation Card ---
        preview = Card(
            "Preview and validation",
            "Inspect sample rows and verify training column structure.",
        )
        self.val_pill = Pill("No selection", "off", preview)
        preview.add_header_widget(self.val_pill)

        split_row = QHBoxLayout()
        split_row.setSpacing(8)
        self.split_label = QLabel("Split / file:", preview)
        self.split_label.setObjectName("Muted")
        self.split_label.setMinimumWidth(104)
        split_row.addWidget(self.split_label)
        self.split_combo = QComboBox(preview)
        self.split_combo.setEnabled(False)
        self.split_combo.setToolTip("Select a data file/split to preview and use")
        self.split_combo.currentIndexChanged.connect(self._sync_split_selection)
        split_row.addWidget(self.split_combo, 1)
        preview.add_layout(split_row)

        self.validation_row = FieldRow(
            "Validation", "Select a local dataset above to inspect", caption_width=104
        )
        self.columns_row = FieldRow("Columns", "—", caption_width=104)
        preview.add(self.validation_row)
        preview.add(self.columns_row)

        self.preview_table = QTableWidget(0, 0, preview)
        self.preview_table.setMaximumHeight(180)
        preview.add(self.preview_table)

        action_row = QHBoxLayout()
        self.use_in_training_btn = QPushButton("Use in Training", preview)
        self.use_in_training_btn.setEnabled(False)
        self.use_in_training_btn.clicked.connect(self._on_use_in_training)
        action_row.addWidget(self.use_in_training_btn)
        action_row.addStretch(1)
        preview.add_layout(action_row)

        self.add(preview)

        self.finish()
        self.refresh()
        if scan_on_init:
            self.scan_datasets()

    def _open_datasets(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def refresh(self) -> None:
        validation = self._storage.validate("datasets_dir")
        errors = [issue for issue in validation.issues if issue.severity is Severity.ERROR]
        note = "\n".join(str(issue) for issue in validation.issues)
        self.datasets_path.set_path(
            validation.path,
            ok=validation.ok,
            exists=validation.exists,
            note=note,
            severity="error" if errors else "warn",
        )
        self.set_status(
            "Dataset workspace ready" if validation.ok else "Dataset storage needs attention",
            "ok" if validation.ok else ("error" if errors else "warn"),
        )

    # --- Hub Search ---

    def _start_search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return
        self.search_button.setEnabled(False)
        self.search_status_pill.set_state("warn", "Searching…")
        self._search_worker = DatasetSearchWorker(self._service, query, self)
        self._search_worker.results_ready.connect(self._on_search_results)
        self._search_worker.search_failed.connect(self._on_search_failed)
        self._search_worker.start()

    def _on_search_results(self, records: list[HubDatasetRecord]) -> None:
        self.search_button.setEnabled(True)
        self.results_table.setRowCount(len(records))
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
        self._state.notify("error", f"Hub dataset search failed: {error}")

    def _on_hub_dataset_selected(self) -> None:
        selected = self.results_table.selectedItems()
        if not selected:
            self._selected_hub_dataset = None
            self.download_button.setEnabled(False)
            return

        row = selected[0].row()
        item = self.results_table.item(row, 0)
        self._selected_hub_dataset = item.data(Qt.ItemDataRole.UserRole)
        self.download_button.setEnabled(True)
        self.download_status_label.setText("")

    def _start_download(self) -> None:
        if not self._selected_hub_dataset:
            return
        repo_id = self._selected_hub_dataset.repo_id
        target_dir = self._storage.path_for("datasets_dir") / repo_id.replace("/", "--")

        self.download_button.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.download_status_label.setText(f"Starting download of {repo_id}…")

        self._download_worker = DatasetDownloadWorker(
            self._service,
            repo_id,
            target_dir,
            self,
        )
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.download_finished.connect(self._on_download_finished)
        self._download_worker.download_failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_progress(self, fraction: float, message: str) -> None:
        self.download_progress.setValue(int(fraction * 100))
        self.download_status_label.setText(message)

    def _on_download_finished(self, destination: Path) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setValue(100)
        self.download_status_label.setText("Download complete")
        self._state.notify("info", f"Dataset imported to {destination.name}")
        self.scan_datasets()

    def _on_download_failed(self, error: str) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setVisible(False)
        self.download_status_label.setText("Download failed")
        self._state.notify("error", f"Dataset download failed: {error}")

    # --- Local Datasets ---

    def scan_datasets(self) -> None:
        """Start a background scan; only update widgets from the result slot."""
        worker = self._scan_worker
        if worker is not None and worker.isRunning():
            self._scan_pending = True
            return
        self.scan_button.setEnabled(False)
        self.local_pill.set_state("warn", "Scanning…")
        self._scan_worker = LocalDatasetScanWorker(self._service, self)
        self._scan_worker.results_ready.connect(self._on_local_scan_results)
        self._scan_worker.scan_failed.connect(self._on_local_scan_failed)
        self._scan_worker.finished.connect(self._on_local_scan_finished)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.start()

    def _on_local_scan_results(self, records: list[LocalDatasetRecord]) -> None:
        self._local_datasets = records
        self.local_table.setRowCount(len(records))
        for row, rec in enumerate(records):
            name_item = QTableWidgetItem(rec.name)
            name_item.setData(Qt.ItemDataRole.UserRole, rec)
            self.local_table.setItem(row, 0, name_item)
            self.local_table.setItem(row, 1, QTableWidgetItem(rec.format.upper()))
            rows_text = f"{rec.row_count:,}" if rec.row_count is not None else "Unknown"
            self.local_table.setItem(row, 2, QTableWidgetItem(rows_text))
            self.local_table.setItem(row, 3, QTableWidgetItem(f"{rec.size_mb:.2f} MB"))
            status_text = "Ready for Training" if rec.is_prepared else "Needs Preparation"
            self.local_table.setItem(row, 4, QTableWidgetItem(status_text))

        self.local_pill.set_state("ok" if records else "off", f"{len(records)} datasets")
        self._state.publish_datasets(records)

    def _on_local_scan_failed(self, error: str) -> None:
        self.local_pill.set_state("error", "Scan failed")
        self._state.notify("error", f"Local dataset scan failed: {error}")

    def _on_local_scan_finished(self) -> None:
        self.scan_button.setEnabled(True)
        self._scan_worker = None
        if self._scan_pending:
            self._scan_pending = False
            self.scan_datasets()

    @staticmethod
    def _candidate_data_files(path: Path) -> list[Path]:
        """List real data files available for a selected dataset.

        A dataset may be a single file or a directory (for example a downloaded
        Hub snapshot whose splits are stored as ``train-00000-of-00001.parquet``
        files). File names are returned as-is; split names are never invented.
        """
        directory = path if path.is_dir() else path.parent
        candidates: dict[str, Path] = {}
        try:
            entries = list(directory.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in _DATA_FILE_SUFFIXES:
                candidates[entry.name] = entry
        # Always include the selected file itself, even with an unusual suffix.
        if path.is_file():
            candidates[path.name] = path
        return [candidates[name] for name in sorted(candidates, key=str.lower)]

    def _populate_split_combo(self, rec: LocalDatasetRecord) -> None:
        """Fill the split/file selector with the dataset's actual data files."""
        files = self._candidate_data_files(rec.path)
        with QSignalBlocker(self.split_combo):
            self.split_combo.clear()
            for file_path in files:
                self.split_combo.addItem(file_path.name, file_path)
            # A selector with a single choice adds no value.
            self.split_combo.setEnabled(len(files) > 1)
            if files:
                preferred = rec.path if rec.path.is_file() and rec.path in files else files[0]
                self.split_combo.setCurrentIndex(files.index(preferred))
        self.split_combo.setToolTip(
            "Choose a split/file to preview and use"
            if len(files) > 1
            else "Only one data file is available for this dataset"
        )
        self._sync_split_selection()

    def _clear_split_selection(self) -> None:
        """Reset the selector and preview when no local dataset is selected."""
        with QSignalBlocker(self.split_combo):
            self.split_combo.clear()
            self.split_combo.setEnabled(False)
        self._selected_split_path = None
        self._selected_split_valid = False
        self.prepare_button.setEnabled(False)
        self.use_in_training_btn.setEnabled(False)
        self.validation_row.set_value("Select a local dataset above to inspect")
        self.columns_row.set_value("—")
        self.val_pill.set_state("off", "No selection")
        self.preview_table.clear()
        self.preview_table.setColumnCount(0)
        self.preview_table.setRowCount(0)

    def _sync_split_selection(self, _index: int | None = None) -> None:
        """Apply the currently selected split/file to the preview and actions."""
        index = self.split_combo.currentIndex()
        path = self.split_combo.itemData(index, Qt.ItemDataRole.UserRole) if index >= 0 else None
        if not isinstance(path, Path):
            self._selected_split_path = None
            self._selected_split_valid = False
            self.prepare_button.setEnabled(False)
            self.use_in_training_btn.setEnabled(False)
            return

        self._selected_split_path = path
        self.prepare_button.setEnabled(self._selected_local_dataset is not None)
        self._show_preview(path)

    def _update_column_selectors(self, columns: list[str]) -> None:
        """Refresh the template column selectors for the selected split."""
        self.inst_combo.clear()
        self.inp_combo.clear()
        self.out_combo.clear()
        self.inp_combo.addItem("")
        for col in columns:
            self.inst_combo.addItem(col)
            self.inp_combo.addItem(col)
            self.out_combo.addItem(col)

        # Guess default column names
        for col in columns:
            lower = col.lower()
            if "instruct" in lower or "prompt" in lower or "question" in lower:
                self.inst_combo.setCurrentText(col)
            elif "input" in lower or "context" in lower:
                self.inp_combo.setCurrentText(col)
            elif (
                "output" in lower
                or "response" in lower
                or "answer" in lower
                or "completion" in lower
            ):
                self.out_combo.setCurrentText(col)

    def _on_local_dataset_selected(self) -> None:
        selected = self.local_table.selectedItems()
        if not selected:
            self._selected_local_dataset = None
            self._clear_split_selection()
            return

        row = selected[0].row()
        item = self.local_table.item(row, 0)
        rec: LocalDatasetRecord = item.data(Qt.ItemDataRole.UserRole)
        self._selected_local_dataset = rec
        self._populate_split_combo(rec)

    def _show_preview(self, file_path: Path) -> None:
        try:
            preview: DatasetPreview = self._service.preview_dataset(file_path, num_rows=10)
            self.columns_row.set_value(", ".join(preview.columns) if preview.columns else "None")
            self.validation_row.set_value(preview.validation_message)
            self.val_pill.set_state(
                "ok" if preview.is_valid_for_training else "warn",
                "Ready" if preview.is_valid_for_training else "Needs columns",
            )
            self._selected_split_valid = bool(preview.is_valid_for_training)
            if preview.columns:
                self._update_column_selectors(preview.columns)

            # Fill preview table
            cols = preview.columns
            self.preview_table.setColumnCount(len(cols))
            self.preview_table.setHorizontalHeaderLabels(cols)
            self.preview_table.setRowCount(len(preview.rows))
            for r_idx, row_dict in enumerate(preview.rows):
                for c_idx, col_name in enumerate(cols):
                    val_str = str(row_dict.get(col_name, ""))
                    # Truncate long strings for preview
                    if len(val_str) > 120:
                        val_str = val_str[:120] + "…"
                    self.preview_table.setItem(r_idx, c_idx, QTableWidgetItem(val_str))

            self.preview_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
        except Exception as error:
            self.validation_row.set_value(f"Preview error: {error}")
            self.val_pill.set_state("error", "Error")
            self._selected_split_valid = False
        self.use_in_training_btn.setEnabled(self._selected_split_valid)

    def _prepare_dataset(self) -> None:
        if not self._selected_local_dataset or not self._selected_split_path:
            return
        inst = self.inst_combo.currentText().strip()
        inp = self.inp_combo.currentText().strip()
        out = self.out_combo.currentText().strip()
        if not inst or not out:
            self._state.notify("warning", "Instruction and Output columns must be selected.")
            return

        try:
            output_file = self._service.prepare_dataset(
                self._selected_split_path,
                instruction_col=inst,
                input_col=inp,
                output_col=out,
            )
            self._state.notify("info", f"Prepared training dataset saved to {output_file}")
            self.scan_datasets()
        except Exception as error:
            self._state.notify("error", f"Preparation failed: {error}")

    def _on_use_in_training(self) -> None:
        if self._selected_split_path:
            self.use_dataset_in_training.emit(str(self._selected_split_path))

    def request_shutdown(self) -> None:
        """Stop background Hub threads when the application closes."""
        workers = [
            worker
            for worker in (self._download_worker, self._search_worker, self._scan_worker)
            if worker is not None
        ]
        for worker in workers:
            with suppress(Exception):
                worker.requestInterruption()
                worker.quit()
        for worker in workers:
            with suppress(Exception):
                worker.wait(500)

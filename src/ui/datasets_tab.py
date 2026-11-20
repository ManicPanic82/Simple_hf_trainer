"""Datasets tab: dataset discovery, preview, validation, and preparation."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from app.state import AppState
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage


class DatasetsTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Datasets",
            "Find or import training data, preview what it contains, and prepare it for "
            "a fine-tuning run.",
            parent,
        )
        self._state = state
        self._storage = storage

        search = Card(
            "Search Hugging Face datasets",
            "Hub search with split, column, and size information before anything is downloaded.",
        )
        search.add_header_widget(Pill("Not implemented yet", "warn", search))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.search_input = QLineEdit(search)
        self.search_input.setPlaceholderText("Search datasets — not implemented yet")
        self.search_input.setEnabled(False)
        row.addWidget(self.search_input, 1)
        self.search_button = QPushButton("Search", search)
        self.search_button.setEnabled(False)
        row.addWidget(self.search_button)
        search.add_layout(row)
        self.add(search)

        preview = Card(
            "Preview and validation",
            "Inspect rows, columns, and splits, then get actionable errors when a dataset "
            "does not match the selected training format.",
        )
        preview.add_header_widget(Pill("Not implemented yet", "warn", preview))
        preview.add(FieldRow("Row preview", "Not implemented yet", caption_width=104))
        preview.add(FieldRow("Validation", "Not implemented yet", caption_width=104))
        self.add(preview)

        local = Card(
            "Local datasets and preparation",
            "Import data from disk and prepare training-ready files with transparent "
            "settings and an explicit output location.",
        )
        local.add_header_widget(Pill("Not implemented yet", "warn", local))
        self.datasets_path = StoragePathRow("datasets_dir", "Datasets folder", local)
        self.datasets_path.open_requested.connect(self._open_datasets)
        local.add(self.datasets_path)
        local.add(
            FieldRow(
                "Preparation",
                "Not implemented yet — prepared datasets will be written inside the "
                "datasets folder and reused by Training.",
                caption_width=104,
            )
        )
        self.add(local)

        self.finish()
        self.refresh()

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

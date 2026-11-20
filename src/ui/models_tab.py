"""Models tab: Hugging Face discovery and the local model library."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QWidget

from app.state import AppState
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage


class ModelsTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Models",
            "Find, download, and inspect Hugging Face models, and reuse them in "
            "Training or Heretic.",
            parent,
        )
        self._state = state
        self._storage = storage

        search = Card(
            "Search Hugging Face Hub",
            "Hub search, model metadata, file listings, and authenticated downloads. "
            "The access token belongs in Windows Credential Manager, never in settings.",
        )
        search.add_header_widget(Pill("Not implemented yet", "warn", search))
        row = QHBoxLayout()
        row.setSpacing(8)
        self.search_input = QLineEdit(search)
        self.search_input.setPlaceholderText("Search models — not implemented yet")
        self.search_input.setEnabled(False)
        row.addWidget(self.search_input, 1)
        self.search_button = QPushButton("Search", search)
        self.search_button.setEnabled(False)
        row.addWidget(self.search_button)
        search.add_layout(row)
        search.add(
            FieldRow(
                "Planned",
                "Model metadata, file sizes, and 6 GB VRAM compatibility warnings.",
                caption_width=104,
            )
        )
        self.add(search)

        library = Card(
            "Local model library",
            "Downloaded models, fine-tuned adapters, and Heretic outputs are registered "
            "here so every workflow can reuse them.",
        )
        library.add_header_widget(Pill("Not implemented yet", "warn", library))
        self.models_path = StoragePathRow("models_dir", "Models folder", library)
        self.models_path.open_requested.connect(self._open_storage)
        library.add(self.models_path)
        self.library_state = FieldRow("Library scan", "Not implemented yet", caption_width=104)
        library.add(self.library_state)
        self.add(library)

        formats = Card(
            "Format and compatibility checks",
            "Safetensors, PyTorch bin, and GGUF detection, plus explicit warnings when a "
            "model cannot fit in 6 GB of VRAM.",
        )
        self.formats_row = FieldRow("Validation", "Not implemented yet", caption_width=104)
        formats.add(self.formats_row)
        self.add(formats)

        self.finish()
        self.refresh()

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

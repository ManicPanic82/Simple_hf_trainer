"""Heretic tab: training-free model modification (abliteration) via the real engine."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QPushButton, QWidget

from app.state import AppState
from services.engine_registry import HERETIC, EngineStatus
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from workers.job_manager import JobSnapshot


class HereticTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Heretic",
            "Modify a compatible model with the real Heretic workflow. It writes a new "
            "model; the original files are left untouched.",
            parent,
        )
        self._state = state
        self._storage = storage

        self.engine_card = EngineCard(
            "Heretic engine",
            "Heretic runs in its own Python environment and uses a separate worker process. "
            "It never shares the GPU with Unsloth.",
            self,
        )
        self.add(self.engine_card)

        setup = Card(
            "Model and output",
            "Pick a model that fits 6 GB of VRAM, choose what to modify, then run the real "
            "engine and inspect its output and logs.",
        )
        setup.add_header_widget(Pill("Compatibility testing required", "warn", setup))
        self.model_combo = QComboBox(setup)
        self.model_combo.addItem("No local models registered yet")
        self.model_combo.setEnabled(False)
        setup.add(
            FieldRow("Model", "Not implemented yet — model library pending", caption_width=104)
        )
        self.outputs_path = StoragePathRow("outputs_dir", "Output folder", setup)
        self.outputs_path.open_requested.connect(self._open_outputs)
        setup.add(self.outputs_path)
        self.run_state_row = FieldRow("Job state", "Idle", caption_width=104)
        setup.add(self.run_state_row)
        self.run_button = QPushButton("Run Heretic", setup)
        self.run_button.setObjectName("Primary")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Not implemented yet")
        setup.add(self.run_button)
        self.add(setup)

        notes = Card(
            "What to expect",
            "Heretic benchmarks the machine, optimises against refusal and divergence scores, "
            "and reports progress while it works. Expect minutes to hours depending on the "
            "model and the 6 GB VRAM budget.",
        )
        notes.add(
            FieldRow(
                "Licence",
                "Heretic is AGPL-3.0 and is invoked as an installed tool in its own "
                "environment; it is not bundled with this application.",
                caption_width=104,
            )
        )
        self.add(notes)

        self.finish()
        self.refresh()

        self._state.job_changed.connect(self.apply_job)
        self._state.engines_changed.connect(self.apply_engines)

    def _open_outputs(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def refresh(self) -> None:
        validation = self._storage.validate("outputs_dir")
        errors = [issue for issue in validation.issues if issue.severity is Severity.ERROR]
        note = "\n".join(str(issue) for issue in validation.issues)
        self.outputs_path.set_path(
            validation.path,
            ok=validation.ok,
            exists=validation.exists,
            note=note,
            severity="error" if errors else "warn",
        )
        self.apply_engines(self._state.engines)

    def apply_engines(self, statuses: dict[str, EngineStatus]) -> None:
        status = statuses.get(HERETIC)
        if status is not None:
            self.engine_card.apply_status(status)

    def apply_job(self, job: JobSnapshot | None) -> None:
        self.run_state_row.set_value("Idle" if job is None else f"{job.state.value} · {job.engine}")

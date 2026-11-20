"""Training tab: Unsloth LoRA/QLoRA job configuration and control."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QWidget

from app.state import AppState
from services.engine_registry import UNSLOTH, EngineStatus
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, Pill, StoragePathRow
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from workers.job_manager import JobSnapshot


class TrainingTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Training",
            "Fine-tune a compatible model with the real Unsloth engine in an isolated "
            "environment. Nothing is simulated.",
            parent,
        )
        self._state = state
        self._storage = storage

        self.engine_card = EngineCard(
            "Unsloth engine",
            "Fine-tuning runs in its own Python environment with its own PyTorch and CUDA "
            "build, separate from this application.",
            self,
        )
        self.add(self.engine_card)

        control = Card(
            "Job control",
            "LoRA and QLoRA are offered only where the selected model, the 6 GB GPU, and "
            "the installed Unsloth stack actually support the configuration.",
        )
        control.add_header_widget(Pill("Compatibility testing required", "warn", control))
        self.job_state_row = FieldRow("Job state", "Idle", caption_width=104)
        self.result_row = FieldRow("Last result", "—", caption_width=104)
        control.add(self.job_state_row)
        control.add(self.result_row)
        control.add(
            FieldRow(
                "GPU policy",
                "Exclusive GPU access. Unsloth and Heretic can never hold the GPU at the "
                "same time; a second request is rejected.",
                caption_width=104,
            )
        )
        self.start_button = QPushButton("Start training", control)
        self.start_button.setObjectName("Primary")
        self.start_button.setEnabled(False)
        self.start_button.setToolTip("Not implemented yet")
        control.add(self.start_button)
        self.add(control)

        outputs = Card(
            "Training outputs",
            "Adapters and logs are written to a job-specific folder that other workflows "
            "can reuse.",
        )
        self.outputs_path = StoragePathRow("outputs_dir", "Outputs folder", outputs)
        self.outputs_path.open_requested.connect(self._open_outputs)
        outputs.add(self.outputs_path)
        self.add(outputs)

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
        status = statuses.get(UNSLOTH)
        if status is not None:
            self.engine_card.apply_status(status)

    def apply_job(self, job: JobSnapshot | None) -> None:
        if job is None:
            self.job_state_row.set_value("Idle")
            return
        self.job_state_row.set_value(f"{job.state.value} · {job.engine}")
        if job.state.is_terminal:
            detail = job.message or job.state.value
            self.result_row.set_value(detail)

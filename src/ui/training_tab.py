"""Training tab: Unsloth LoRA/QLoRA job configuration, execution, and live monitoring."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from app.state import AppState
from services.dataset_service import DatasetService, LocalDatasetRecord
from services.engine_registry import UNSLOTH, EngineRegistry, EngineStatus
from services.model_service import LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, StoragePathRow
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from workers.job_manager import JobManager, JobSnapshot
from workers.protocol import MessageType, WorkerMessage
from workers.worker_process import WorkerProcess


class TrainingTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        job_manager: JobManager | None = None,
        registry: EngineRegistry | None = None,
        model_service: ModelService | None = None,
        dataset_service: DatasetService | None = None,
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
        self._job_manager = job_manager or JobManager(parent=self)
        self._registry = registry or EngineRegistry(storage.settings, parent=self)
        self._model_service = model_service or ModelService(storage)
        self._dataset_service = dataset_service or DatasetService(storage)

        self._active_worker: WorkerProcess | None = None
        self._active_job_id: str | None = None
        self._pending_completion: tuple[bool, str] | None = None

        # --- Engine Status Card ---
        self.engine_card = EngineCard(
            "Unsloth engine",
            "Fine-tuning runs in its own Python environment with its own PyTorch and CUDA "
            "build, separate from this application.",
            self,
        )
        self.add(self.engine_card)

        # --- Configuration Card ---
        cfg_card = Card(
            "Fine-tuning configuration",
            "Select a model and prepared dataset, and configure 4-bit LoRA hyperparameters.",
        )

        grid = QGridLayout()
        grid.setSpacing(10)

        # Model selection
        grid.addWidget(QLabel("Base Model:", cfg_card), 0, 0)
        self.model_combo = QComboBox(cfg_card)
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("Select local model or enter HF repo (e.g. unsloth/Qwen2.5-0.5B-Instruct)")
        grid.addWidget(self.model_combo, 0, 1, 1, 3)

        # Dataset selection
        grid.addWidget(QLabel("Training Dataset:", cfg_card), 1, 0)
        self.dataset_combo = QComboBox(cfg_card)
        self.dataset_combo.setEditable(True)
        self.dataset_combo.setPlaceholderText("Select prepared dataset from datasets folder")
        grid.addWidget(self.dataset_combo, 1, 1, 1, 3)

        # Hyperparameters
        grid.addWidget(QLabel("Max Steps:", cfg_card), 2, 0)
        self.steps_spin = QSpinBox(cfg_card)
        self.steps_spin.setRange(1, 10000)
        self.steps_spin.setValue(5)
        grid.addWidget(self.steps_spin, 2, 1)

        grid.addWidget(QLabel("Batch Size:", cfg_card), 2, 2)
        self.batch_spin = QSpinBox(cfg_card)
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(1)
        grid.addWidget(self.batch_spin, 2, 3)

        grid.addWidget(QLabel("Learning Rate:", cfg_card), 3, 0)
        self.lr_edit = QLineEdit("2e-4", cfg_card)
        grid.addWidget(self.lr_edit, 3, 1)

        grid.addWidget(QLabel("Max Seq Length:", cfg_card), 3, 2)
        self.seq_spin = QSpinBox(cfg_card)
        self.seq_spin.setRange(64, 4096)
        self.seq_spin.setSingleStep(64)
        self.seq_spin.setValue(256)
        grid.addWidget(self.seq_spin, 3, 3)

        grid.addWidget(QLabel("LoRA Rank (r):", cfg_card), 4, 0)
        self.r_spin = QSpinBox(cfg_card)
        self.r_spin.setRange(4, 64)
        self.r_spin.setValue(8)
        grid.addWidget(self.r_spin, 4, 1)

        grid.addWidget(QLabel("LoRA Alpha:", cfg_card), 4, 2)
        self.alpha_spin = QSpinBox(cfg_card)
        self.alpha_spin.setRange(8, 128)
        self.alpha_spin.setValue(16)
        grid.addWidget(self.alpha_spin, 4, 3)

        cfg_card.add_layout(grid)
        self.add(cfg_card)

        # --- Job Control and Live Progress Card ---
        control = Card(
            "Job control & live monitoring",
            "Exclusive GPU lock enforces that Unsloth and Heretic never share GPU memory simultaneously.",
        )
        self.job_state_row = FieldRow("Job state", "Idle", caption_width=104)
        self.result_row = FieldRow("Last result", "—", caption_width=104)
        control.add(self.job_state_row)
        control.add(self.result_row)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)
        self.step_label = QLabel("Step: —", control)
        self.step_label.setObjectName("StripValue")
        self.loss_label = QLabel("Loss: —", control)
        self.loss_label.setObjectName("StripValue")
        metrics_layout.addWidget(self.step_label)
        metrics_layout.addWidget(self.loss_label)
        metrics_layout.addStretch(1)
        control.add_layout(metrics_layout)

        self.progress_bar = QProgressBar(control)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control.add(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.start_button = QPushButton("Start training", control)
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._start_training)
        btn_row.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel job", control)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_training)
        btn_row.addWidget(self.cancel_button)
        btn_row.addStretch(1)
        control.add_layout(btn_row)

        log_label = QLabel("Live Engine Log", control)
        log_label.setObjectName("SectionLabel")
        control.add(log_label)

        self.log_view = QTextEdit(control)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setStyleSheet("font-family: 'Cascadia Mono', Consolas, monospace; font-size: 8.5pt;")
        control.add(self.log_view)

        self.add(control)

        # --- Outputs Card ---
        outputs = Card(
            "Training outputs",
            "Adapters and logs are written to a job-specific folder that other workflows can reuse.",
        )
        self.outputs_path = StoragePathRow("outputs_dir", "Outputs folder", outputs)
        self.outputs_path.open_requested.connect(self._open_outputs)
        outputs.add(self.outputs_path)
        self.last_adapter_row = FieldRow("Last adapter", "None saved yet", caption_width=104)
        outputs.add(self.last_adapter_row)
        self.add(outputs)

        self.finish()
        self.refresh()

        self._state.job_changed.connect(self.apply_job)
        self._state.engines_changed.connect(self.apply_engines)
        self._state.models_changed.connect(self._update_model_choices)
        self._state.datasets_changed.connect(self._update_dataset_choices)

    def _open_outputs(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def select_model(self, model_ref: str) -> None:
        idx = self.model_combo.findText(model_ref)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(model_ref)

    def select_dataset(self, dataset_path: str) -> None:
        idx = self.dataset_combo.findText(dataset_path)
        if idx >= 0:
            self.dataset_combo.setCurrentIndex(idx)
        else:
            self.dataset_combo.setEditText(dataset_path)

    def _update_model_choices(self, models: list[LocalModelRecord]) -> None:
        curr = self.model_combo.currentText()
        self.model_combo.clear()
        for m in models:
            if m.kind in ("base", "abliterated"):
                self.model_combo.addItem(f"{m.name} ({m.format.upper()})", str(m.path))
        if curr:
            self.select_model(curr)
        elif self.model_combo.count() == 0:
            self.model_combo.addItem("unsloth/Qwen2.5-0.5B-Instruct", "unsloth/Qwen2.5-0.5B-Instruct")

    def _update_dataset_choices(self, datasets: list[LocalDatasetRecord]) -> None:
        curr = self.dataset_combo.currentText()
        self.dataset_combo.clear()
        for d in datasets:
            self.dataset_combo.addItem(d.name, str(d.path))
        if curr:
            self.select_dataset(curr)

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
        if status is None:
            self.set_status("Checking Unsloth", "info")
            return

        self.engine_card.apply_status(status)
        is_avail = status.is_available
        has_no_active = self._active_worker is None or not self._active_worker.running
        self.start_button.setEnabled(is_avail and has_no_active)
        if is_avail:
            self.set_status("Unsloth engine available", "ok")
            self.start_button.setToolTip("Start Unsloth fine-tuning")
        else:
            level = "error" if status.badge() == "check failed" else "warn"
            self.set_status(f"Unsloth {status.badge()}", level)
            self.start_button.setToolTip(f"Unsloth engine unavailable: {status.detail}")

    def apply_job(self, job: JobSnapshot | None) -> None:
        if job is None:
            self.job_state_row.set_value("Idle")
            return
        self.job_state_row.set_value(f"{job.state.value} · {job.engine}")
        if job.state.is_terminal:
            detail = job.message or job.state.value
            self.result_row.set_value(detail)
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    def _start_training(self) -> None:
        model_name = self.model_combo.currentData() or self.model_combo.currentText().strip()
        if not model_name:
            self._state.notify("warning", "Please select or enter a model name to fine-tune.")
            return

        dataset_path = self.dataset_combo.currentData() or self.dataset_combo.currentText().strip()
        if not dataset_path or not Path(str(dataset_path)).is_file():
            self._state.notify(
                "error",
                "Select an existing local training dataset in the Datasets tab before starting fine-tuning.",
            )
            return

        # Check interpreter
        unsloth_status = self._registry.status(UNSLOTH)
        interpreter = unsloth_status.interpreter
        if not interpreter or not interpreter.is_file():
            # Fallback standard path
            fallback = Path("environments/unsloth/Scripts/python.exe").resolve()
            if fallback.is_file():
                interpreter = fallback
            else:
                self._state.notify("error", "Unsloth interpreter not found. Set it in Settings.")
                return

        worker_script = Path(__file__).resolve().parents[1] / "workers" / "unsloth_worker.py"
        if not worker_script.is_file():
            self._state.notify("error", f"Worker script not found: {worker_script}")
            return

        try:
            job = self._job_manager.submit(UNSLOTH, model_ref=model_name)
        except Exception as error:
            self._state.notify("error", f"Cannot start training: {error}")
            return

        try:
            self._active_job_id = job.job_id
            self._pending_completion = None
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.log_view.clear()
            self.step_label.setText("Step: 0")
            self.loss_label.setText("Loss: —")

            outputs_dir = self._storage.path_for("outputs_dir") / f"unsloth_{job.job_id}"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            try:
                lr_val = float(self.lr_edit.text().strip())
            except ValueError:
                lr_val = 2e-4

            config_data = {
                "job_id": job.job_id,
                "model_name": model_name,
                "dataset_path": dataset_path,
                "output_dir": str(outputs_dir),
                "max_steps": self.steps_spin.value(),
                "max_seq_length": self.seq_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": lr_val,
                "lora_r": self.r_spin.value(),
                "lora_alpha": self.alpha_spin.value(),
            }

            scratch_dir = self._storage.path_for("outputs_dir") / ".scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)

            worker = WorkerProcess(
                interpreter=interpreter,
                script=worker_script,
                args=["--config", json.dumps(config_data)],
                cwd=scratch_dir,
                parent=self,
            )
            self._active_worker = worker

            worker.started.connect(self._on_worker_started)
            worker.message_received.connect(self._on_worker_message)
            worker.exited.connect(self._on_worker_exited)
            worker.failed_to_start.connect(self._on_worker_failed_to_start)
            worker.start()
        except Exception as error:
            if self._active_worker is not None and self._active_worker.running:
                self._state.notify(
                    "error",
                    f"Training worker started but setup reported an error: {error}",
                )
            else:
                self._handle_setup_failure(job.job_id, error)

    def _handle_setup_failure(self, job_id: str, error: Exception) -> None:
        """Fail a job whose worker could not be prepared or started."""
        self._job_manager.mark_failed(
            job_id,
            message=f"Could not prepare training job: {error}",
        )
        self._active_job_id = None
        self._active_worker = None
        self._pending_completion = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._state.notify("error", f"Could not prepare training job: {error}")

    def _cancel_training(self) -> None:
        if self._active_job_id and self._active_worker and self._active_worker.running:
            self._job_manager.request_cancel(self._active_job_id)
            sent = self._active_worker.request_cancel(self._active_job_id)
            self.cancel_button.setEnabled(False)
            if sent:
                self._state.notify(
                    "info",
                    "Cancellation requested. The worker has up to 10 seconds to stop cleanly before it is terminated.",
                )
            else:
                self._state.notify("warning", "The cancellation request could not be sent; waiting for worker exit.")

    def _on_worker_started(self) -> None:
        if self._active_job_id and self._active_worker:
            self._job_manager.mark_running(self._active_job_id, pid=self._active_worker.pid)

    def _on_worker_message(self, msg: WorkerMessage) -> None:
        payload = msg.payload
        if msg.type is MessageType.PROGRESS:
            frac = float(payload.get("fraction", 0.0))
            self.progress_bar.setValue(int(frac * 100))
            step = payload.get("step")
            total = payload.get("total")
            if step is not None:
                self.step_label.setText(f"Step: {step}/{total}")

        elif msg.type is MessageType.METRIC:
            name = payload.get("name")
            val = payload.get("value")
            if name == "loss" and val is not None:
                self.loss_label.setText(f"Loss: {float(val):.4f}")

        elif msg.type is MessageType.LOG:
            line = payload.get("message", "")
            self.log_view.append(line)

        elif msg.type is MessageType.RESULT:
            out_dir = payload.get("output_dir", "")
            self.last_adapter_row.set_value(out_dir)

        elif msg.type is MessageType.JOB_COMPLETE:
            success = bool(payload.get("success", False))
            message = str(payload.get("message") or (
                "Training completed successfully." if success else "Training finished with error."
            ))
            self._pending_completion = (success, message)

    def _on_worker_exited(self, exit_code: int) -> None:
        if self._active_job_id:
            job_id = self._active_job_id
            snapshot = self._job_manager.snapshot(job_id)
            completion = self._pending_completion
            if snapshot is not None and snapshot.state.value == "cancelling":
                self._job_manager.handle_worker_exit(job_id, exit_code)
            elif completion is not None and completion[0] and exit_code == 0:
                self._job_manager.mark_succeeded(job_id, message=completion[1])
                self._model_service.scan_local_models()
                self._state.notify("info", "LoRA fine-tuning completed successfully!")
            elif completion is not None:
                self._job_manager.mark_failed(
                    job_id,
                    message=completion[1],
                    exit_code=exit_code,
                )
            else:
                self._job_manager.handle_worker_exit(job_id, exit_code)
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_worker_failed_to_start(self, error: str) -> None:
        if self._active_job_id:
            self._job_manager.mark_failed(self._active_job_id, message=f"Failed to start: {error}")
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._state.notify("error", f"Unsloth worker failed to start: {error}")

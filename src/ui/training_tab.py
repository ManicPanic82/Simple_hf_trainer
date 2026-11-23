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
    QWidget,
)

from app.state import AppState
from services.dataset_service import DatasetService, LocalDatasetRecord
from services.engine_registry import UNSLOTH, EngineRegistry, EngineStatus
from services.model_service import LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, StoragePathRow
from ui.console import ConsoleWidget
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from ui.training_dashboard import TrainingDashboard
from workers.job_manager import JobManager, JobSnapshot, JobState
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
        self._queued_jobs: dict[str, dict] = {}
        self._queue_connected = False

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
            "Select a model and prepared dataset, and configure 4-bit LoRA hyperparameters. "
            "Each export archives the exact dataset files and training records with the model.",
        )

        grid = QGridLayout()
        grid.setSpacing(10)

        # Model selection
        grid.addWidget(QLabel("Base Model:", cfg_card), 0, 0)
        self.model_combo = QComboBox(cfg_card)
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText(
            "Select local model or enter HF repo (e.g. unsloth/Qwen2.5-0.5B-Instruct)"
        )
        grid.addWidget(self.model_combo, 0, 1, 1, 3)

        # Dataset selection
        grid.addWidget(QLabel("Training Dataset:", cfg_card), 1, 0)
        self.dataset_combo = QComboBox(cfg_card)
        self.dataset_combo.setEditable(True)
        self.dataset_combo.setPlaceholderText("Select prepared dataset from datasets folder")
        grid.addWidget(self.dataset_combo, 1, 1, 1, 3)

        # Hyperparameters
        grid.addWidget(QLabel("Max Steps (0 = epochs):", cfg_card), 2, 0)
        self.steps_spin = QSpinBox(cfg_card)
        self.steps_spin.setRange(0, 100000)
        self.steps_spin.setValue(5)
        self.steps_spin.setToolTip(
            "Set to 0 to train for the configured number of epochs instead of a fixed step count."
        )
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

        # Epoch-based training is selected by setting Max Steps to zero.
        grid.addWidget(QLabel("Epochs:", cfg_card), 5, 0)
        self.epochs_spin = QSpinBox(cfg_card)
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(3)
        grid.addWidget(self.epochs_spin, 5, 1)

        grid.addWidget(QLabel("Warmup Steps:", cfg_card), 5, 2)
        self.warmup_spin = QSpinBox(cfg_card)
        self.warmup_spin.setRange(0, 100000)
        self.warmup_spin.setValue(0)
        grid.addWidget(self.warmup_spin, 5, 3)

        grid.addWidget(QLabel("Optimizer:", cfg_card), 6, 0)
        self.optimizer_combo = QComboBox(cfg_card)
        self.optimizer_combo.addItem("AdamW 8-bit (low VRAM)", "adamw_8bit")
        self.optimizer_combo.addItem("AdamW (PyTorch)", "adamw_torch")
        self.optimizer_combo.addItem("Paged AdamW 8-bit", "paged_adamw_8bit")
        self.optimizer_combo.setToolTip(
            "8-bit optimizers reduce optimizer memory. Availability depends on the installed "
            "bitsandbytes/Unsloth environment."
        )
        grid.addWidget(self.optimizer_combo, 6, 1, 1, 3)

        grid.addWidget(QLabel("Save Strategy:", cfg_card), 7, 0)
        self.save_strategy_combo = QComboBox(cfg_card)
        self.save_strategy_combo.addItem("Disabled", "no")
        self.save_strategy_combo.addItem("Every N steps", "steps")
        self.save_strategy_combo.addItem("Every epoch", "epoch")
        grid.addWidget(self.save_strategy_combo, 7, 1)

        grid.addWidget(QLabel("Save Interval:", cfg_card), 7, 2)
        self.save_steps_spin = QSpinBox(cfg_card)
        self.save_steps_spin.setRange(1, 100000)
        self.save_steps_spin.setValue(50)
        grid.addWidget(self.save_steps_spin, 7, 3)

        grid.addWidget(QLabel("Evaluation:", cfg_card), 8, 0)
        self.eval_strategy_combo = QComboBox(cfg_card)
        self.eval_strategy_combo.addItem("Disabled", "no")
        self.eval_strategy_combo.addItem("Every N steps", "steps")
        self.eval_strategy_combo.addItem("Every epoch", "epoch")
        grid.addWidget(self.eval_strategy_combo, 8, 1)

        grid.addWidget(QLabel("Eval Interval:", cfg_card), 8, 2)
        self.eval_steps_spin = QSpinBox(cfg_card)
        self.eval_steps_spin.setRange(1, 100000)
        self.eval_steps_spin.setValue(50)
        grid.addWidget(self.eval_steps_spin, 8, 3)

        grid.addWidget(QLabel("Evaluation Dataset:", cfg_card), 9, 0)
        self.eval_dataset_combo = QComboBox(cfg_card)
        self.eval_dataset_combo.setEditable(True)
        self.eval_dataset_combo.setPlaceholderText("Optional; required when evaluation is enabled")
        self.eval_dataset_combo.addItem("No evaluation dataset", "")
        grid.addWidget(self.eval_dataset_combo, 9, 1, 1, 3)

        # Export strategy. A merged fp16 model can be picked as the base model
        # for a later continued-training job; adapter-only stays small.
        grid.addWidget(QLabel("Export:", cfg_card), 10, 0)
        self.export_combo = QComboBox(cfg_card)
        self.export_combo.addItem("Merged model (16-bit) — reusable base", "merged_16bit")
        self.export_combo.addItem("Adapter only (LoRA)", "adapter")
        self.export_combo.setToolTip(
            "Merged saves a full 16-bit model with the training baked in, so it can be "
            "selected as the base model for continued training. It needs several GB of "
            "disk space and RAM. Adapter only saves the small LoRA modules."
        )
        grid.addWidget(self.export_combo, 10, 1, 1, 3)

        grid.addWidget(QLabel("Merged model name:", cfg_card), 11, 0)
        self.output_model_name_edit = QLineEdit(cfg_card)
        self.output_model_name_edit.setPlaceholderText("Base model name + -finetuned")
        self.output_model_name_edit.setToolTip(
            "Name of the merged model's folder in the Models library. Leave blank to "
            "use the source model name with '-finetuned' appended. The source model "
            "is never modified."
        )
        grid.addWidget(self.output_model_name_edit, 11, 1, 1, 3)
        self.export_combo.currentIndexChanged.connect(self._update_export_name_state)

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
        self.eval_loss_label = QLabel("Eval loss: —", control)
        self.eval_loss_label.setObjectName("StripValue")
        metrics_layout.addWidget(self.step_label)
        metrics_layout.addWidget(self.loss_label)
        metrics_layout.addWidget(self.eval_loss_label)
        metrics_layout.addStretch(1)
        control.add_layout(metrics_layout)

        self.dashboard = TrainingDashboard(control)
        control.add(self.dashboard)
        self._state.hardware_changed.connect(self.dashboard.set_hardware)
        if self._state.hardware is not None:
            self.dashboard.set_hardware(self._state.hardware)

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

        log_label = QLabel("Training Console", control)
        log_label.setObjectName("SectionLabel")
        control.add(log_label)
        self.console = ConsoleWidget(parent=control)
        control.add(self.console)
        self.log_view = self.console.text_edit

        self.add(control)

        # --- Outputs Card ---
        outputs = Card(
            "Training outputs",
            "Adapters and logs are written to a job-specific folder that other workflows can reuse.",
        )
        self.outputs_path = StoragePathRow("outputs_dir", "Outputs folder", outputs)
        self.outputs_path.open_requested.connect(self._open_outputs)
        outputs.add(self.outputs_path)
        self.last_adapter_row = FieldRow("Last export", "None saved yet", caption_width=104)
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

    def _update_export_name_state(self, *_args) -> None:
        """Only ask for a custom output name when exporting a merged model."""
        merged = self.export_combo.currentData() == "merged_16bit"
        self.output_model_name_edit.setEnabled(merged)
        self.output_model_name_edit.setToolTip(
            "Name of the merged model's folder in the Models library. Leave blank to "
            "use the source model name with '-finetuned' appended. The source model "
            "is never modified."
            if merged
            else "Adapter-only exports keep their job folder name."
        )

    def _validate_training_output_root(self, model_name: str) -> str | None:
        """Prevent a configured output folder from being nested inside the source model."""
        source = Path(str(model_name))
        if not source.is_dir():
            return None
        try:
            source_path = source.resolve()
            output_root = self._storage.path_for("outputs_dir").resolve()
        except OSError as error:
            return (
                f"Cannot verify that the training output is separate from the source model: {error}"
            )
        if output_root == source_path or source_path in output_root.parents:
            return (
                "The training outputs folder is inside the selected source model. "
                "Choose a separate outputs folder so the original model remains untouched."
            )
        return None

    @staticmethod
    def _resolve_output_model_name(model_name: str, requested_name: str) -> str:
        """Return a safe single Windows folder name for a merged model export."""
        source_name = str(model_name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        name = requested_name.strip() or f"{source_name or 'model'}-finetuned"
        invalid = '<>:"/\\|?*'
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if (
            name in {".", ".."}
            or len(name) > 120
            or any(char in invalid or ord(char) < 32 for char in name)
            or name.endswith((".", " "))
            or name.split(".", 1)[0].upper() in reserved
        ):
            raise ValueError(
                "Merged model name must be a valid Windows folder name (1–120 characters, "
                "without path separators or reserved characters)."
            )
        return name

    def _update_model_choices(self, models: list[LocalModelRecord]) -> None:
        curr = self.model_combo.currentText()
        self.model_combo.clear()
        for m in models:
            if m.kind in ("base", "abliterated"):
                self.model_combo.addItem(f"{m.name} ({m.format.upper()})", str(m.path))
        if curr:
            self.select_model(curr)
        elif self.model_combo.count() == 0:
            self.model_combo.addItem(
                "unsloth/Qwen2.5-0.5B-Instruct", "unsloth/Qwen2.5-0.5B-Instruct"
            )

    def _update_dataset_choices(self, datasets: list[LocalDatasetRecord]) -> None:
        curr = self.dataset_combo.currentText()
        eval_curr = self.eval_dataset_combo.currentText()
        self.dataset_combo.clear()
        self.eval_dataset_combo.clear()
        self.eval_dataset_combo.addItem("No evaluation dataset", "")
        for d in datasets:
            self.dataset_combo.addItem(d.name, str(d.path))
            self.eval_dataset_combo.addItem(d.name, str(d.path))
        if curr:
            self.select_dataset(curr)
        if eval_curr:
            idx = self.eval_dataset_combo.findText(eval_curr)
            if idx >= 0:
                self.eval_dataset_combo.setCurrentIndex(idx)
            else:
                self.eval_dataset_combo.setEditText(eval_curr)

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
            self.start_button.setEnabled(False)
            self.start_button.setToolTip("Waiting for the Unsloth engine check.")
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

    def _compatibility_advisory(self, model_name: str) -> str | None:
        """Return a non-blocking advisory when model/GPU compatibility is unknown.

        Unsloth is the only component that can decide compatibility, and it does
        so when the model is loaded. This helper never claims a verdict: it only
        flags missing evidence (no GPU snapshot, or a model that is not a known
        local model) so the user is warned before the job starts. An unknown
        value is advisory text, never a reason to block a job.
        """
        reasons: list[str] = []

        hardware = getattr(self._state, "hardware", None)
        gpu = getattr(hardware, "gpu", None) if hardware is not None else None
        if hardware is None or gpu is None or not getattr(hardware, "gpu_available", False):
            reasons.append("GPU not detected")
        elif not getattr(gpu, "name", None):
            reasons.append("GPU name unavailable")

        model_verified = False
        models = getattr(self._state, "models", None)
        if isinstance(models, (list, tuple)):
            candidate = str(model_name)
            for record in models:
                if candidate in (
                    str(getattr(record, "path", "")),
                    str(getattr(record, "name", "")),
                ):
                    architecture = str(getattr(record, "architecture", "") or "")
                    model_verified = bool(architecture) and architecture.lower() != "unknown"
                    break
        if not model_verified:
            reasons.append("model not verified in local library")

        if not reasons:
            return None
        detail = "; ".join(reasons)
        return (
            "[compatibility advisory] Compatibility not verified for this model/GPU "
            f"({detail}); Unsloth will report incompatibilities at load time."
        )

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

        eval_strategy = self.eval_strategy_combo.currentData() or "no"
        eval_dataset = (
            self.eval_dataset_combo.currentData() or self.eval_dataset_combo.currentText().strip()
        )
        if eval_strategy != "no":
            if not eval_dataset or not Path(str(eval_dataset)).is_file():
                self._state.notify(
                    "error",
                    "Choose an existing local evaluation dataset when evaluation is enabled.",
                )
                return
            if Path(str(eval_dataset)).resolve() == Path(str(dataset_path)).resolve():
                self._state.notify(
                    "error",
                    "Training and evaluation datasets must be different files.",
                )
                return

        try:
            learning_rate = float(self.lr_edit.text().strip())
        except ValueError:
            self._state.notify("error", "Learning rate must be a valid positive number.")
            return
        if not 0 < learning_rate < 1:
            self._state.notify("error", "Learning rate must be greater than 0 and less than 1.")
            return
        output_safety_error = self._validate_training_output_root(str(model_name))
        if output_safety_error:
            self._state.notify("error", output_safety_error)
            return
        if self.export_combo.currentData() == "merged_16bit":
            try:
                self._resolve_output_model_name(str(model_name), self.output_model_name_edit.text())
            except ValueError as error:
                self._state.notify("error", str(error))
                return

        unsloth_status = self._registry.status(UNSLOTH)
        interpreter = unsloth_status.interpreter
        if not interpreter or not interpreter.is_file():
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

        config_data = self._build_config(job.job_id, model_name, dataset_path)
        if job.state is JobState.QUEUED:
            self._queued_jobs[job.job_id] = {
                "config": config_data,
                "interpreter": interpreter,
                "script": worker_script,
            }
            self._connect_queue_signal()
            self.console.append(f"[queue] Job {job.job_id} queued; it starts when the GPU is free.")
            self._state.notify("info", f"Training job {job.job_id} queued.")
            return

        self._launch_job(job.job_id, config_data, interpreter, worker_script)

    def _build_config(self, job_id: str, model_name: str, dataset_path: str) -> dict:
        job_dir = self._storage.path_for("outputs_dir") / f"unsloth_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        export_strategy = self.export_combo.currentData() or "merged_16bit"
        if export_strategy == "merged_16bit":
            output_model_name = self._resolve_output_model_name(
                str(model_name), self.output_model_name_edit.text()
            )
            outputs_dir = job_dir / output_model_name
        else:
            output_model_name = "adapter"
            outputs_dir = job_dir
        try:
            lr_val = float(self.lr_edit.text().strip())
        except ValueError:
            lr_val = 2e-4
        eval_data = self.eval_dataset_combo.currentData()
        if eval_data == "":
            eval_dataset_path = None
        elif eval_data is None:
            eval_dataset_path = self.eval_dataset_combo.currentText().strip() or None
        else:
            eval_dataset_path = str(eval_data)
        return {
            "job_id": job_id,
            "model_name": model_name,
            "dataset_path": dataset_path,
            "output_dir": str(outputs_dir),
            "job_output_dir": str(job_dir),
            "output_model_name": output_model_name,
            "max_steps": self.steps_spin.value(),
            "max_seq_length": self.seq_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": lr_val,
            "lora_r": self.r_spin.value(),
            "lora_alpha": self.alpha_spin.value(),
            "num_train_epochs": self.epochs_spin.value(),
            "warmup_steps": self.warmup_spin.value(),
            "optim": self.optimizer_combo.currentData() or "adamw_8bit",
            "save_strategy": self.save_strategy_combo.currentData() or "no",
            "save_steps": self.save_steps_spin.value(),
            "eval_strategy": self.eval_strategy_combo.currentData() or "no",
            "eval_steps": self.eval_steps_spin.value(),
            "eval_dataset_path": eval_dataset_path,
            "export_strategy": export_strategy,
        }

    def _connect_queue_signal(self) -> None:
        if self._queue_connected:
            return
        signal = getattr(self._job_manager, "job_start_requested", None)
        if signal is not None and hasattr(signal, "connect"):
            signal.connect(self._on_job_start_requested)
            self._queue_connected = True

    def _on_job_start_requested(self, snapshot: JobSnapshot) -> None:
        if snapshot.state is not JobState.STARTING:
            return
        spec = self._queued_jobs.pop(snapshot.job_id, None)
        if spec is None:
            return
        self.console.append(f"[queue] Job {snapshot.job_id} is starting now.")
        self._launch_job(snapshot.job_id, spec["config"], spec["interpreter"], spec["script"])

    def _launch_job(self, job_id: str, config_data: dict, interpreter, worker_script) -> None:
        try:
            self._active_job_id = job_id
            self._pending_completion = None
            self.start_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.console.start_session("Unsloth training", str(config_data.get("model_name") or ""))
            advisory = self._compatibility_advisory(str(config_data["model_name"]))
            if advisory:
                self.console.append(advisory)
                self.start_button.setToolTip(advisory)
            self.step_label.setText("Step: 0")
            self.loss_label.setText("Loss: —")
            self.eval_loss_label.setText("Eval loss: —")
            self.dashboard.reset()
            self.dashboard.begin_run()

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

            worker.log_line.connect(self._on_worker_log_line)
            worker.protocol_error.connect(self._on_worker_protocol_error)
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
                self._handle_setup_failure(job_id, error)

    def _handle_setup_failure(self, job_id: str, error: Exception) -> None:
        """Fail a job whose worker could not be prepared or started."""
        self.dashboard.finish_run()
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
                self._state.notify(
                    "warning",
                    "The cancellation request could not be sent; waiting for worker exit.",
                )

    @property
    def worker_running(self) -> bool:
        """Whether the training worker process is still active."""
        return self._active_worker is not None and self._active_worker.running

    def request_shutdown(self) -> None:
        """Request graceful cancellation before the application closes."""
        if self.worker_running and self._active_job_id:
            self._cancel_training()

    def _on_worker_started(self) -> None:
        if self._active_job_id and self._active_worker:
            self._job_manager.mark_running(self._active_job_id, pid=self._active_worker.pid)

    def _on_worker_message(self, msg: WorkerMessage) -> None:
        payload = msg.payload

        if msg.type is MessageType.HELLO:
            pid = payload.get("pid")
            detail = f" (PID {pid})" if pid else ""
            self.console.append(f"[worker] Started{detail}.")

        elif msg.type is MessageType.READY:
            engine = payload.get("engine", "training engine")
            version = payload.get("version")
            detail = f" {version}" if version else ""
            self.console.append(f"[worker] {engine} ready{detail}.")

        elif msg.type is MessageType.ERROR:
            message = str(payload.get("message") or "Unknown worker error.")
            self.console.append(f"[engine error] {message}")

        elif msg.type is MessageType.PROGRESS:
            frac = float(payload.get("fraction", 0.0))
            self.progress_bar.setValue(int(frac * 100))
            self.dashboard.set_progress(frac)
            step = payload.get("step")
            total = payload.get("total")
            if step is not None:
                self.step_label.setText(f"Step: {step}/{total}")
            epoch = payload.get("epoch")
            if isinstance(epoch, (int, float)):
                self.dashboard.add_metric("epoch", int(step or 0), float(epoch))

        elif msg.type is MessageType.METRIC:
            name = payload.get("name")
            val = payload.get("value")
            if val is not None:
                value = float(val)
                step = int(payload.get("step", 0) or 0)
                self.dashboard.add_metric(str(name), step, value)
                if name == "loss":
                    self.loss_label.setText(f"Loss: {value:.4f}")
                elif name == "eval_loss":
                    self.eval_loss_label.setText(f"Eval loss: {value:.4f}")

        elif msg.type is MessageType.LOG:
            line = payload.get("message", "")
            self.console.append(str(line))

        elif msg.type is MessageType.RESULT:
            if payload.get("kind") == "architecture":
                metadata = payload.get("architecture")
                if isinstance(metadata, dict):
                    self.dashboard.set_architecture(metadata)
            else:
                out_dir = payload.get("output_dir", "")
                if out_dir:
                    self.last_adapter_row.set_value(out_dir)

        elif msg.type is MessageType.JOB_COMPLETE:
            success = bool(payload.get("success", False))
            message = str(
                payload.get("message")
                or (
                    "Training completed successfully."
                    if success
                    else "Training finished with error."
                )
            )
            self._pending_completion = (success, message)

    def _on_worker_log_line(self, line: str) -> None:
        line = str(line).strip()
        if line:
            self.console.append(f"[engine stderr] {line}")

    def _on_worker_protocol_error(self, error: object) -> None:
        self.console.append(f"[protocol warning] {error}")

    def _on_worker_exited(self, exit_code: int) -> None:
        if self._active_job_id:
            job_id = self._active_job_id
            snapshot = self._job_manager.snapshot(job_id)
            completion = self._pending_completion
            if completion is not None and completion[0] and exit_code == 0:
                self.progress_bar.setValue(100)
                self.dashboard.set_progress(1.0)
            self.dashboard.finish_run()

            if snapshot is not None and snapshot.state.value == "cancelling":
                failed_job = self._job_manager.handle_worker_exit(job_id, exit_code)
                if failed_job is not None:
                    self.console.append(
                        f"[worker exit] Job {job_id} finished after cancellation "
                        f"(exit code {exit_code}): {failed_job.message or failed_job.state.value}"
                    )

            elif completion is not None and completion[0] and exit_code == 0:
                self._job_manager.mark_succeeded(job_id, message=completion[1])
                self.console.append(f"[worker exit] {completion[1]}")
                self._state.publish_models(self._model_service.scan_local_models())
                self._state.notify("info", "LoRA fine-tuning completed successfully!")

            elif completion is not None:
                failed_job = self._job_manager.mark_failed(
                    job_id,
                    message=completion[1],
                    exit_code=exit_code,
                )
                self.console.append(f"[worker exit] {completion[1]} (exit code {exit_code})")

            else:
                failed_job = self._job_manager.handle_worker_exit(job_id, exit_code)
                if failed_job is not None and failed_job.state.value == "failed":
                    self.console.append(f"[worker exit] {failed_job.message}")
                elif failed_job is not None:
                    self.console.append(
                        f"[worker exit] Job finished with state "
                        f"{failed_job.state.value} (exit code {exit_code})."
                    )

            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None

        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_worker_failed_to_start(self, error: str) -> None:
        self.dashboard.finish_run()
        if self._active_job_id:
            self._job_manager.mark_failed(self._active_job_id, message=f"Failed to start: {error}")
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._state.notify("error", f"Unsloth worker failed to start: {error}")

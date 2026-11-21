"""Heretic tab: training-free model modification (abliteration) via the real engine."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from app.state import AppState
from services.engine_registry import HERETIC, EngineRegistry, EngineStatus
from services.model_service import LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, StoragePathRow
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from workers.job_manager import JobManager, JobSnapshot
from workers.protocol import MessageType, WorkerMessage
from workers.worker_process import WorkerProcess


class HereticTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        job_manager: JobManager | None = None,
        registry: EngineRegistry | None = None,
        model_service: ModelService | None = None,
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
        self._job_manager = job_manager or JobManager(parent=self)
        self._registry = registry or EngineRegistry(storage.settings, parent=self)
        self._model_service = model_service or ModelService(storage)

        self._active_worker: WorkerProcess | None = None
        self._active_job_id: str | None = None
        self._pending_completion: tuple[bool, str] | None = None

        # --- Engine Status Card ---
        self.engine_card = EngineCard(
            "Heretic engine",
            "Heretic runs in its own Python environment and uses a separate worker process. "
            "It never shares the GPU with Unsloth.",
            self,
        )
        self.add(self.engine_card)

        # --- Model and Setup Card ---
        setup = Card(
            "Model and abliteration setup",
            "Pick a model that fits 6 GB of VRAM, configure optimization trials, and choose export format.",
        )

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(QLabel("Target Model:", setup), 0, 0)
        self.model_combo = QComboBox(setup)
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText(
            "Select local model or enter HF repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
        )
        grid.addWidget(self.model_combo, 0, 1, 1, 3)

        grid.addWidget(QLabel("Optimization Trials:", setup), 1, 0)
        self.trials_spin = QSpinBox(setup)
        self.trials_spin.setRange(1, 100)
        self.trials_spin.setValue(2)
        grid.addWidget(self.trials_spin, 1, 1)

        grid.addWidget(QLabel("Export Strategy:", setup), 1, 2)
        self.export_combo = QComboBox(setup)
        self.export_combo.addItems(["merge", "adapter"])
        grid.addWidget(self.export_combo, 1, 3)

        setup.add_layout(grid)

        self.outputs_path = StoragePathRow("outputs_dir", "Output folder", setup)
        self.outputs_path.open_requested.connect(self._open_outputs)
        setup.add(self.outputs_path)

        self.add(setup)

        # --- Control & Live Progress Card ---
        control = Card(
            "Abliteration control & live monitoring",
            "Heretic optimizes weight adjustments using Optuna to minimize refusal and KL divergence.",
        )
        self.run_state_row = FieldRow("Job state", "Idle", caption_width=104)
        control.add(self.run_state_row)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)
        self.trial_metric_label = QLabel("Trial: —", control)
        self.trial_metric_label.setObjectName("StripValue")
        self.refusal_metric_label = QLabel("Refusal rate: —", control)
        self.refusal_metric_label.setObjectName("StripValue")
        self.kl_metric_label = QLabel("KL divergence: —", control)
        self.kl_metric_label.setObjectName("StripValue")
        metrics_layout.addWidget(self.trial_metric_label)
        metrics_layout.addWidget(self.refusal_metric_label)
        metrics_layout.addWidget(self.kl_metric_label)
        metrics_layout.addStretch(1)
        control.add_layout(metrics_layout)

        self.progress_bar = QProgressBar(control)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control.add(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.run_button = QPushButton("Run Heretic", control)
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._start_heretic)
        btn_row.addWidget(self.run_button)

        self.cancel_button = QPushButton("Cancel run", control)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_heretic)
        btn_row.addWidget(self.cancel_button)
        btn_row.addStretch(1)
        control.add_layout(btn_row)

        log_label = QLabel("Live Engine Log", control)
        log_label.setObjectName("SectionLabel")
        control.add(log_label)

        self.log_view = QTextEdit(control)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setStyleSheet(
            "font-family: 'Cascadia Mono', Consolas, monospace; font-size: 8.5pt;"
        )
        control.add(self.log_view)

        self.last_model_row = FieldRow("Saved model", "None saved yet", caption_width=104)
        control.add(self.last_model_row)

        self.add(control)

        # --- Information Card ---
        notes = Card(
            "What to expect",
            "Heretic benchmarks the machine, optimizes against refusal and divergence scores, "
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
        self._state.models_changed.connect(self._update_model_choices)

    def _open_outputs(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def select_model(self, model_ref: str) -> None:
        idx = self.model_combo.findText(model_ref)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(model_ref)

    def _update_model_choices(self, models: list[LocalModelRecord]) -> None:
        curr = self.model_combo.currentText()
        self.model_combo.clear()
        for m in models:
            if m.kind in ("base", "abliterated"):
                self.model_combo.addItem(f"{m.name} ({m.format.upper()})", str(m.path))
        if curr:
            self.select_model(curr)
        elif self.model_combo.count() == 0:
            self.model_combo.addItem("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")

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
        if status is None:
            self.set_status("Checking Heretic", "info")
            self.run_button.setEnabled(False)
            self.run_button.setToolTip("Waiting for the Heretic engine check.")
            return

        self.engine_card.apply_status(status)
        is_avail = status.is_available
        has_no_active = self._active_worker is None or not self._active_worker.running
        self.run_button.setEnabled(is_avail and has_no_active)
        if is_avail:
            self.set_status("Heretic engine available", "ok")
            # Compatibility cannot be proven from metadata alone. Never block on
            # an unknown, but make the limitation visible before the user starts.
            if self._state.hardware is None:
                self.run_button.setToolTip(
                    "Run Heretic abliteration — GPU information unavailable; "
                    "model compatibility is not verified."
                )
            else:
                self.run_button.setToolTip(
                    "Run Heretic abliteration — verify the selected model fits in "
                    "VRAM; compatibility is not certified automatically."
                )
        else:
            level = "error" if status.badge() == "check failed" else "warn"
            self.set_status(f"Heretic {status.badge()}", level)
            self.run_button.setToolTip(f"Heretic engine unavailable: {status.detail}")

    def apply_job(self, job: JobSnapshot | None) -> None:
        if job is None:
            self.run_state_row.set_value("Idle")
            return
        self.run_state_row.set_value(f"{job.state.value} · {job.engine}")
        if job.state.is_terminal:
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    def _start_heretic(self) -> None:
        model_name = self.model_combo.currentData() or self.model_combo.currentText().strip()
        if not model_name:
            self._state.notify("warning", "Please select or enter a model name to abliterate.")
            return

        heretic_status = self._registry.status(HERETIC)
        interpreter = heretic_status.interpreter
        if not interpreter or not interpreter.is_file():
            fallback = Path("environments/heretic/Scripts/python.exe").resolve()
            if fallback.is_file():
                interpreter = fallback
            else:
                self._state.notify("error", "Heretic interpreter not found. Set it in Settings.")
                return

        worker_script = Path(__file__).resolve().parents[1] / "workers" / "heretic_worker.py"
        if not worker_script.is_file():
            self._state.notify("error", f"Worker script not found: {worker_script}")
            return

        try:
            job = self._job_manager.submit(HERETIC, model_ref=model_name)
        except Exception as error:
            self._state.notify("error", f"Cannot start Heretic: {error}")
            return

        try:
            self._active_job_id = job.job_id
            self._pending_completion = None
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.log_view.clear()
            self.trial_metric_label.setText("Trial: 0")
            self.refusal_metric_label.setText("Refusal: —")
            self.kl_metric_label.setText("KL Div: —")

            outputs_dir = self._storage.path_for("outputs_dir") / f"heretic_{job.job_id}"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            config_data = {
                "job_id": job.job_id,
                "model_name": model_name,
                "trials": self.trials_spin.value(),
                "export_strategy": self.export_combo.currentText(),
                "output_dir": str(outputs_dir / "model"),
                "seed": 42,
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
                    f"Heretic worker started but setup reported an error: {error}",
                )
            else:
                self._handle_setup_failure(job.job_id, error)

    def _handle_setup_failure(self, job_id: str, error: Exception) -> None:
        """Fail a job whose worker could not be prepared or started."""
        self._job_manager.mark_failed(
            job_id,
            message=f"Could not prepare Heretic job: {error}",
        )
        self._active_job_id = None
        self._active_worker = None
        self._pending_completion = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._state.notify("error", f"Could not prepare Heretic job: {error}")

    def _cancel_heretic(self) -> None:
        if self._active_job_id and self._active_worker and self._active_worker.running:
            self._job_manager.request_cancel(self._active_job_id)
            sent = self._active_worker.request_cancel(self._active_job_id)
            self.cancel_button.setEnabled(False)
            if sent:
                self._state.notify(
                    "info",
                    "Cancellation requested. Heretic has up to 10 seconds to stop cleanly before it is terminated.",
                )
            else:
                self._state.notify(
                    "warning",
                    "The cancellation request could not be sent; waiting for worker exit.",
                )

    @property
    def worker_running(self) -> bool:
        """Whether the Heretic worker process is still active."""
        return self._active_worker is not None and self._active_worker.running

    def request_shutdown(self) -> None:
        """Request graceful cancellation before the application closes."""
        if self.worker_running and self._active_job_id:
            self._cancel_heretic()

    def _on_worker_started(self) -> None:
        if self._active_job_id and self._active_worker:
            self._job_manager.mark_running(self._active_job_id, pid=self._active_worker.pid)

    def _on_worker_log_line(self, line: str) -> None:
        line = str(line).strip()
        if line:
            self.log_view.append(f"[engine stderr] {line}")

    def _on_worker_protocol_error(self, error: object) -> None:
        self.log_view.append(f"[protocol warning] {error}")

    def _on_worker_message(self, msg: WorkerMessage) -> None:
        payload = msg.payload
        if msg.type is MessageType.HELLO:
            pid = payload.get("pid")
            detail = f" (PID {pid})" if pid else ""
            self.log_view.append(f"[worker] Started{detail}.")

        elif msg.type is MessageType.READY:
            engine = payload.get("engine", "Heretic engine")
            version = payload.get("version")
            detail = f" {version}" if version else ""
            self.log_view.append(f"[worker] {engine} ready{detail}.")

        elif msg.type is MessageType.ERROR:
            message = str(payload.get("message") or "Unknown worker error.")
            self.log_view.append(f"[engine error] {message}")

        elif msg.type is MessageType.PROGRESS:
            frac = float(payload.get("fraction", 0.0))
            self.progress_bar.setValue(int(frac * 100))
            step = payload.get("step")
            total = payload.get("total")
            if step is not None:
                self.trial_metric_label.setText(f"Trial: {step}/{total}")

        elif msg.type is MessageType.METRIC:
            name = payload.get("name")
            val = payload.get("value")
            if name == "refusal_rate" and val is not None:
                self.refusal_metric_label.setText(f"Refusal: {float(val) * 100:.1f}%")
            elif name == "kl_divergence" and val is not None:
                self.kl_metric_label.setText(f"KL Div: {float(val):.4f}")

        elif msg.type is MessageType.LOG:
            line = payload.get("message", "")
            self.log_view.append(line)

        elif msg.type is MessageType.RESULT:
            out_dir = payload.get("output_dir", "")
            self.last_model_row.set_value(out_dir)

        elif msg.type is MessageType.JOB_COMPLETE:
            success = bool(payload.get("success", False))
            message = str(
                payload.get("message")
                or (
                    "Heretic abliteration completed." if success else "Heretic finished with error."
                )
            )
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
                self._state.publish_models(self._model_service.scan_local_models())
                self._state.notify("info", "Heretic abliteration completed successfully!")
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
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_worker_failed_to_start(self, error: str) -> None:
        if self._active_job_id:
            self._job_manager.mark_failed(self._active_job_id, message=f"Failed to start: {error}")
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._state.notify("error", f"Heretic worker failed to start: {error}")

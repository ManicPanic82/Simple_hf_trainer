"""Convert a merged Hugging Face model to GGUF and register it with Ollama."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from ui.components import Card, FieldRow
from ui.tab_page import TabPage


class OllamaExportTab(TabPage):
    """Non-blocking llama.cpp conversion, quantisation and Ollama import workflow."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Ollama Export",
            "Convert a merged Hugging Face model to GGUF, quantise it, then import and verify "
            "the resulting model in your local Ollama installation.",
            parent,
        )
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._stages: list[tuple[str, str, list[str]]] = []
        self._stage_index = 0
        self._cancel_requested = False
        self._final_gguf: Path | None = None
        self._modelfile: Path | None = None

        card = Card(
            "Conversion settings",
            "The source model is read-only. Intermediate and final GGUF files are written "
            "to the selected output directory.",
        )
        grid = QGridLayout()
        grid.setSpacing(9)

        self.source_edit = QLineEdit(card)
        self.source_edit.setPlaceholderText("Folder containing config.json and model weights")
        self.source_button = QPushButton("Browse…", card)
        self.source_button.clicked.connect(self._browse_source)
        self._add_path_row(grid, 0, "Hugging Face model", self.source_edit, self.source_button)

        self.python_edit = QLineEdit(card)
        self.python_edit.setPlaceholderText("Python interpreter with llama.cpp conversion dependencies")
        self.python_button = QPushButton("Browse…", card)
        self.python_button.clicked.connect(lambda: self._browse_file(self.python_edit, "Python executable (*.exe)"))
        self._add_path_row(grid, 1, "Conversion Python", self.python_edit, self.python_button)

        self.converter_edit = QLineEdit(card)
        self.converter_edit.setPlaceholderText("llama.cpp\convert_hf_to_gguf.py")
        self.converter_button = QPushButton("Browse…", card)
        self.converter_button.clicked.connect(lambda: self._browse_file(self.converter_edit, "Python scripts (*.py)"))
        self._add_path_row(grid, 2, "llama.cpp converter", self.converter_edit, self.converter_button)

        self.quantizer_edit = QLineEdit(card)
        self.quantizer_edit.setPlaceholderText("llama-quantize.exe")
        self.quantizer_button = QPushButton("Browse…", card)
        self.quantizer_button.clicked.connect(lambda: self._browse_file(self.quantizer_edit, "Executable (*.exe)"))
        self._add_path_row(grid, 3, "Quantizer executable", self.quantizer_edit, self.quantizer_button)

        self.ollama_edit = QLineEdit(card)
        self.ollama_edit.setPlaceholderText("ollama.exe")
        self.ollama_button = QPushButton("Browse…", card)
        self.ollama_button.clicked.connect(lambda: self._browse_file(self.ollama_edit, "Executable (*.exe)"))
        self._add_path_row(grid, 4, "Ollama executable", self.ollama_edit, self.ollama_button)

        self.output_edit = QLineEdit(card)
        self.output_edit.setPlaceholderText("Choose a destination for GGUF files and Modelfile")
        self.output_button = QPushButton("Browse…", card)
        self.output_button.clicked.connect(self._browse_output)
        self._add_path_row(grid, 5, "Output directory", self.output_edit, self.output_button)

        self.name_edit = QLineEdit("my-finetuned-model", card)
        grid.addWidget(QLabel("Ollama model name:"), 6, 0)
        grid.addWidget(self.name_edit, 6, 1, 1, 2)

        self.quant_combo = QComboBox(card)
        self.quant_combo.addItem("Q4_K_M · recommended balance", "Q4_K_M")
        self.quant_combo.addItem("Q5_K_M · higher quality / larger", "Q5_K_M")
        self.quant_combo.addItem("Q8_0 · near-full precision / largest", "Q8_0")
        self.quant_combo.addItem("F16 · no quantisation", "F16")
        grid.addWidget(QLabel("Quantisation:"), 7, 0)
        grid.addWidget(self.quant_combo, 7, 1, 1, 2)

        card.add_layout(grid)
        self.add(card)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Convert and import to Ollama", self)
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start_export)
        action_row.addWidget(self.start_button)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_export)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch(1)
        self.add_layout(action_row)

        self.status_row = FieldRow("Conversion status", "Ready", caption_width=130)
        self.add(self.status_row)
        self.log = QLabel("Select a merged Hugging Face model to begin.")
        self.log.setObjectName("Console")
        self.log.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log.setWordWrap(True)
        self.log.setMinimumHeight(150)
        self.add(self.log)
        self.set_status("Ollama export ready")

    @staticmethod
    def _add_path_row(grid, row, label, edit, button) -> None:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(edit, row, 1)
        grid.addWidget(button, row, 2)

    @property
    def worker_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select merged Hugging Face model")
        if path:
            self.source_edit.setText(path)
            if not self.name_edit.text().strip() or self.name_edit.text() == "my-finetuned-model":
                self.name_edit.setText(Path(path).name.lower().replace("_", "-"))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select GGUF output directory")
        if path:
            self.output_edit.setText(path)

    def _browse_file(self, edit: QLineEdit, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select executable or script", "", file_filter)
        if path:
            edit.setText(path)

    def _append(self, message: str) -> None:
        previous = self.log.text()
        self.log.setText((previous + "\n" + message).strip()[-12000:])

    def start_export(self) -> None:
        if self.worker_running:
            return
        source = Path(self.source_edit.text().strip()).expanduser()
        python = Path(self.python_edit.text().strip()).expanduser()
        converter = Path(self.converter_edit.text().strip()).expanduser()
        quantizer = Path(self.quantizer_edit.text().strip()).expanduser()
        ollama = Path(self.ollama_edit.text().strip()).expanduser()
        output = Path(self.output_edit.text().strip()).expanduser()
        name = self.name_edit.text().strip().lower()
        quant = str(self.quant_combo.currentData())

        if not source.is_dir() or not (source / "config.json").is_file():
            self._fail("Select a merged Hugging Face model folder containing config.json.")
            return
        if (source / "adapter_config.json").is_file():
            self._fail(
                "This is a LoRA adapter, not a merged model. Export a merged 16-bit model "
                "from Training first, then convert that model here."
            )
            return
        # Enforce source/output separation before checking optional tools such as
        # the quantizer. This is a filesystem safety rule, not a tool prerequisite.
        try:
            source_resolved = source.resolve()
            output_resolved = output.resolve()
            if output_resolved == source_resolved or source_resolved in output_resolved.parents:
                self._fail("Choose an output directory outside the source model folder.")
                return
        except OSError as error:
            self._fail(f"Could not resolve export paths: {error}")
            return
        if not python.is_file() or not converter.is_file() or not ollama.is_file():
            self._fail("Select valid Python, llama.cpp converter and Ollama executable paths.")
            return
        if quant != "F16" and not quantizer.is_file():
            self._fail("Select a valid llama-quantize executable, or choose F16.")
            return
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", name):
            self._fail("Ollama model name must be 2–64 lowercase letters, numbers, dots, underscores or hyphens.")
            return
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._fail(f"Could not create output directory: {error}")
            return

        f16_path = output / f"{name}.f16.gguf"
        final_path = output / (f"{name}.gguf" if quant == "F16" else f"{name}.{quant.lower()}.gguf")
        modelfile = output / f"{name}.Modelfile"
        self._final_gguf = final_path
        self._modelfile = modelfile
        protected_outputs = [final_path, modelfile]
        if quant != "F16":
            protected_outputs.append(f16_path)
        existing = [path for path in protected_outputs if path.exists()]
        if existing:
            self._fail(
                "Refusing to overwrite existing export files: "
                + ", ".join(str(path) for path in existing)
            )
            return

        self._stages = [
            ("Convert HF to GGUF (F16)", str(python),
             [str(converter), "--outfile", str(f16_path), "--outtype", "f16", str(source)]),
        ]
        if quant != "F16":
            self._stages.append(
                (f"Quantise to {quant}", str(quantizer), [str(f16_path), str(final_path), quant])
            )
        else:
            self._stages[0] = (
                "Convert HF to GGUF (F16)", str(python),
                [str(converter), "--outfile", str(final_path), "--outtype", "f16", str(source)],
            )
        try:
            modelfile.write_text(f'FROM "{final_path.as_posix()}"\n', encoding="utf-8")
        except OSError as error:
            self._fail(f"Could not write Ollama Modelfile: {error}")
            return
        self._stages.extend([
            ("Import into Ollama", str(ollama), ["create", name, "-f", str(modelfile)]),
            ("Verify Ollama model", str(ollama), ["show", name]),
        ])
        self._stage_index = 0
        self._cancel_requested = False
        self.log.setText("")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_row.set_value("Converting")
        self._append(f"Source: {source}")
        self._append(f"Output: {final_path}")
        self._start_next_stage()

    def _start_next_stage(self) -> None:
        if self._stage_index >= len(self._stages):
            self.status_row.set_value("Complete")
            self._append(f"Success. Ollama model is ready: {self.name_edit.text().strip()}")
            self._append(f"GGUF: {self._final_gguf}")
            self._append(f"Modelfile: {self._modelfile}")
            self._finish()
            return
        label, program, arguments = self._stages[self._stage_index]
        self.status_row.set_value(f"{self._stage_index + 1}/{len(self._stages)} · {label}")
        self._append(f"\n=== {label} ===")
        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.start()

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode(errors="replace").strip()
        if data:
            self._append(data)

    def _read_error(self) -> None:
        data = bytes(self.process.readAllStandardError()).decode(errors="replace").strip()
        if data:
            self._append(data)

    def _process_finished(self, exit_code: int, exit_status) -> None:
        self._read_output()
        self._read_error()
        if self._cancel_requested:
            self.status_row.set_value("Cancelled")
            self._append("Conversion cancelled. Partial output files were retained.")
            self._finish()
            return
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self._fail(f"Stage failed (exit code {exit_code}). Review the output above.")
            return
        self._stage_index += 1
        self._start_next_stage()

    def _process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._fail(f"Could not start process: {self.process.errorString()}")

    def cancel_export(self) -> None:
        if self.worker_running:
            self._cancel_requested = True
            self.process.terminate()
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
            self._append("Conversion cancelled. Partial output files were retained.")

    def request_shutdown(self) -> None:
        self.cancel_export()

    def _fail(self, message: str) -> None:
        self.status_row.set_value("Failed")
        self._append(message)
        self._finish()

    def _finish(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)


__all__ = ["OllamaExportTab"]

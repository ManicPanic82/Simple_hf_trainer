"""Settings dialog: storage folders, engine interpreters, and polling."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.credentials import CredentialStore
from services.settings_service import (
    MAX_POLL_INTERVAL_MS,
    MIN_POLL_INTERVAL_MS,
    SCHEMA_VERSION,
    AppSettings,
    EngineSettings,
    PathSettings,
    SettingsService,
    UiSettings,
)
from services.storage_service import Severity, StorageService
from ui.components import Pill


class _PathEditor(QWidget):
    """Line edit plus browse button for one path."""

    def __init__(
        self,
        value: str,
        caption: str,
        *,
        mode: str = "directory",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel(caption, self)
        label.setObjectName("Muted")
        label.setMinimumWidth(130)
        layout.addWidget(label)

        self.edit = QLineEdit(value, self)
        self.edit.setPlaceholderText("Absolute path")
        layout.addWidget(self.edit, 1)

        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)

    def _browse(self) -> None:
        if self._mode == "directory":
            selected = QFileDialog.getExistingDirectory(
                self, "Select folder", self.edit.text() or str(Path.home())
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select interpreter",
                self.edit.text() or str(Path.home()),
                "Python (python.exe)",
            )
        if selected:
            self.edit.setText(str(Path(selected)))

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: str) -> None:
        self.edit.setText(value)


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: AppSettings,
        settings_service: SettingsService,
        credentials: CredentialStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(760)
        self._settings_service = settings_service
        self._credentials = credentials
        self._result_settings: AppSettings | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        heading = QLabel("Settings", self)
        heading.setObjectName("TabTitle")
        layout.addWidget(heading)
        hint = QLabel(
            "Paths must be absolute. Missing application folders are created on save; "
            "existing data is never modified or deleted. Hugging Face tokens are stored in "
            "Windows Credential Manager, not in this file.",
            self,
        )
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        storage_box = QGridLayout()
        storage_box.setHorizontalSpacing(12)
        storage_box.setVerticalSpacing(8)
        layout.addLayout(storage_box)

        self.models_edit = _PathEditor(str(settings.paths.models_dir), "Models folder", parent=self)
        self.datasets_edit = _PathEditor(
            str(settings.paths.datasets_dir), "Datasets folder", parent=self
        )
        self.outputs_edit = _PathEditor(
            str(settings.paths.outputs_dir), "Outputs folder", parent=self
        )
        self.cache_edit = _PathEditor(
            str(settings.paths.hf_cache_dir), "HF cache folder", parent=self
        )
        for row, editor in enumerate(
            (self.models_edit, self.datasets_edit, self.outputs_edit, self.cache_edit)
        ):
            storage_box.addWidget(editor, row, 0)

        engines_label = QLabel("Engine environments (isolated interpreters)", self)
        engines_label.setObjectName("SectionLabel")
        layout.addWidget(engines_label)
        engines_hint = QLabel(
            "Leave blank to use environments\\unsloth\\Scripts\\python.exe and "
            "environments\\heretic\\Scripts\\python.exe inside this repository.",
            self,
        )
        engines_hint.setObjectName("Muted")
        engines_hint.setWordWrap(True)
        layout.addWidget(engines_hint)

        self.unsloth_edit = _PathEditor(
            settings.engines.unsloth_python, "Unsloth python.exe", mode="file", parent=self
        )
        self.heretic_edit = _PathEditor(
            settings.engines.heretic_python, "Heretic python.exe", mode="file", parent=self
        )
        layout.addWidget(self.unsloth_edit)
        layout.addWidget(self.heretic_edit)

        poll_row = QHBoxLayout()
        poll_row.setSpacing(8)
        poll_label = QLabel("Hardware poll interval", self)
        poll_label.setObjectName("Muted")
        poll_label.setMinimumWidth(130)
        poll_row.addWidget(poll_label)
        self.poll_spin = QSpinBox(self)
        self.poll_spin.setRange(MIN_POLL_INTERVAL_MS, MAX_POLL_INTERVAL_MS)
        self.poll_spin.setSingleStep(500)
        self.poll_spin.setSuffix(" ms")
        self.poll_spin.setValue(settings.ui.poll_interval_ms)
        poll_row.addWidget(self.poll_spin)
        poll_row.addStretch(1)
        layout.addLayout(poll_row)

        cred_label = QLabel("Hugging Face Authentication", self)
        cred_label.setObjectName("SectionLabel")
        layout.addWidget(cred_label)

        cred_row = QHBoxLayout()
        cred_row.setSpacing(8)
        hf_token_label = QLabel("User access token", self)
        hf_token_label.setObjectName("Muted")
        hf_token_label.setMinimumWidth(130)
        cred_row.addWidget(hf_token_label)

        self.token_edit = QLineEdit(self)
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("hf_... (stored securely in Windows Credential Manager)")
        cred_row.addWidget(self.token_edit, 1)

        self.show_token_btn = QPushButton("Show", self)
        self.show_token_btn.clicked.connect(self._toggle_token_visibility)
        cred_row.addWidget(self.show_token_btn)

        self.save_token_btn = QPushButton("Store Token", self)
        self.save_token_btn.clicked.connect(self._save_token)
        cred_row.addWidget(self.save_token_btn)

        self.clear_token_btn = QPushButton("Clear", self)
        self.clear_token_btn.clicked.connect(self._clear_token)
        cred_row.addWidget(self.clear_token_btn)

        self.token_pill = Pill("No token stored", "off", self)
        cred_row.addWidget(self.token_pill)
        layout.addLayout(cred_row)

        self._refresh_token_status()

        self.status_pill = Pill("", "off", self)
        self.status_pill.setVisible(False)
        layout.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignLeft)
        self.message = QLabel("", self)
        self.message.setWordWrap(True)
        self.message.setObjectName("Muted")
        self.message.setVisible(False)
        layout.addWidget(self.message)

        buttons = QDialogButtonBox(self)
        self.save_button = buttons.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        self.save_button.setObjectName("Primary")
        cancel = buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        cancel.clicked.connect(self.reject)
        self.save_button.clicked.connect(self._save)
        layout.addWidget(buttons)

    def _toggle_token_visibility(self) -> None:
        if self.token_edit.echoMode() == QLineEdit.EchoMode.Password:
            self.token_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_token_btn.setText("Hide")
        else:
            self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_token_btn.setText("Show")

    def _refresh_token_status(self) -> None:
        if not self._credentials or not self._credentials.available:
            self.token_pill.set_state("off", "Credential store unavailable")
            self.save_token_btn.setEnabled(False)
            self.clear_token_btn.setEnabled(False)
            return

        token = self._credentials.get_token()
        if token:
            self.token_pill.set_state("ok", "Token stored securely")
            if not self.token_edit.text():
                self.token_edit.setPlaceholderText("•••••••••••••••• (saved)")
        else:
            self.token_pill.set_state("off", "No token stored")

    def _save_token(self) -> None:
        if not self._credentials:
            return
        token = self.token_edit.text().strip()
        if token:
            self._credentials.set_token(token)
            self.token_edit.clear()
            self._refresh_token_status()
            self.status_pill.setVisible(True)
            self.status_pill.set_state("ok", "Token saved in Windows Credential Manager")

    def _clear_token(self) -> None:
        if not self._credentials:
            return
        self._credentials.delete_token()
        self.token_edit.clear()
        self._refresh_token_status()
        self.status_pill.setVisible(True)
        self.status_pill.set_state("info", "Token removed from Windows Credential Manager")

    # -- result ---------------------------------------------------------

    def result_settings(self) -> AppSettings | None:
        return self._result_settings

    def _collect(self) -> AppSettings:
        return AppSettings(
            schema_version=SCHEMA_VERSION,
            paths=PathSettings(
                models_dir=Path(self.models_edit.value()),
                datasets_dir=Path(self.datasets_edit.value()),
                outputs_dir=Path(self.outputs_edit.value()),
                hf_cache_dir=Path(self.cache_edit.value()),
            ),
            engines=EngineSettings(
                unsloth_python=self.unsloth_edit.value(),
                heretic_python=self.heretic_edit.value(),
            ),
            ui=UiSettings(poll_interval_ms=int(self.poll_spin.value())),
        )

    def _save(self) -> None:
        candidate = self._collect()
        errors: list[str] = []
        warnings: list[str] = []

        for label, path in candidate.paths.items():
            if not str(path).strip():
                errors.append(f"{label.replace('_dir', '')}: path is empty.")
            elif not path.is_absolute():
                errors.append(f"{label.replace('_dir', '')}: '{path}' is not an absolute path.")

        if errors:
            self._show_issues(errors, warnings)
            return

        validation = StorageService(candidate).validate_all()
        for result in validation.results:
            for issue in result.issues:
                text = f"{result.label}: {issue.message}"
                if issue.severity is Severity.ERROR:
                    errors.append(text)
                else:
                    warnings.append(text)

        for name, value in (
            ("Unsloth", candidate.engines.unsloth_python),
            ("Heretic", candidate.engines.heretic_python),
        ):
            if value and not Path(value).is_file():
                warnings.append(f"{name}: interpreter not found at {value}.")

        if errors:
            self._show_issues(errors, warnings)
            return

        try:
            self._settings_service.save(candidate)
        except OSError as error:
            self._show_issues([f"Could not save settings: {error}"], warnings)
            return

        created = StorageService(candidate).ensure_directories()
        for result in created.results:
            for issue in result.issues:
                if issue.severity is Severity.ERROR:
                    warnings.append(f"{result.label}: {issue.message}")

        self._result_settings = candidate
        self.accept()

    def _show_issues(self, errors: list[str], warnings: list[str]) -> None:
        self.status_pill.setVisible(True)
        self.message.setVisible(True)
        if errors:
            self.status_pill.set_state("error", "Fix the highlighted problems")
            self.message.setText("\n".join(errors + warnings))
        else:
            self.status_pill.set_state("warn", "Saved with warnings")
            self.message.setText("\n".join(warnings))

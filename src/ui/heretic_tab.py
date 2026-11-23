"""Heretic tab: a two-column model-modification workstation.

Left column: source model, modification profile, modifier, scoring, hardware
and execution, optimization, output/provenance, and raw TOML.
Right column: an always-visible run monitor and a permanent live console that
streams the real worker stdout/stderr.

The worker, job manager, GPU lock, cancellation and shutdown paths are the
existing ones.  This module only changes presentation and log handling.
"""

from __future__ import annotations

import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.state import AppState
from services.credentials import CredentialStore
from services.engine_registry import HERETIC, EngineRegistry, EngineStatus
from services.heretic_config import (
    CHECKPOINT_ACTIONS,
    EXPORT_STRATEGIES,
    MODEL_ACTIONS,
    MODIFIER_LABELS,
    OPTIMIZATIONS,
    PRESET_LABELS,
    QUANTIZATIONS,
    REPRODUCIBILITY,
    SCORER_KEYWORD,
    SCORER_KL,
    SCORER_LABELS,
    HereticConfig,
    HereticConfigError,
    PluginEntry,
    available_presets,
    load_preset,
)
from services.model_service import LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, CollapsibleSection, FieldRow, Pill, StoragePathRow
from ui.console import ConsoleWidget
from ui.engine_card import EngineCard
from ui.storage_action import open_storage_path
from ui.tab_page import TabPage
from workers.job_manager import JobManager, JobSnapshot, JobState
from workers.protocol import MessageType, WorkerMessage
from workers.worker_process import WorkerProcess

_CUSTOM = "Custom…"
_MODIFIER_BY_PATH = {path: label for label, path in MODIFIER_LABELS.items()}
_MODIFIER_BY_LABEL = dict(MODIFIER_LABELS.items())
_SCORER_BY_PATH = {path: label for label, path in SCORER_LABELS.items()}
_SCORER_METRIC = {SCORER_KEYWORD: "refusal_rate", SCORER_KL: "kl_divergence"}
# Presets documented on heretic-project.org/configuration (Heretic 2.1.0).
_UPSTREAM_PROFILES = frozenset({"default", "noslop", "nohumor", "nosyco", "emotions"})


def _bundled_profile_ids() -> frozenset[str]:
    """Profile ids shipped in this repository's ``config/heretic_presets``."""
    try:
        return frozenset(preset_id for preset_id, _label in available_presets()) - {"compact"}
    except OSError:
        return frozenset(PRESET_LABELS) - {"compact"}


_BUNDLED_PROFILES = _bundled_profile_ids()


def _dataset_text(spec: dict) -> str:
    return str(spec.get("dataset", "") or "")


def _dataset_split(spec: dict) -> str:
    return str(spec.get("split", "") or "")


def _dataset_column(spec: dict) -> str:
    return str(spec.get("column", "") or "")


def _major_version(version: str | None) -> int | None:
    if not version:
        return None
    match = re.match(r"^\s*(\d+)", str(version))
    return int(match.group(1)) if match else None


def _profile_origin(preset_id: str | None) -> str:
    if not preset_id:
        return "Unknown profile"
    if preset_id == "compact":
        return "Built-in low-VRAM profile (this application)"
    if preset_id in _UPSTREAM_PROFILES:
        return "Upstream Heretic preset"
    if preset_id in _BUNDLED_PROFILES:
        return "Bundled profile (this application)"
    return "User-defined profile"


def _form_grid() -> QGridLayout:
    grid = QGridLayout()
    grid.setSpacing(8)
    grid.setColumnStretch(1, 1)
    return grid


def _compact_combo(combo: QComboBox, chars: int = 16) -> QComboBox:
    """Keep long item labels from forcing the column wider than the pane."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return combo


def _add_form_row(
    grid: QGridLayout,
    row: int,
    label: str,
    widget: QWidget,
    *,
    span: int = 1,
) -> int:
    grid.addWidget(QLabel(label), row, 0)
    grid.addWidget(widget, row, 1, 1, span)
    return row + 1


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class HereticTrialSelectionDialog(QDialog):
    """Dialog letting the user pick one of the top trials to re-run."""

    def __init__(
        self,
        *,
        results: list[dict[str, object]],
        recommended_index: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Trial to Re-Run")
        self.setMinimumWidth(580)
        self._selected_trial: int | None = None
        self._trial_results = results  # type: ignore[assignment]

        layout = QVBoxLayout(self)
        label = QLabel(
            "Heretic has completed all trials. The system recommends a choice below, but you can pick any trial to re-run.\n\n"
            "For each trial: "
            "<ul><li>Refusal rate</li><li>KL divergence (lower is better)</li></ul>"
        )
        label.setWordWrap(True)
        label.setFixedHeight(80)
        layout.addWidget(label)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for index, result in enumerate(results):
            trial_num = result.get("trial", index + 1)
            refusals = result.get("refusals")
            kl_div = result.get("kl_divergence")
            mark = "★ " if index == recommended_index else "  "
            text = f"{mark}Trial {trial_num}: {refusals} (KL: {kl_div})"
            radio = QRadioButton(text, self)
            self._group.addButton(radio, index)
            if index == recommended_index:
                radio.setChecked(True)
                self._selected_trial = index
            layout.addWidget(radio)

        # Add "Run this trial" and "Go back to all trials" buttons
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn:
            ok_btn.setText("Run this trial")
            ok_btn.setDefault(True)
        layout.addWidget(box)

        self._group.buttonToggled.connect(self._on_button_toggled)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

    def _on_button_toggled(self, button: QAbstractButton, checked: bool) -> None:
        if checked and button is not None:
            self._selected_trial = self._group.id(button)  # type: ignore[arg-type]

    def selected_trial(self) -> int | None:
        return self._selected_trial


class HereticTab(TabPage):
    def __init__(
        self,
        state: AppState,
        storage: StorageService,
        job_manager: JobManager | None = None,
        registry: EngineRegistry | None = None,
        model_service: ModelService | None = None,
        credentials: CredentialStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "Heretic",
            "Modify a compatible model with the real Heretic engine. Configuration, run "
            "monitoring, and the complete live console stay visible side by side; the "
            "original model files are never modified.",
            parent,
            body_scroll=False,
        )
        self._state = state
        self._storage = storage
        self._job_manager = job_manager or JobManager(parent=self)
        self._registry = registry or EngineRegistry(storage.settings, parent=self)
        self._model_service = model_service or ModelService(storage)
        self._credentials = credentials

        self._active_worker: WorkerProcess | None = None
        self._active_job_id: str | None = None
        self._pending_completion: tuple[bool, str] | None = None
        self._queued_jobs: dict[str, dict] = {}
        self._queue_connected = False
        self._config = HereticConfig.compact()
        self._scorer_rows: list[dict] = []
        self._heretic_status: EngineStatus | None = None
        self._current_profile_id = "compact"
        self._pending_output_dir: str | None = None
        self._pending_artifacts: list[str] | None = None
        self._pending_model_action: str | None = None
        self._latest_metrics: dict[str, float] = {}
        self._best_metrics: dict[str, float] = {}
        self._directions: dict[str, str] = {}
        self._trial_results: dict[int, dict[str, float]] = {}
        self._awaiting_trial_selection: bool = False
        self._run_started_monotonic: float | None = None
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._update_elapsed)

        # -- header state -------------------------------------------------
        self.version_pill = Pill("Heretic · version unknown", "off", self)
        self.worker_pill = Pill("Worker · idle", "off", self)
        self.model_pill = Pill("No model selected", "warn", self)
        for pill in (self.version_pill, self.worker_pill, self.model_pill):
            self.add_header_widget(pill)

        # -- two-column workstation --------------------------------------
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)

        left_scroll = QScrollArea(self.splitter)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._left_content = QWidget(left_scroll)
        self._left_content.setObjectName("TabContent")
        self._left_layout = QVBoxLayout(self._left_content)
        self._left_layout.setContentsMargins(0, 4, 8, 4)
        self._left_layout.setSpacing(12)
        left_scroll.setWidget(self._left_content)

        right = QWidget(self.splitter)
        right.setObjectName("TabContent")
        self._right_layout = QVBoxLayout(right)
        self._right_layout.setContentsMargins(4, 4, 0, 4)
        self._right_layout.setSpacing(12)

        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([640, 560])
        self.content.addWidget(self.splitter, 1)

        self._build_left_column()
        self._build_right_column()

        self.apply_config_to_form(self._config)
        self.refresh()

        self._state.job_changed.connect(self.apply_job)
        self._state.engines_changed.connect(self.apply_engines)
        self._state.models_changed.connect(self._update_model_choices)
        self._state.hardware_changed.connect(self._update_hardware)

    # ==================================================================
    # Left column
    # ==================================================================

    def _build_left_column(self) -> None:
        layout = self._left_layout

        self.engine_card = EngineCard(
            "Heretic engine",
            "Heretic runs in its own Python environment and uses a separate worker "
            "process. It never shares the GPU with Unsloth.",
            self._left_content,
        )
        layout.addWidget(self.engine_card)

        layout.addWidget(self._build_source_card())
        layout.addWidget(self._build_profile_card())
        layout.addWidget(self._build_modifier_section())
        layout.addWidget(self._build_scoring_section())
        layout.addWidget(self._build_execution_section())
        layout.addWidget(self._build_optimization_section())
        layout.addWidget(self._build_output_section())
        layout.addWidget(self._build_toml_section())

        notes = Card(
            "What to expect",
            "Heretic is a training-free abliteration tool, not a fine-tuning run. A "
            "profile with many trials can take hours; the compact profile uses 4-bit "
            "quantization, two trials and short prompt slices for a 6 GB GPU.",
            self._left_content,
        )
        notes.add(
            FieldRow(
                "Install",
                "Upstream requires Python 3.10+ with a matching PyTorch build installed "
                "in the same environment, then pip install -U heretic-llm.",
                caption_width=104,
            )
        )
        notes.add(
            FieldRow(
                "Licence",
                "Heretic is AGPL-3.0 and is invoked as an installed tool in its own "
                "environment; it is not bundled with this application.",
                caption_width=104,
            )
        )
        layout.addWidget(notes)
        layout.addStretch(1)

    def _build_source_card(self) -> Card:
        card = Card(
            "Source model",
            "The source is read only. Heretic writes a new model to the output "
            "destination configured below.",
            self._left_content,
        )
        grid = _form_grid()
        row = 0

        model_row = QWidget(card)
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        self.model_combo = QComboBox(model_row)
        self.model_combo.setEditable(True)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.model_combo.setMinimumContentsLength(12)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.model_combo.setPlaceholderText(
            "Select a local model or enter a Hugging Face repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
        )
        self.model_combo.addItem("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
        self.model_combo.setCurrentIndex(0)
        self.model_combo.currentIndexChanged.connect(self._update_model_metadata)
        self.model_combo.editTextChanged.connect(self._update_model_metadata)
        model_layout.addWidget(self.model_combo, 1)
        self.browse_model_button = QPushButton("Browse…", model_row)
        self.browse_model_button.clicked.connect(self._browse_model_folder)
        model_layout.addWidget(self.browse_model_button)
        row = _add_form_row(grid, row, "Model:", model_row, span=2)

        self.model_commit_edit = QLineEdit(card)
        self.model_commit_edit.setPlaceholderText("blank = repository default, or a commit hash")
        row = _add_form_row(grid, row, "Revision:", self.model_commit_edit, span=2)

        self.model_meta_label = QLabel("", card)
        self.model_meta_label.setObjectName("Muted")
        self.model_meta_label.setWordWrap(True)
        row = _add_form_row(grid, row, "Library metadata:", self.model_meta_label, span=2)

        note = QLabel(
            "Local library models are listed with their detected format and size. An "
            "external repository ID is resolved and validated by Heretic at load time.",
            card,
        )
        note.setObjectName("Muted")
        note.setWordWrap(True)
        grid.addWidget(note, row, 0, 1, 3)
        card.add_layout(grid)
        return card

    def _build_profile_card(self) -> Card:
        card = Card(
            "Modification profile",
            "Profiles are upstream Heretic starting points, not locked modes. Loading "
            "one replaces the form values below; every value stays editable.",
            self._left_content,
        )
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)
        row_layout.addWidget(QLabel("Profile:", card))
        self.preset_combo = QComboBox(card)
        self.preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.preset_combo.setMinimumContentsLength(14)
        for preset_id, label in available_presets():
            self.preset_combo.addItem(label, preset_id)
        self.preset_combo.currentIndexChanged.connect(self._on_profile_changed)
        row_layout.addWidget(self.preset_combo, 1)
        self.load_preset_button = QPushButton("Load profile", card)
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        row_layout.addWidget(self.load_preset_button)
        card.add_layout(row_layout)

        self.profile_origin_label = QLabel("", card)
        self.profile_origin_label.setObjectName("Muted")
        self.profile_origin_label.setWordWrap(True)
        card.add(self.profile_origin_label)
        self._select_preset("compact")
        self._on_profile_changed()
        return card

    def _build_modifier_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Modifier",
            subtitle=(
                "Heretic applies exactly one modifier. ARA is the Heretic 2.x default; "
                "directional abliteration is the classic method."
            ),
            expanded=True,
            parent=self._left_content,
        )
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Modifier:", section))
        self.modifier_combo = _compact_combo(QComboBox(section), 14)
        self.modifier_combo.addItems([*_MODIFIER_LABELS_ORDER(), _CUSTOM])
        self.modifier_combo.currentTextChanged.connect(self._on_modifier_changed)
        row.addWidget(self.modifier_combo, 1)
        self.modifier_plugin_edit = QLineEdit(section)
        self.modifier_plugin_edit.setPlaceholderText("Custom plugin import path or file.py:Class")
        self.modifier_plugin_edit.setEnabled(False)
        row.addWidget(self.modifier_plugin_edit, 2)
        section.add_layout(row)

        advanced = CollapsibleSection(
            "Advanced modifier prompt sources",
            subtitle=(
                "Prompt datasets used by the selected modifier. Scorer-specific options "
                "such as marker lists and classifier models remain editable in Raw TOML."
            ),
            expanded=False,
            parent=section,
        )
        grid = _form_grid()
        self.good_dataset_edit = QLineEdit(advanced)
        self.good_split_edit = QLineEdit(advanced)
        self.good_column_edit = QLineEdit(advanced)
        self.bad_dataset_edit = QLineEdit(advanced)
        self.bad_split_edit = QLineEdit(advanced)
        self.bad_column_edit = QLineEdit(advanced)
        for index, (name, dataset, split, column) in enumerate(
            (
                (
                    "Good prompts",
                    self.good_dataset_edit,
                    self.good_split_edit,
                    self.good_column_edit,
                ),
                ("Bad prompts", self.bad_dataset_edit, self.bad_split_edit, self.bad_column_edit),
            ),
            start=0,
        ):
            dataset.setPlaceholderText("dataset id or path")
            split.setPlaceholderText("split")
            column.setPlaceholderText("column")
            grid.addWidget(QLabel(name, advanced), index, 0)
            grid.addWidget(dataset, index, 1, 1, 2)
            grid.addWidget(split, index, 3)
            grid.addWidget(column, index, 4)
        advanced.add_layout(grid)
        section.add(advanced)
        return section

    def _build_scoring_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Scoring",
            subtitle=(
                "Scorers are co-optimized by Optuna. At least one scorer must be set to "
                "minimize or maximize."
            ),
            expanded=True,
            parent=self._left_content,
        )
        self.scorers_layout = QVBoxLayout()
        self.scorers_layout.setSpacing(6)
        section.add_layout(self.scorers_layout)
        add_scorer = QPushButton("Add scorer", section)
        add_scorer.clicked.connect(lambda: self._add_scorer_row())
        section.add(add_scorer)

        advanced = CollapsibleSection(
            "Advanced scoring prompt sources",
            subtitle=(
                "Prompt datasets for the keyword-rate and KL-divergence scorers. "
                "Other scorer options are edited in Raw TOML."
            ),
            expanded=False,
            parent=section,
        )
        grid = _form_grid()
        self.kw_dataset_edit = QLineEdit(advanced)
        self.kw_split_edit = QLineEdit(advanced)
        self.kw_column_edit = QLineEdit(advanced)
        self.kl_dataset_edit = QLineEdit(advanced)
        self.kl_split_edit = QLineEdit(advanced)
        self.kl_column_edit = QLineEdit(advanced)
        for index, (name, dataset, split, column) in enumerate(
            (
                ("Keyword scorer", self.kw_dataset_edit, self.kw_split_edit, self.kw_column_edit),
                ("KL scorer", self.kl_dataset_edit, self.kl_split_edit, self.kl_column_edit),
            ),
            start=0,
        ):
            dataset.setPlaceholderText("dataset id or path")
            split.setPlaceholderText("split")
            column.setPlaceholderText("column")
            grid.addWidget(QLabel(name, advanced), index, 0)
            grid.addWidget(dataset, index, 1, 1, 2)
            grid.addWidget(split, index, 3)
            grid.addWidget(column, index, 4)
        advanced.add_layout(grid)
        section.add(advanced)
        return section

    def _build_execution_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Hardware & execution",
            subtitle=(
                "Real Heretic 2.x loading options. The detected hardware is advisory; "
                "Heretic decides what actually fits when it loads the model."
            ),
            expanded=True,
            parent=self._left_content,
        )
        self.hardware_summary_label = QLabel("Detecting hardware…", section)
        self.hardware_summary_label.setObjectName("Muted")
        self.hardware_summary_label.setWordWrap(True)
        section.add(self.hardware_summary_label)

        self.vram_advisory_label = QLabel("", section)
        self.vram_advisory_label.setObjectName("Muted")
        self.vram_advisory_label.setWordWrap(True)
        section.add(self.vram_advisory_label)

        grid = _form_grid()
        row = 0
        self.dtypes_edit = QLineEdit(section)
        self.dtypes_edit.setPlaceholderText("auto, float16, bfloat16, float32")
        row = _add_form_row(grid, row, "Dtypes to try:", self.dtypes_edit, span=2)

        self.quant_combo = _compact_combo(QComboBox(section), 12)
        self.quant_combo.addItems(list(QUANTIZATIONS))
        self.quant_combo.setToolTip(
            "Quantization methods exposed by the installed Heretic schema. Upstream "
            "documents bnb_4bit as reducing model VRAM by roughly 70%, and it requires "
            "bitsandbytes in the Heretic environment."
        )
        row = _add_form_row(grid, row, "Quantization:", self.quant_combo, span=2)

        self.device_map_edit = QLineEdit(section)
        self.device_map_edit.setPlaceholderText('auto, or a JSON device table such as {"0": 0}')
        row = _add_form_row(grid, row, "Device map:", self.device_map_edit, span=2)

        self.max_memory_edit = QLineEdit(section)
        self.max_memory_edit.setPlaceholderText(
            'blank = automatic; e.g. {"0": "5GB", "cpu": "32GB"}'
        )
        row = _add_form_row(grid, row, "Max memory (JSON):", self.max_memory_edit, span=2)

        self.offload_check = QCheckBox("Move analysis tensors to CPU", section)
        row = _add_form_row(grid, row, "Memory offload:", self.offload_check, span=2)

        self.batch_spin = QSpinBox(section)
        self.batch_spin.setRange(0, 4096)
        self.batch_spin.setToolTip("0 lets Heretic benchmark an optimal batch size.")
        row = _add_form_row(grid, row, "Batch size (0 = auto):", self.batch_spin, span=2)

        self.max_batch_spin = QSpinBox(section)
        self.max_batch_spin.setRange(1, 4096)
        row = _add_form_row(grid, row, "Max batch size:", self.max_batch_spin, span=2)

        self.max_response_spin = QSpinBox(section)
        self.max_response_spin.setRange(1, 8192)
        row = _add_form_row(grid, row, "Max response tokens:", self.max_response_spin, span=2)

        note = QLabel(
            "Heretic decides what actually fits when it loads the model. These values "
            "are passed through unchanged, and the engine log reports the real result.",
            section,
        )
        note.setObjectName("Muted")
        note.setWordWrap(True)
        grid.addWidget(note, row, 0, 1, 3)
        row += 1
        section.add_layout(grid)

        advanced = CollapsibleSection(
            "Advanced generation settings",
            expanded=False,
            parent=section,
        )
        advanced_grid = _form_grid()
        adv_row = 0
        self.response_prefix_edit = QLineEdit(advanced)
        self.response_prefix_edit.setPlaceholderText("blank = detect automatically")
        adv_row = _add_form_row(
            advanced_grid, adv_row, "Response prefix:", self.response_prefix_edit, span=2
        )
        self.system_prompt_edit = QLineEdit(advanced)
        adv_row = _add_form_row(
            advanced_grid, adv_row, "System prompt:", self.system_prompt_edit, span=2
        )
        self.cot_edit = QTextEdit(advanced)
        self.cot_edit.setMaximumHeight(64)
        self.cot_edit.setPlaceholderText("One 'initializer :: finalizer' pair per line")
        adv_row = _add_form_row(advanced_grid, adv_row, "CoT blocks:", self.cot_edit, span=2)
        self.debug_check = QCheckBox("Print torch/debug information", advanced)
        adv_row = _add_form_row(advanced_grid, adv_row, "Debug output:", self.debug_check, span=2)
        advanced.add_layout(advanced_grid)
        section.add(advanced)
        return section

    def _build_optimization_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Optimization",
            expanded=True,
            parent=self._left_content,
        )
        grid = _form_grid()
        row = 0
        self.trials_spin = QSpinBox(section)
        self.trials_spin.setRange(1, 10_000)
        row = _add_form_row(grid, row, "Optimization trials:", self.trials_spin, span=2)

        self.startup_spin = QSpinBox(section)
        self.startup_spin.setRange(0, 10_000)
        row = _add_form_row(grid, row, "Startup (random) trials:", self.startup_spin, span=2)

        self.seed_edit = QLineEdit(section)
        self.seed_edit.setPlaceholderText("blank = random seed")
        row = _add_form_row(grid, row, "Seed:", self.seed_edit, span=2)

        self.checkpoint_combo = _compact_combo(QComboBox(section), 16)
        self.checkpoint_combo.addItems(["Restart from scratch", "Continue previous run"])
        row = _add_form_row(grid, row, "Existing checkpoint:", self.checkpoint_combo, span=2)

        self.trial_index_spin = QSpinBox(section)
        self.trial_index_spin.setRange(0, 9999)
        self.trial_index_spin.setToolTip(
            "Index in the sorted Pareto front to export. Upstream notes that the trial "
            "with the fewest refusals is not always best: any trial with fewer than about "
            "10 refusals in 100 suppresses refusals, and the lowest KL divergence among "
            "those best preserves the original model."
        )
        row = _add_form_row(grid, row, "Trial to export:", self.trial_index_spin, span=2)

        self.checkpoint_label = QLabel("(set by the output destination)", section)
        self.checkpoint_label.setObjectName("Muted")
        self.checkpoint_label.setWordWrap(True)
        row = _add_form_row(grid, row, "Study checkpoint dir:", self.checkpoint_label, span=2)
        section.add_layout(grid)
        return section

    def _build_output_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Output & provenance",
            subtitle=(
                "The modified model is written to a separate destination. A run.json "
                "record stores the source model, installed Heretic version, starting "
                "profile, execution settings and the exact config.toml."
            ),
            expanded=True,
            parent=self._left_content,
        )
        grid = _form_grid()
        row = 0

        destination_row = QWidget(section)
        destination_layout = QHBoxLayout(destination_row)
        destination_layout.setContentsMargins(0, 0, 0, 0)
        destination_layout.setSpacing(8)
        self.destination_edit = QLineEdit(destination_row)
        self.destination_edit.setPlaceholderText("blank = the configured outputs folder")
        self.destination_edit.textChanged.connect(self._update_provenance_label)
        destination_layout.addWidget(self.destination_edit, 1)
        self.browse_destination_button = QPushButton("Browse…", destination_row)
        self.browse_destination_button.clicked.connect(self._browse_destination)
        destination_layout.addWidget(self.browse_destination_button)
        row = _add_form_row(grid, row, "Destination:", destination_row, span=2)

        self.output_name_edit = QLineEdit(section)
        self.output_name_edit.setPlaceholderText("e.g. Qwen2.5-0.5B-Instruct-Modified")
        self.output_name_edit.setToolTip(
            "Name of the saved modified model in the Models library. This is a new "
            "output folder; the selected source model is not renamed or overwritten."
        )
        self.output_name_edit.textChanged.connect(self._update_provenance_label)
        row = _add_form_row(grid, row, "Output model name:", self.output_name_edit, span=2)

        self.outputs_path = StoragePathRow("outputs_dir", "Outputs folder", section)
        self.outputs_path.open_requested.connect(self._open_outputs)
        grid.addWidget(self.outputs_path, row, 0, 1, 3)
        row += 1

        self.export_combo = _compact_combo(QComboBox(section), 12)
        self.export_combo.addItems(list(EXPORT_STRATEGIES))
        row = _add_form_row(grid, row, "Export strategy:", self.export_combo, span=2)

        self.model_action_combo = _compact_combo(QComboBox(section), 12)
        self.model_action_combo.addItems(list(MODEL_ACTIONS))
        row = _add_form_row(grid, row, "After optimization:", self.model_action_combo, span=2)

        self.upload_repo_edit = QLineEdit(section)
        self.upload_repo_edit.setPlaceholderText("username/model-heretic")
        row = _add_form_row(grid, row, "HF repository:", self.upload_repo_edit, span=2)

        self.private_check = QCheckBox("Create a private repository", section)
        row = _add_form_row(grid, row, "Upload visibility:", self.private_check, span=2)

        self.repro_combo = _compact_combo(QComboBox(section), 12)
        self.repro_combo.addItems(["(not set)", *REPRODUCIBILITY])
        row = _add_form_row(grid, row, "Reproducibility info:", self.repro_combo, span=2)

        self.modified_type_edit = QLineEdit(section)
        row = _add_form_row(grid, row, "Modified model type:", self.modified_type_edit, span=2)

        self.shard_edit = QLineEdit(section)
        row = _add_form_row(grid, row, "Max shard size:", self.shard_edit, span=2)

        self.tags_edit = QLineEdit(section)
        self.tags_edit.setPlaceholderText("comma-separated tags")
        row = _add_form_row(grid, row, "Upload tags:", self.tags_edit, span=2)

        self.shell_check = QCheckBox("Offer in-context Python shell (unrestricted)", section)
        row = _add_form_row(grid, row, "Advanced:", self.shell_check, span=2)

        self.provenance_label = QLabel("", section)
        self.provenance_label.setObjectName("Muted")
        self.provenance_label.setWordWrap(True)
        self.provenance_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        grid.addWidget(self.provenance_label, row, 0, 1, 3)
        section.add_layout(grid)
        return section

    def _build_toml_section(self) -> CollapsibleSection:
        section = CollapsibleSection(
            "Raw TOML",
            subtitle=(
                "Edit anything not represented by a widget, including custom scorer "
                "and modifier plugins. Unknown keys are preserved."
            ),
            expanded=False,
            parent=self._left_content,
        )
        self.toml_view = QTextEdit(section)
        self.toml_view.setStyleSheet(
            "font-family: 'Cascadia Mono', Consolas, monospace; font-size: 8.5pt;"
        )
        self.toml_view.setMinimumHeight(220)
        section.add(self.toml_view, 1)

        buttons = QHBoxLayout()
        refresh = QPushButton("Refresh from form", section)
        refresh.clicked.connect(self._refresh_toml_view)
        apply_toml = QPushButton("Apply TOML to form", section)
        apply_toml.setObjectName("Primary")
        apply_toml.clicked.connect(self._apply_toml_to_form)
        validate = QPushButton("Validate", section)
        validate.clicked.connect(self._validate_config)
        open_file = QPushButton("Open…", section)
        open_file.clicked.connect(self._open_config_file)
        save_file = QPushButton("Save…", section)
        save_file.clicked.connect(self._save_config_file)
        for button in (refresh, apply_toml, validate, open_file, save_file):
            buttons.addWidget(button)
        buttons.addStretch(1)
        section.add_layout(buttons)

        self.toml_status = QLabel("", section)
        self.toml_status.setWordWrap(True)
        section.add(self.toml_status)
        return section

    # ==================================================================
    # Right column
    # ==================================================================

    def _build_right_column(self) -> None:
        monitor = Card("Run monitor", None, self)
        monitor.layout().setContentsMargins(14, 10, 14, 10)
        monitor.body.setSpacing(4)

        self.run_button = QPushButton("Start Heretic", monitor)
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._start_heretic)
        monitor.add_header_widget(self.run_button)
        self.cancel_button = QPushButton("Cancel", monitor)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_heretic)
        monitor.add_header_widget(self.cancel_button)
        self.open_output_button = QPushButton("Open output", monitor)
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output_location)
        monitor.add_header_widget(self.open_output_button)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.run_state_row = FieldRow("Job state", "Idle", caption_width=88)
        self.stage_row = FieldRow("Stage", "Idle", caption_width=88)
        self.model_loading_row = FieldRow("Model load", "Not started", caption_width=88)
        self.prompt_prep_row = FieldRow("Prompts", "Not started", caption_width=88)
        self.elapsed_row = FieldRow("Elapsed", "00:00", caption_width=88)
        self.export_status_row = FieldRow("Export", "Not started", caption_width=88)
        self.error_status_row = FieldRow("Errors", "None", caption_width=88)
        grid.addWidget(self.run_state_row, 0, 0)
        grid.addWidget(self.stage_row, 0, 1)
        grid.addWidget(self.model_loading_row, 1, 0)
        grid.addWidget(self.prompt_prep_row, 1, 1)
        grid.addWidget(self.elapsed_row, 2, 0)
        grid.addWidget(self.export_status_row, 2, 1)
        grid.addWidget(self.error_status_row, 3, 0, 1, 2)
        monitor.add_layout(grid)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        self.trial_metric_label = QLabel("Trial: —", monitor)
        self.trial_metric_label.setObjectName("StripValue")
        self.refusal_metric_label = QLabel("Refusal: —", monitor)
        self.refusal_metric_label.setObjectName("StripValue")
        self.kl_metric_label = QLabel("KL Div: —", monitor)
        self.kl_metric_label.setObjectName("StripValue")
        self.extra_metric_label = QLabel("", monitor)
        self.extra_metric_label.setObjectName("StripValue")
        self.best_score_label = QLabel("Best: —", monitor)
        self.best_score_label.setObjectName("StripValue")
        for label in (
            self.trial_metric_label,
            self.refusal_metric_label,
            self.kl_metric_label,
            self.extra_metric_label,
            self.best_score_label,
        ):
            metrics_layout.addWidget(label)
        metrics_layout.addStretch(1)

        self.progress_bar = QProgressBar(monitor)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(180)
        metrics_layout.addWidget(self.progress_bar)
        monitor.add_layout(metrics_layout)

        self.last_model_row = FieldRow("Saved model", "None saved yet", caption_width=88)
        monitor.add(self.last_model_row)

        console_card = Card("Live Heretic output", None, self)
        console_card.body.setSpacing(4)
        self.console = ConsoleWidget(max_visible_blocks=4000, parent=console_card)
        self.console.saved.connect(
            lambda path: self._state.notify("info", f"Heretic console log saved to {path}")
        )
        self.console.save_failed.connect(
            lambda error: self._state.notify(
                "error", f"Could not save Heretic console log: {error}"
            )
        )
        console_card.add(self.console, 1)
        # Backwards-compatible alias used by existing tests and callers.
        self.log_view = self.console.text_edit

        self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setHandleWidth(8)
        self.right_splitter.addWidget(monitor)
        self.right_splitter.addWidget(console_card)
        self.right_splitter.setStretchFactor(0, 0)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setSizes([170, 460])
        monitor.setMinimumHeight(150)
        console_card.setMinimumHeight(300)
        self._right_layout.addWidget(self.right_splitter, 1)

        self.console.start_session("Idle", "No Heretic run has started in this session.")
        self.console.append(
            "Console ready. Start a run to stream Heretic stdout and stderr.",
            level="system",
        )

    # ==================================================================
    # Form <-> config
    # ==================================================================

    @staticmethod
    def _find_combo(combo: QComboBox, value: str | None) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _on_modifier_changed(self, text: str) -> None:
        self.modifier_plugin_edit.setEnabled(text == _CUSTOM)

    def _add_scorer_row(self, entry: PluginEntry | None = None) -> None:
        row: dict[str, QWidget] = {}
        combo = QComboBox()
        combo.addItems([*_SCORER_ORDER(), _CUSTOM])
        custom = QLineEdit()
        custom.setPlaceholderText("plugin import path or file.py:Class")
        custom.setEnabled(False)
        optimization = QComboBox()
        optimization.addItems(list(OPTIMIZATIONS))
        instance = QLineEdit()
        instance.setPlaceholderText("instance name (optional)")
        remove = QPushButton("Remove")

        def on_combo(text: str) -> None:
            custom.setEnabled(text == _CUSTOM)

        combo.currentTextChanged.connect(on_combo)
        row.update(
            plugin=combo,
            custom=custom,
            optimization=optimization,
            instance=instance,
            remove=remove,
        )
        self._scorer_rows.append(row)
        container = QWidget()
        line = QHBoxLayout(container)
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(combo, 2)
        line.addWidget(custom, 3)
        line.addWidget(optimization, 1)
        line.addWidget(instance, 2)
        line.addWidget(remove)
        self.scorers_layout.addWidget(container)
        remove.clicked.connect(lambda: self._remove_scorer_row(row, container))

        plugin = entry.plugin if entry else SCORER_LABELS[next(iter(SCORER_LABELS))]
        if plugin in _SCORER_BY_PATH:
            combo.setCurrentText(_SCORER_BY_PATH[plugin])
        else:
            combo.setCurrentText(_CUSTOM)
            custom.setText(plugin)
        if entry and entry.optimization:
            optimization.setCurrentText(entry.optimization)
        if entry and entry.instance_name:
            instance.setText(entry.instance_name)

    def _remove_scorer_row(self, row: dict, container: QWidget) -> None:
        self._scorer_rows = [item for item in self._scorer_rows if item is not row]
        self.scorers_layout.removeWidget(container)
        container.setParent(None)

    def _collect_scorers(self) -> list[PluginEntry]:
        entries: list[PluginEntry] = []
        for row in self._scorer_rows:
            plugin = row["plugin"].currentText()
            path = (
                row["custom"].text().strip() if plugin == _CUSTOM else SCORER_LABELS.get(plugin, "")
            )
            if not path:
                continue
            entries.append(
                PluginEntry(
                    plugin=path,
                    optimization=row["optimization"].currentText(),
                    instance_name=row["instance"].text().strip() or None,
                )
            )
        return entries

    def _write_dataset(
        self,
        tables: dict[str, dict],
        class_name: str,
        key: str,
        dataset: QLineEdit,
        split: QLineEdit,
        column: QLineEdit,
    ) -> None:
        existing = dict(tables.get(class_name, {}).get(key, {}) or {})
        if dataset.text().strip():
            existing["dataset"] = dataset.text().strip()
        else:
            existing.pop("dataset", None)
        if split.text().strip():
            existing["split"] = split.text().strip()
        else:
            existing.pop("split", None)
        if column.text().strip():
            existing["column"] = column.text().strip()
        else:
            existing.pop("column", None)
        table = tables.setdefault(class_name, {})
        if existing:
            table[key] = existing
        else:
            table.pop(key, None)

    def config_from_form(self) -> HereticConfig:
        """Build a config from widgets, preserving fields only in Raw TOML."""
        cfg = copy.deepcopy(self._config)
        cfg.dtypes = [item.strip() for item in self.dtypes_edit.text().split(",") if item.strip()]
        cfg.quantization = self.quant_combo.currentText()
        device_text = self.device_map_edit.text().strip()
        if device_text.startswith("{"):
            try:
                parsed_map = json.loads(device_text)
            except ValueError as error:
                raise HereticConfigError(f"device_map: {error}") from error
            if not isinstance(parsed_map, dict) or not all(
                isinstance(key, str) and isinstance(item, (int, str)) and not isinstance(item, bool)
                for key, item in parsed_map.items()
            ):
                raise HereticConfigError(
                    "device_map: expected an object of device -> integer/string values"
                )
            cfg.device_map = dict(parsed_map)
        else:
            cfg.device_map = device_text or "auto"
        memory_text = self.max_memory_edit.text().strip()
        if memory_text:
            try:
                parsed = json.loads(memory_text)
                if not isinstance(parsed, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
                ):
                    raise ValueError("expected an object of device -> memory strings")
                cfg.max_memory = dict(parsed)
            except ValueError as error:
                raise HereticConfigError(f"max_memory: {error}") from error
        else:
            cfg.max_memory = None
        cfg.offload_outputs_to_cpu = self.offload_check.isChecked()
        cfg.batch_size = self.batch_spin.value()
        cfg.max_batch_size = self.max_batch_spin.value()
        cfg.max_response_length = self.max_response_spin.value()
        cfg.response_prefix = self.response_prefix_edit.text().strip() or None
        cfg.system_prompt = self.system_prompt_edit.text()
        blocks: list[list[str]] = []
        for line in self.cot_edit.toPlainText().splitlines():
            if " :: " in line:
                initializer, finalizer = line.split(" :: ", 1)
                blocks.append([initializer.strip(), finalizer.strip()])
        cfg.chain_of_thought_blocks = blocks
        cfg.print_debug_information = self.debug_check.isChecked()

        cfg.modifiers = [
            PluginEntry(
                plugin=(
                    self.modifier_plugin_edit.text().strip()
                    if self.modifier_combo.currentText() == _CUSTOM
                    else _MODIFIER_BY_LABEL.get(self.modifier_combo.currentText(), "")
                )
            )
        ]
        cfg.scorers = self._collect_scorers()

        cfg.n_trials = self.trials_spin.value()
        cfg.n_startup_trials = self.startup_spin.value()
        seed_text = self.seed_edit.text().strip()
        try:
            cfg.seed = int(seed_text) if seed_text else None
        except ValueError as error:
            raise HereticConfigError(f"seed must be an integer: {error}") from error
        cfg.checkpoint_action = CHECKPOINT_ACTIONS[
            0 if self.checkpoint_combo.currentIndex() <= 0 else 1
        ]
        cfg.trial_index = self.trial_index_spin.value()

        cfg.export_strategy = self.export_combo.currentText()
        cfg.model_action = self.model_action_combo.currentText()
        cfg.upload_repo_id = self.upload_repo_edit.text().strip() or None
        cfg.upload_repo_private = self.private_check.isChecked()
        repro = self.repro_combo.currentText()
        cfg.upload_reproducibility_information = None if repro == "(not set)" else repro
        cfg.modified_model_type = self.modified_type_edit.text().strip() or "decensored"
        cfg.max_shard_size = self.shard_edit.text().strip() or "5GB"
        cfg.upload_tags = [tag.strip() for tag in self.tags_edit.text().split(",") if tag.strip()]
        cfg.enable_shell = self.shell_check.isChecked()

        tables = cfg.plugin_tables
        modifier_tables = tables.setdefault("modifier", {})
        class_name = cfg.modifier_class_name()
        self._write_dataset(
            modifier_tables,
            class_name,
            "good_prompts",
            self.good_dataset_edit,
            self.good_split_edit,
            self.good_column_edit,
        )
        self._write_dataset(
            modifier_tables,
            class_name,
            "bad_prompts",
            self.bad_dataset_edit,
            self.bad_split_edit,
            self.bad_column_edit,
        )
        scorer_tables = tables.setdefault("scorer", {})
        self._write_dataset(
            scorer_tables,
            "KeywordRate",
            "prompts",
            self.kw_dataset_edit,
            self.kw_split_edit,
            self.kw_column_edit,
        )
        self._write_dataset(
            scorer_tables,
            "KLDivergence",
            "prompts",
            self.kl_dataset_edit,
            self.kl_split_edit,
            self.kl_column_edit,
        )
        cfg.model = self._selected_model()
        cfg.model_commit = self.model_commit_edit.text().strip() or None
        return cfg

    def apply_config_to_form(self, cfg: HereticConfig) -> None:
        self._config = copy.deepcopy(cfg)
        self.dtypes_edit.setText(", ".join(cfg.dtypes))
        self.quant_combo.setCurrentText(cfg.quantization)
        self.device_map_edit.setText(
            json.dumps(cfg.device_map) if isinstance(cfg.device_map, dict) else cfg.device_map
        )
        self.max_memory_edit.setText(json.dumps(cfg.max_memory) if cfg.max_memory else "")
        self.offload_check.setChecked(cfg.offload_outputs_to_cpu)
        self.batch_spin.setValue(cfg.batch_size)
        self.max_batch_spin.setValue(cfg.max_batch_size)
        self.max_response_spin.setValue(cfg.max_response_length)
        self.response_prefix_edit.setText(cfg.response_prefix or "")
        self.system_prompt_edit.setText(cfg.system_prompt)
        self.cot_edit.setPlainText(
            "\n".join(" :: ".join(pair) for pair in cfg.chain_of_thought_blocks)
        )
        self.debug_check.setChecked(cfg.print_debug_information)

        modifier_path = cfg.modifiers[0].plugin if cfg.modifiers else ""
        label = _MODIFIER_BY_PATH.get(modifier_path, _CUSTOM)
        self.modifier_combo.setCurrentText(label)
        self.modifier_plugin_edit.setText("" if label != _CUSTOM else modifier_path)
        self.modifier_plugin_edit.setEnabled(label == _CUSTOM)

        for row in list(self._scorer_rows):
            container = row["plugin"].parentWidget()
            self._remove_scorer_row(row, container)
        for entry in cfg.scorers:
            self._add_scorer_row(entry)

        self.trials_spin.setValue(cfg.n_trials)
        self.startup_spin.setValue(cfg.n_startup_trials)
        self.seed_edit.setText("" if cfg.seed is None else str(cfg.seed))
        self.checkpoint_combo.setCurrentIndex(1 if cfg.checkpoint_action == "continue" else 0)
        self.trial_index_spin.setValue(cfg.trial_index or 0)

        self.export_combo.setCurrentText(cfg.export_strategy or "merge")
        self.model_action_combo.setCurrentText(cfg.model_action or "save")
        self.upload_repo_edit.setText(cfg.upload_repo_id or "")
        self.private_check.setChecked(bool(cfg.upload_repo_private))
        self.repro_combo.setCurrentText(cfg.upload_reproducibility_information or "(not set)")
        self.modified_type_edit.setText(cfg.modified_model_type)
        self.shard_edit.setText(str(cfg.max_shard_size))
        self.tags_edit.setText(", ".join(cfg.upload_tags))
        self.shell_check.setChecked(cfg.enable_shell)
        self.model_commit_edit.setText(cfg.model_commit or "")

        tables = cfg.plugin_tables
        modifier_table = tables.get("modifier", {}).get(cfg.modifier_class_name(), {})
        scorer_tables = tables.get("scorer", {})
        pairs = (
            (
                modifier_table,
                "good_prompts",
                self.good_dataset_edit,
                self.good_split_edit,
                self.good_column_edit,
            ),
            (
                modifier_table,
                "bad_prompts",
                self.bad_dataset_edit,
                self.bad_split_edit,
                self.bad_column_edit,
            ),
            (
                scorer_tables.get("KeywordRate", {}),
                "prompts",
                self.kw_dataset_edit,
                self.kw_split_edit,
                self.kw_column_edit,
            ),
            (
                scorer_tables.get("KLDivergence", {}),
                "prompts",
                self.kl_dataset_edit,
                self.kl_split_edit,
                self.kl_column_edit,
            ),
        )
        for source, key, dataset, split, column in pairs:
            spec = source.get(key, {}) or {}
            dataset.setText(_dataset_text(spec))
            split.setText(_dataset_split(spec))
            column.setText(_dataset_column(spec))
        self._refresh_toml_view()
        self._update_provenance_label()

    # ==================================================================
    # Raw TOML
    # ==================================================================

    def _refresh_toml_view(self) -> None:
        try:
            cfg = self.config_from_form()
        except HereticConfigError as error:
            self.toml_status.setText(f"Cannot render TOML: {error}")
            return
        self.toml_view.setPlainText(cfg.to_toml())
        self.toml_status.setText("Form settings rendered as config.toml.")

    def _apply_toml_to_form(self) -> None:
        try:
            cfg = HereticConfig.from_toml(
                self.toml_view.toPlainText(), base=HereticConfig.default()
            )
        except HereticConfigError as error:
            self.toml_status.setText(str(error))
            self._state.notify("error", f"Heretic config could not be applied: {error}")
            return
        self.apply_config_to_form(cfg)
        problems = cfg.validate()
        self.toml_status.setText(
            "Applied. " + ("Issues: " + "; ".join(problems) if problems else "Configuration valid.")
        )

    def _validate_config(self) -> list[str]:
        try:
            cfg = HereticConfig.from_toml(
                self.toml_view.toPlainText(), base=HereticConfig.default()
            )
        except HereticConfigError as error:
            self.toml_status.setText(str(error))
            return [str(error)]
        problems = cfg.validate()
        self.toml_status.setText(
            "Configuration valid." if not problems else "Issues: " + "; ".join(problems)
        )
        return problems

    def _open_config_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Heretic config.toml", "", "TOML files (*.toml);;All files (*.*)"
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            self._state.notify("error", f"Could not read config: {error}")
            return
        self.toml_view.setPlainText(text)
        self._apply_toml_to_form()

    def _save_config_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Heretic config.toml", "config.toml", "TOML files (*.toml)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.toml_view.toPlainText(), encoding="utf-8")
        except OSError as error:
            self._state.notify("error", f"Could not save config: {error}")
            return
        self._state.notify("info", f"Heretic config saved to {path}")

    # ==================================================================
    # Setup / state
    # ==================================================================

    def _open_outputs(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def _selected_model(self) -> str:
        text = self.model_combo.currentText().strip()
        index = self.model_combo.currentIndex()
        if index >= 0 and self.model_combo.itemText(index) == text:
            data = self.model_combo.itemData(index)
            if data:
                return str(data)
        return text

    def select_model(self, model_ref: str) -> None:
        idx = self.model_combo.findData(model_ref)
        if idx < 0:
            idx = self.model_combo.findText(model_ref)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(model_ref)
        self._update_model_metadata()

    def _browse_model_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select a local model folder")
        if path:
            self.model_combo.setEditText(path)
            self._update_model_metadata()

    def _browse_destination(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select an output destination")
        if path:
            self.destination_edit.setText(path)

    def _select_preset(self, preset_id: str) -> None:
        index = self.preset_combo.findData(preset_id)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

    def _on_profile_changed(self, *_args) -> None:
        preset_id = self.preset_combo.currentData()
        origin = _profile_origin(preset_id)
        if self._heretic_status is not None:
            major = _major_version(self._heretic_status.version)
            if major is not None and major != 2:
                origin = (
                    f"Unavailable: this profile requires Heretic 2.x; installed version is "
                    f"{self._heretic_status.version}."
                )
        self.profile_origin_label.setText(f"{origin} · {self.preset_combo.currentText()}")
        self._update_provenance_label()

    def _load_selected_preset(self) -> None:
        preset_id = self.preset_combo.currentData()
        if not preset_id:
            return
        if self._heretic_status is not None:
            major = _major_version(self._heretic_status.version)
            if major is not None and major != 2:
                self._state.notify(
                    "warning",
                    f"Profiles require Heretic 2.x; installed version is "
                    f"{self._heretic_status.version}.",
                )
                return
        try:
            cfg = load_preset(preset_id)
        except (HereticConfigError, OSError) as error:
            self._state.notify("error", f"Could not load Heretic profile: {error}")
            return
        self._current_profile_id = preset_id
        self.apply_config_to_form(cfg)
        self._on_profile_changed()
        self._state.notify("info", f"Loaded Heretic profile: {self.preset_combo.currentText()}")

    def _update_model_choices(self, models: list[LocalModelRecord]) -> None:
        previous = self._selected_model()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models:
            if model.kind in ("base", "abliterated"):
                self.model_combo.addItem(f"{model.name} ({model.format.upper()})", str(model.path))
        self.model_combo.blockSignals(False)
        if previous:
            self.select_model(previous)
        elif self.model_combo.count() == 0:
            self.model_combo.addItem("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
            self.model_combo.setCurrentIndex(0)
        self._update_model_metadata()

    def _find_model_record(self, model_ref: str) -> LocalModelRecord | None:
        models = getattr(self._state, "models", None)
        if not isinstance(models, (list, tuple)):
            return None
        for record in models:
            path = str(getattr(record, "path", "") or "")
            name = str(getattr(record, "name", "") or "")
            if model_ref in (path, name) or (path and Path(model_ref) == Path(path)):
                return record
        return None

    def _update_model_metadata(self, *_args) -> None:
        if not hasattr(self, "model_meta_label"):
            return
        model_ref = self._selected_model()
        record = self._find_model_record(model_ref)
        if record is not None:
            params = (
                f" · ~{record.parameter_count / 1e9:.1f}B parameters"
                if record.parameter_count
                else ""
            )
            self.model_meta_label.setText(
                f"{record.architecture} · {record.format.upper()} · {record.size_mb:.1f} MB"
                f"{params}\n{record.status_detail}"
            )
            state = {
                "ok": "ok",
                "warn": "warn",
                "error": "error",
            }.get(str(record.vram_status), "info")
            self.model_pill.set_state(state, f"Model · {record.name}")
        elif model_ref:
            self.model_meta_label.setText(
                "External Hugging Face ID or path. Compatibility is determined by Heretic "
                "when the model is loaded."
            )
            self.model_pill.set_state("info", f"Model · {model_ref}")
        else:
            self.model_meta_label.setText(
                "Select a local model or enter a Hugging Face repository ID."
            )
            self.model_pill.set_state("warn", "No model selected")
        self._update_provenance_label()

    def _update_hardware(self, snapshot: object) -> None:
        if not hasattr(self, "hardware_summary_label"):
            return
        gpu = getattr(snapshot, "gpu", None) if snapshot is not None else None
        error = getattr(snapshot, "gpu_error", None) if snapshot is not None else None
        if gpu is None:
            reason = f" ({error})" if error else ""
            self.hardware_summary_label.setText(
                f"No NVIDIA GPU detected via nvidia-smi{reason}. Upstream Heretic also "
                "supports AMD GPUs (ROCm) and CPU-only processing, which is orders of "
                "magnitude slower and practical only for tiny models. Check the engine "
                "log for the device Heretic actually selects."
            )
            self.vram_advisory_label.setText("")
            return
        name = getattr(gpu, "name", "Unknown GPU")
        total = getattr(gpu, "memory_total_mb", None)
        used = getattr(gpu, "memory_used_mb", None)
        driver = getattr(gpu, "driver_version", None)
        if not isinstance(total, (int, float)):
            total = None
        if not isinstance(used, (int, float)):
            used = None
        parts = [str(name)]
        if total:
            parts.append(f"{total / 1024:.1f} GB VRAM")
        if used is not None and total:
            parts.append(f"{used / 1024:.1f} GB in use")
        if driver:
            parts.append(f"driver {driver}")
        self.hardware_summary_label.setText(" · ".join(parts))
        if total and total <= 8192:
            self.vram_advisory_label.setText(
                f"Detected {total / 1024:.1f} GB VRAM. Upstream guidance is about 2.5 GB "
                "of VRAM per billion parameters, and bnb_4bit can reduce model VRAM by "
                "roughly 70%. Prefer bnb_4bit, a small batch size, and CPU offload. Models "
                "that do not fit fail at load time; this is guidance, not a compatibility "
                "verdict."
            )
        else:
            self.vram_advisory_label.setText(
                "Upstream guidance is about 2.5 GB of VRAM per billion parameters, and "
                "bnb_4bit can reduce model VRAM by roughly 70%. Heretic reports the actual "
                "load result for the selected model and settings."
            )

    def _update_provenance_label(self, *_args) -> None:
        if not hasattr(self, "provenance_label"):
            return
        model = self._selected_model() if hasattr(self, "model_combo") else ""
        version = (
            self._heretic_status.version
            if self._heretic_status is not None and self._heretic_status.version
            else "unknown"
        )
        destination = (
            self.destination_edit.text().strip() if hasattr(self, "destination_edit") else ""
        )
        if not destination:
            try:
                destination = str(self._storage.path_for("outputs_dir") / "heretic_<job>")
            except Exception:  # noqa: BLE001 - label must never break the UI
                destination = "the configured outputs folder"
        profile_label = self.preset_combo.currentText() if hasattr(self, "preset_combo") else ""
        self.provenance_label.setText(
            f"Source: {model or '(none)'} · Heretic {version} · "
            f"starting profile: {profile_label or '(none)'} · output: {destination}"
        )

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
        self._update_model_metadata()
        self._update_hardware(getattr(self._state, "hardware", None))
        self._update_provenance_label()
        self._apply_profile_support()

    def _apply_profile_support(self) -> None:
        if self._heretic_status is None:
            return
        major = _major_version(self._heretic_status.version)
        supported = major is None or major == 2
        self.preset_combo.setEnabled(supported)
        self.load_preset_button.setEnabled(supported)
        if not supported:
            self.profile_origin_label.setText(
                f"Profiles require Heretic 2.x; installed version is "
                f"{self._heretic_status.version}. Configure the run manually or in Raw TOML."
            )

    def apply_engines(self, statuses: dict[str, EngineStatus]) -> None:
        status = statuses.get(HERETIC)
        if status is None:
            self._heretic_status = None
            self.set_status("Checking Heretic", "info")
            self.version_pill.set_state("off", "Heretic · checking")
            self.run_button.setEnabled(False)
            self.run_button.setToolTip("Waiting for the Heretic engine check.")
            return

        self._heretic_status = status
        self.engine_card.apply_status(status)
        version = status.version or "unknown"
        self.version_pill.set_state("ok" if status.is_available else "warn", f"Heretic · {version}")
        is_available = status.is_available
        has_no_active = self._active_worker is None or not self._active_worker.running
        self.run_button.setEnabled(is_available and has_no_active)
        if is_available:
            self.set_status("Heretic engine available", "ok")
            if getattr(self._state, "hardware", None) is None:
                self.run_button.setToolTip(
                    "Start Heretic — GPU information unavailable; model compatibility is "
                    "not verified."
                )
            else:
                self.run_button.setToolTip(
                    "Start Heretic — verify the selected model and settings fit in VRAM."
                )
        else:
            level = "error" if status.badge() == "check failed" else "warn"
            self.set_status(f"Heretic {status.badge()}", level)
            self.run_button.setToolTip(f"Heretic engine unavailable: {status.detail}")
        self._apply_profile_support()
        self._update_provenance_label()

    def apply_job(self, job: JobSnapshot | None) -> None:
        if job is None:
            self.run_state_row.set_value("Idle")
            if self._active_worker is None:
                self._set_worker_pill("Worker · idle", "off")
            self._refresh_run_button()
            return
        self.run_state_row.set_value(f"{job.state.value} · {job.engine}")
        self._set_worker_pill_for_state(job.state)
        if job.state.is_terminal:
            self.cancel_button.setEnabled(False)
            self._run_timer.stop()
            self._refresh_run_button()

    def _refresh_run_button(self) -> None:
        """Enable Start only when the engine is available and no run is active."""
        status = self._heretic_status
        available = status is not None and status.is_available
        busy = self._active_worker is not None and self._active_worker.running
        self.run_button.setEnabled(available and not busy)

    # ==================================================================
    # Launch / protocol
    # ==================================================================

    def _output_paths(self, job_id: str) -> tuple[Path, Path, Path]:
        raw = self.destination_edit.text().strip()
        base = Path(raw) if raw else self._storage.path_for("outputs_dir")
        job_dir = base / f"heretic_{job_id}"
        name = self.output_name_edit.text().strip() or "model"
        return job_dir, job_dir / name, job_dir / "checkpoints"

    def _validate_output_target(self, model_ref: str) -> list[str]:
        problems: list[str] = []
        raw = self.destination_edit.text().strip()
        if raw:
            destination = Path(raw)
            if not destination.is_absolute():
                problems.append(
                    "Output destination must be an absolute path or blank for the outputs folder."
                )
            elif destination.exists() and not destination.is_dir():
                problems.append("Output destination exists but is not a folder.")
        name = self.output_name_edit.text().strip() or "model"
        invalid = '<>:"/\\|?*'
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if (
            name in {".", ".."}
            or len(name) > 120
            or any(char in invalid or ord(char) < 32 for char in name)
            or name.endswith((".", " "))
            or name.split(".", 1)[0].upper() in reserved
        ):
            problems.append(
                "Output model name must be a valid Windows folder name (1–120 characters, "
                "without path separators or reserved characters)."
            )
        source = Path(model_ref)
        if source.is_dir():
            try:
                _, model_dir, _ = self._output_paths("validation")
                source_resolved = source.resolve()
                target_resolved = model_dir.resolve()
                if target_resolved == source_resolved or source_resolved in target_resolved.parents:
                    problems.append(
                        "The output destination would overwrite the source model. Choose a "
                        "different destination or output folder name."
                    )
            except OSError:
                pass
        return problems

    def _start_heretic(self) -> None:
        model_name = self._selected_model()
        if not model_name:
            self._fail_start("Please select or enter a model name to modify.")
            return

        try:
            cfg = self.config_from_form()
        except HereticConfigError as error:
            self._fail_start(f"Invalid Heretic configuration: {error}")
            return
        cfg.model = model_name
        problems = cfg.validate()
        problems.extend(self._validate_output_target(model_name))
        if problems:
            self.toml_status.setText("Issues: " + "; ".join(problems))
            self._fail_start("Heretic configuration is invalid: " + problems[0], notify=False)
            return

        heretic_status = self._registry.status(HERETIC)
        interpreter = heretic_status.interpreter
        if not interpreter or not interpreter.is_file():
            fallback = Path("environments/heretic/Scripts/python.exe").resolve()
            if fallback.is_file():
                interpreter = fallback
            else:
                self._fail_start("Heretic interpreter not found. Set it in Settings.")
                return

        worker_script = Path(__file__).resolve().parents[1] / "workers" / "heretic_worker.py"
        if not worker_script.is_file():
            self._fail_start(f"Worker script not found: {worker_script}")
            return

        self._awaiting_trial_selection = False
        try:
            job = self._job_manager.submit(HERETIC, model_ref=model_name)
        except Exception as error:  # noqa: BLE001 - report, never crash the GUI
            self._fail_start(f"Cannot start Heretic: {error}")
            return

        config_data = self._build_config(job.job_id, model_name, interpreter)
        if job.state is JobState.QUEUED:
            self._queued_jobs[job.job_id] = {
                "config": config_data,
                "interpreter": interpreter,
                "script": worker_script,
            }
            self._connect_queue_signal()
            self.console.append(
                f"Job {job.job_id} queued; it starts when the GPU is free.",
                level="system",
                source="queue",
            )
            self._set_worker_pill("Worker · queued", "info")
            self._state.notify("info", f"Heretic job {job.job_id} queued.")
            return

        self._launch_job(job.job_id, config_data, interpreter, worker_script)

    def _fail_start(self, message: str, *, notify: bool = True) -> None:
        self._set_error_status(message)
        self.console.append(message, level="error", source="validation")
        if notify:
            self._state.notify("error", message)

    def _build_config(
        self,
        job_id: str,
        model_name: str,
        interpreter: object | None = None,
        *,
        trial_index: int | None = None,
    ) -> dict:
        job_dir, model_dir, checkpoints_dir = self._output_paths(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = self.config_from_form()
        except HereticConfigError:
            cfg = copy.deepcopy(self._config)
        cfg.model = model_name
        if trial_index is not None:
            cfg.trial_index = trial_index
        headless = cfg.headless_copy(model_dir, checkpoints_dir)
        self._pending_model_action = headless.model_action
        data = {
            "job_id": job_id,
            "model_name": model_name,
            "trials": headless.n_trials,
            "export_strategy": headless.export_strategy,
            "output_dir": str(model_dir),
            "toml": headless.to_toml(),
        }
        self._write_provenance(job_dir, job_id, model_name, headless, interpreter)
        self.checkpoint_label.setText(str(checkpoints_dir))
        return data

    def _write_provenance(
        self,
        job_dir: Path,
        job_id: str,
        model_name: str,
        headless: HereticConfig,
        interpreter: object | None,
    ) -> None:
        record = {
            "application": "Simple HF Trainer",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "job_id": job_id,
            "source_model": model_name,
            "source_model_commit": headless.model_commit,
            "source_is_local": Path(model_name).is_dir(),
            "heretic_version": (
                self._heretic_status.version if self._heretic_status is not None else None
            ),
            "starting_profile": {
                "id": self._current_profile_id,
                "label": self.preset_combo.currentText(),
            },
            "interpreter": str(interpreter) if interpreter is not None else None,
            "output_dir": str(headless.save_directory),
            "checkpoints_dir": str(headless.study_checkpoint_dir),
            "export_strategy": headless.export_strategy,
            "model_action": headless.model_action,
            "n_trials": headless.n_trials,
            "n_startup_trials": headless.n_startup_trials,
            "seed": headless.seed,
            "quantization": headless.quantization,
            "device_map": headless.device_map,
            "batch_size": headless.batch_size,
            "config_toml": headless.to_toml(),
        }
        try:
            (job_dir / "run.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            self.console.append(
                f"Could not write run metadata to {job_dir / 'run.json'}: {error}",
                level="warning",
                source="provenance",
            )

    def _worker_environment(self, config_data: dict) -> dict[str, str]:
        """Pass the stored HF token to Heretic only for upload jobs."""
        toml_text = config_data.get("toml")
        if not toml_text:
            return {}
        try:
            parsed = HereticConfig.from_toml(toml_text)
        except HereticConfigError:
            return {}
        if parsed.model_action != "upload":
            return {}
        if self._credentials is None or not self._credentials.available:
            return {}
        token = self._credentials.get_token()
        return {"HF_TOKEN": token} if token else {}

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
        self.console.append(
            f"Job {snapshot.job_id} is starting now.",
            level="system",
            source="queue",
        )
        self._launch_job(snapshot.job_id, spec["config"], spec["interpreter"], spec["script"])

    def _launch_job(self, job_id: str, config_data: dict, interpreter, worker_script) -> None:
        try:
            self._active_job_id = job_id
            self._pending_completion = None
            self._pending_output_dir = None
            self._pending_artifacts = None
            self._latest_metrics.clear()
            self._best_metrics.clear()
            self._trial_results.clear()
            self._refresh_objective_directions()
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.open_output_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self._left_content.setEnabled(False)
            self._run_started_monotonic = time.monotonic()
            self._run_timer.start()
            self._set_worker_pill("Worker · starting", "info")
            self._set_stage("Starting worker")
            self.model_loading_row.set_value("Waiting for engine")
            self.prompt_prep_row.set_value("Not started")
            self.export_status_row.set_value("Not started")
            self._set_error_status("None")
            self.trial_metric_label.setText("Trial: 0")
            self.refusal_metric_label.setText("Refusal: —")
            self.kl_metric_label.setText("KL Div: —")
            self.extra_metric_label.setText("")
            self.best_score_label.setText("Best score: —")

            model_name = str(config_data.get("model_name", ""))
            self.console.start_session(
                f"Heretic run {job_id}",
                f"{model_name} · {self.preset_combo.currentText()}",
            )

            scratch_dir = self._storage.path_for("outputs_dir") / ".scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)

            worker = WorkerProcess(
                interpreter=interpreter,
                script=worker_script,
                args=["--config", json.dumps(config_data)],
                cwd=scratch_dir,
                env=self._worker_environment(config_data),
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
        except Exception as error:  # noqa: BLE001 - report, never crash the GUI
            if self._active_worker is not None and self._active_worker.running:
                self._state.notify(
                    "error",
                    f"Heretic worker started but setup reported an error: {error}",
                )
                self.console.append(
                    f"Worker setup error after start: {error}",
                    level="error",
                    source="worker",
                )
            else:
                self._handle_setup_failure(job_id, error)

    def _refresh_objective_directions(self) -> None:
        self._directions = {}
        try:
            cfg = self.config_from_form()
        except HereticConfigError:
            cfg = self._config
        for entry in cfg.scorers:
            metric = _SCORER_METRIC.get(entry.plugin)
            if metric and entry.optimization in ("minimize", "maximize"):
                self._directions[metric] = entry.optimization

    def _handle_setup_failure(self, job_id: str, error: Exception) -> None:
        """Fail a job whose worker could not be prepared or started."""
        self._job_manager.mark_failed(
            job_id,
            message=f"Could not prepare Heretic job: {error}",
        )
        self._active_job_id = None
        self._active_worker = None
        self._pending_completion = None
        self._run_timer.stop()
        self._run_started_monotonic = None
        self._left_content.setEnabled(True)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._set_worker_pill("Worker · failed", "error")
        self._set_error_status(f"Could not prepare Heretic job: {error}")
        self.console.append(
            f"Could not prepare Heretic job: {error}",
            level="error",
            source="worker",
        )
        self._state.notify("error", f"Could not prepare Heretic job: {error}")

    def _cancel_heretic(self) -> None:
        if self._active_job_id and self._active_worker and self._active_worker.running:
            self._job_manager.request_cancel(self._active_job_id)
            sent = self._active_worker.request_cancel(self._active_job_id)
            self.cancel_button.setEnabled(False)
            self._set_worker_pill("Worker · cancelling", "warn")
            self.console.append(
                "Cancellation requested. The worker has up to 10 seconds to stop cleanly "
                "before it is terminated.",
                level="warning",
                source="cancel",
            )
            if sent:
                self._state.notify(
                    "info",
                    "Cancellation requested. Heretic has up to 10 seconds to stop cleanly "
                    "before it is terminated.",
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
            self._set_worker_pill("Worker · running", "ok")
            self._set_stage("Worker started")
            self.console.append(
                f"Worker process started (PID {self._active_worker.pid}).",
                level="system",
                source="worker",
            )

    def _on_worker_log_line(self, line: str) -> None:
        line = str(line).strip()
        if not line:
            return
        lowered = line.lower()
        if any(word in lowered for word in ("traceback", "exception", "error", "failed")):
            level = "error"
        elif "warn" in lowered:
            level = "warning"
        else:
            level = "info"
        self.console.append(line, level=level, source="engine stderr")
        if level in ("warning", "error"):
            self._set_error_status(line)
        self._set_stage_from_text(line)

    def _on_worker_protocol_error(self, error: object) -> None:
        message = str(error)
        self.console.append(message, level="warning", source="protocol warning")
        self._set_error_status(message)

    def _on_worker_message(self, msg: WorkerMessage) -> None:
        payload = msg.payload
        if msg.type is MessageType.HELLO:
            pid = payload.get("pid")
            detail = f" (PID {pid})" if pid else ""
            self.console.append(f"Worker started{detail}.", level="system", source="worker")

        elif msg.type is MessageType.READY:
            engine = payload.get("engine", "Heretic engine")
            version = payload.get("version")
            detail = f" {version}" if version else ""
            self.console.append(f"{engine} ready{detail}.", level="system", source="worker")
            if version:
                self.version_pill.set_state("ok", f"Heretic · {version}")

        elif msg.type is MessageType.ERROR:
            message = str(payload.get("message") or "Unknown worker error.")
            self.console.append(message, level="error", source="engine error")
            self._set_error_status(message)
            self._set_stage("Error")

        elif msg.type is MessageType.PROGRESS:
            frac = float(payload.get("fraction", 0.0))
            self.progress_bar.setValue(int(frac * 100))
            step = payload.get("step")
            total = payload.get("total")
            if step is not None:
                self.trial_metric_label.setText(f"Trial: {step}/{total}")
                if total:
                    self._set_stage(f"Trial {step}/{total}")

        elif msg.type is MessageType.METRIC:
            self._apply_metric(payload)

        elif msg.type is MessageType.LOG:
            message = str(payload.get("message", ""))
            level = str(payload.get("level", "info"))
            if level not in ("info", "warning", "error"):
                level = "info"
            stream = str(payload.get("stream", "") or "")
            source = {
                "stdout": "stdout",
                "stderr": "stderr",
                "system": "worker",
            }.get(stream)
            self.console.append(message, level=level, source=source)
            if level in ("warning", "error"):
                self._set_error_status(message)
            self._set_stage_from_text(message)

        elif msg.type is MessageType.RESULT:
            out_dir = str(payload.get("output_dir", "") or "")
            artifacts = payload.get("artifacts")
            self._pending_output_dir = out_dir or None
            self._pending_artifacts = (
                [str(item) for item in artifacts] if isinstance(artifacts, list) else None
            )
            self.last_model_row.set_value(out_dir)
            self.open_output_button.setEnabled(bool(out_dir))
            action = self._pending_model_action or "save"
            self.export_status_row.set_value(
                f"Uploaded to {self.upload_repo_edit.text().strip()}"
                if action == "upload"
                else f"Saved to {out_dir}"
            )
            self.console.append(
                f"Output: {out_dir}",
                level="system",
                source="export",
            )

        elif msg.type is MessageType.JOB_COMPLETE:
            success = bool(payload.get("success", False))
            message = str(
                payload.get("message")
                or (
                    "Heretic modification completed." if success else "Heretic finished with error."
                )
            )
            self._pending_completion = (success, message)
            self._set_stage("Complete" if success else "Failed")
            self.export_status_row.set_value("Complete" if success else f"Failed: {message}")
            self.console.append(
                message,
                level="info" if success else "error",
                source="job",
            )

    def _apply_metric(self, payload: dict) -> None:
        name = payload.get("name")
        value = payload.get("value")
        step = payload.get("step")
        if not isinstance(name, str) or value is None or step is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return

        # Store per-trial metrics
        if step not in self._trial_results:
            self._trial_results[step] = {}
        self._trial_results[step][name] = numeric

        # Also update the latest metrics (for current trial)
        self._latest_metrics[name] = numeric

        if name == "refusal_rate":
            self.refusal_metric_label.setText(f"Refusal: {numeric * 100:.1f}%")
        elif name == "kl_divergence":
            self.kl_metric_label.setText(f"KL Div: {numeric:.4f}")
        else:
            self.extra_metric_label.setText(f"{name}: {numeric:.4f}")

        direction = self._directions.get(name)
        if direction == "minimize":
            self._best_metrics[name] = min(self._best_metrics.get(name, numeric), numeric)
        elif direction == "maximize":
            self._best_metrics[name] = max(self._best_metrics.get(name, numeric), numeric)
        parts = []
        if "refusal_rate" in self._best_metrics:
            parts.append(f"best refusal {self._best_metrics['refusal_rate'] * 100:.1f}%")
        if "kl_divergence" in self._best_metrics:
            parts.append(f"best KL {self._best_metrics['kl_divergence']:.4f}")
        if parts:
            self.best_score_label.setText("Best observed: " + " · ".join(parts))

    def _show_trial_selection_dialog(
        self, results: list[dict[str, object]], recommended_index: int = 0
    ) -> int | None:
        """Show the trial selection dialog and return the user's choice."""
        dlg = HereticTrialSelectionDialog(
            results=results,
            recommended_index=recommended_index,
            parent=self,
        )
        if dlg.exec() == QDialog.Accepted:
            idx = dlg.selected_trial()
            return (idx + 1) if idx is not None else None
        return None

    def _trial_ranked_results(self, limit: int = 3) -> list[dict[str, object]]:
        """Rank completed trials by the documented Pareto preference.

        Upstream guidance: the trial with the fewest refusals is not always the
        best; among trials whose refusals stay under the accepted ceiling, the
        lowest KL divergence best preserves the original model.  We surface the
        top trials so the user can make an informed choice.
        """
        collected: list[dict[str, object]] = []
        for step in sorted(
            self._trial_results,
            key=lambda value: (int(value), 0) if isinstance(value, int) else (10**9, 0),
        ):
            metrics = self._trial_results.get(step, {})
            refusal = metrics.get("refusal_rate")
            kl = metrics.get("kl_divergence")
            try:
                refusal_value = float(refusal) if refusal is not None else None
            except (TypeError, ValueError):
                refusal_value = None
            try:
                kl_value = float(kl) if kl is not None else None
            except (TypeError, ValueError):
                kl_value = None
            collected.append(
                {
                    "trial": int(step) + 1,
                    "refusals": refusal_value,
                    "refusal_rate": refusal_value,
                    "kl_divergence": kl_value,
                }
            )

        # System recommendation: prefer the lowest KL divergence among trials
        # whose refusal rate is within the documented acceptable ceiling.
        def _score(trial: dict[str, object]) -> tuple:
            refusal_rate = trial.get("refusal_rate")
            kl = trial.get("kl_divergence")
            refusal_penalty = (
                0.0 if (refusal_rate is not None and float(refusal_rate) < 0.10) else 1.0
            )
            kl_value = float(kl) if kl is not None else float("inf")
            return (refusal_penalty, kl_value)

        ranked = sorted(collected, key=_score)

        results: list[dict[str, object]] = []
        for trial in ranked:
            result: dict[str, object] = dict(trial)  # type: ignore[arg-type]
            refusal_rate = trial.get("refusal_rate")
            kl_divergence = trial.get("kl_divergence")
            result["refusals"] = (
                f"{float(refusal_rate) * 100:.1f}%" if refusal_rate is not None else "—"
            )
            result["kl_divergence"] = (
                f"{float(kl_divergence):.4f}" if kl_divergence is not None else "—"
            )
            results.append(result)
            if len(results) >= limit:
                break

        recommended_index = -1
        if results and self._trial_results:
            best_step = min(
                self._trial_results,
                key=lambda s: (
                    0.0
                    if (
                        self._trial_results.get(s, {}).get("refusal_rate") is not None
                        and float(self._trial_results.get(s, {}).get("refusal_rate", 1)) < 0.10
                    )
                    else 1.0,
                    float(self._trial_results.get(s, {}).get("kl_divergence") or float("inf")),
                ),
            )
            for index, trial in enumerate(results):
                if int(trial.get("trial", 0)) - 1 == best_step:
                    recommended_index = index
                    break
            if recommended_index < 0 and results:
                recommended_index = 0

        return results, recommended_index

    def _set_stage_from_text(self, text: str) -> None:
        lowered = text.lower()
        if "loading model" in lowered or "loading checkpoint" in lowered:
            self._set_stage("Loading model")
            self.model_loading_row.set_value(text)
        elif "response prefix" in lowered:
            self._set_stage("Preparing prompts")
            self.prompt_prep_row.set_value(text)
        elif "batch size" in lowered:
            self._set_stage("Preparing batches")
            self.prompt_prep_row.set_value(text)
        elif re.search(r"running trial \d+ of \d+", lowered):
            self._set_stage(text)
        elif "evaluat" in lowered:
            self._set_stage("Evaluating")
        elif "upload" in lowered:
            self._set_stage("Uploading model")
            self.export_status_row.set_value(text)
        elif "saving" in lowered or "exporting" in lowered or "export" in lowered:
            self._set_stage("Exporting model")
            self.export_status_row.set_value(text)
        elif "benchmark" in lowered:
            self._set_stage("Benchmarking")

    def _set_stage(self, text: str) -> None:
        self.stage_row.set_value(text)

    def _set_error_status(self, text: str) -> None:
        self.error_status_row.set_value(text)

    def _set_worker_pill(self, text: str, state: str) -> None:
        self.worker_pill.set_state(state, text)

    def _set_worker_pill_for_state(self, state: JobState) -> None:
        mapping = {
            JobState.QUEUED: ("Worker · queued", "info"),
            JobState.STARTING: ("Worker · starting", "info"),
            JobState.RUNNING: ("Worker · running", "ok"),
            JobState.CANCELLING: ("Worker · cancelling", "warn"),
            JobState.SUCCEEDED: ("Worker · completed", "ok"),
            JobState.FAILED: ("Worker · failed", "error"),
            JobState.CANCELLED: ("Worker · cancelled", "warn"),
        }
        text, level = mapping.get(state, ("Worker · idle", "off"))
        self._set_worker_pill(text, level)

    def _update_elapsed(self) -> None:
        if self._run_started_monotonic is None:
            return
        self.elapsed_row.set_value(_format_elapsed(time.monotonic() - self._run_started_monotonic))

    def _verify_output(self) -> list[str]:
        if self._pending_model_action != "save":
            return []
        if self._pending_artifacts is not None:
            if not self._pending_artifacts:
                return ["Heretic reported success but produced no output files."]
            return []
        if self._pending_output_dir:
            path = Path(self._pending_output_dir)
            if not path.is_dir() or not any(path.iterdir()):
                return [f"Expected output was not found in {path}."]
        return []

    def _on_worker_exited(self, exit_code: int) -> None:
        self._run_timer.stop()
        self._run_started_monotonic = None
        self._left_content.setEnabled(True)
        if self._active_job_id:
            job_id = self._active_job_id
            snapshot = self._job_manager.snapshot(job_id)
            completion = self._pending_completion
            if snapshot is not None and snapshot.state.value == "cancelling":
                self._job_manager.handle_worker_exit(job_id, exit_code)
                self._set_worker_pill("Worker · cancelled", "warn")
                self._set_stage("Cancelled")
                self.console.append(
                    f"Job {job_id} stopped after cancellation (exit code {exit_code}).",
                    level="warning",
                    source="worker exit",
                )
            elif completion is not None and completion[0] and exit_code == 0:
                problems = self._verify_output()
                if problems:
                    self._job_manager.mark_failed(
                        job_id,
                        message=problems[0],
                        exit_code=exit_code,
                    )
                    self._set_worker_pill("Worker · failed", "error")
                    self._set_stage("Failed")
                    self._set_error_status(problems[0])
                    self.export_status_row.set_value(f"Failed: {problems[0]}")
                    self.console.append(problems[0], level="error", source="verification")
                    self._state.notify("error", problems[0])
                else:
                    # Show trial selection dialog if we have multiple completed trials
                    results, recommended_index = self._trial_ranked_results(limit=3)
                    if len(results) > 1 and not self._awaiting_trial_selection:
                        self._awaiting_trial_selection = True
                        selected = self._show_trial_selection_dialog(results, recommended_index)
                        if selected is not None:
                            # User selected a trial; re-run Heretic with that trial index
                            self.console.append(
                                f"Re-running trial {selected}...",
                                level="info",
                                source="trial selection",
                            )
                            try:
                                selected_config = self.config_from_form()
                            except HereticConfigError as error:
                                self._job_manager.mark_succeeded(job_id, message=completion[1])
                                self._fail_start(
                                    f"Cannot re-run selected trial: {error}", notify=False
                                )
                            else:
                                selected_config.model = self._selected_model()
                                problems = selected_config.validate()
                                if problems:
                                    self._job_manager.mark_succeeded(job_id, message=completion[1])
                                    self.toml_status.setText("Issues: " + "; ".join(problems))
                                    self._fail_start(
                                        f"Cannot re-run selected trial: {problems[0]}",
                                        notify=False,
                                    )
                                else:
                                    engine_status = self._registry.status(HERETIC)
                                    interpreter = engine_status.interpreter
                                    if interpreter is None or not interpreter.is_file():
                                        self._job_manager.mark_succeeded(
                                            job_id, message=completion[1]
                                        )
                                        self._fail_start(
                                            "Cannot re-run selected trial: Heretic interpreter "
                                            "is no longer available.",
                                            notify=False,
                                        )
                                    else:
                                        # Release the completed run's GPU ownership before
                                        # submitting the selected-trial run.
                                        self._job_manager.mark_succeeded(
                                            job_id,
                                            message="Completed; selected trial queued to re-run.",
                                        )
                                        selected_config.trial_index = selected - 1
                                        rerun = self._job_manager.submit(
                                            HERETIC, model_ref=selected_config.model
                                        )
                                        rerun_config = self._build_config(
                                            rerun.job_id,
                                            selected_config.model,
                                            interpreter,
                                            trial_index=selected_config.trial_index,
                                        )
                                        worker_script = (
                                            Path(__file__).resolve().parents[1]
                                            / "workers"
                                            / "heretic_worker.py"
                                        )
                                        if rerun.state is JobState.QUEUED:
                                            self._queued_jobs[rerun.job_id] = {
                                                "config": rerun_config,
                                                "interpreter": interpreter,
                                                "script": worker_script,
                                            }
                                            self._connect_queue_signal()
                                            self.console.append(
                                                f"Selected trial re-run {rerun.job_id} queued; "
                                                "it starts when the GPU is free.",
                                                level="system",
                                                source="queue",
                                            )
                                            self._set_worker_pill("Worker · queued", "info")
                                        else:
                                            # The current worker's exit handler still has cleanup
                                            # to perform. Start the next process on the next event
                                            # loop turn so that cleanup cannot clear its state.
                                            QTimer.singleShot(
                                                0,
                                                lambda job_id=rerun.job_id, config=rerun_config, script=worker_script, python=interpreter: (
                                                    self._launch_job(job_id, config, python, script)
                                                ),
                                            )
                    else:
                        self._job_manager.mark_succeeded(job_id, message=completion[1])
                        self._set_worker_pill("Worker · completed", "ok")
                        self._set_stage("Complete")
                        self.console.append(
                            completion[1],
                            level="info",
                            source="worker exit",
                        )
                        self._state.publish_models(self._model_service.scan_local_models())
                        self._state.notify("info", "Heretic modification completed successfully!")
            elif completion is not None:
                self._job_manager.mark_failed(
                    job_id,
                    message=completion[1],
                    exit_code=exit_code,
                )
                self._set_worker_pill("Worker · failed", "error")
                self._set_stage("Failed")
                self.console.append(
                    f"{completion[1]} (exit code {exit_code})",
                    level="error",
                    source="worker exit",
                )
            else:
                self._job_manager.handle_worker_exit(job_id, exit_code)
                self._set_worker_pill("Worker · stopped", "warn")
                self.console.append(
                    f"Worker exited with code {exit_code} without a completion message.",
                    level="warning",
                    source="worker exit",
                )
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self._refresh_run_button()
        self.cancel_button.setEnabled(False)

    def _on_worker_failed_to_start(self, error: str) -> None:
        if self._active_job_id:
            self._job_manager.mark_failed(self._active_job_id, message=f"Failed to start: {error}")
            self._active_job_id = None
            self._active_worker = None
            self._pending_completion = None
        self._run_timer.stop()
        self._run_started_monotonic = None
        self._left_content.setEnabled(True)
        self._refresh_run_button()
        self.cancel_button.setEnabled(False)
        self._set_worker_pill("Worker · failed", "error")
        self._set_error_status(f"Failed to start: {error}")
        self.console.append(
            f"Heretic worker failed to start: {error}",
            level="error",
            source="worker",
        )
        self._state.notify("error", f"Heretic worker failed to start: {error}")

    def _open_output_location(self) -> None:
        target = self._pending_output_dir
        if not target:
            return
        path = Path(target)
        if not path.is_dir():
            self._state.notify("warning", f"Output folder does not exist: {path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._state.notify("warning", f"Could not open {path} in Explorer.")


def _MODIFIER_LABELS_ORDER() -> tuple[str, ...]:
    return tuple(MODIFIER_LABELS)


def _SCORER_ORDER() -> tuple[str, ...]:
    return tuple(SCORER_LABELS)

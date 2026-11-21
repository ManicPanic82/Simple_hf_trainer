"""Heretic tab: full Heretic 2.x configuration, run monitoring and hand-off."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
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
    QUANTIZATIONS,
    REPRODUCIBILITY,
    SCORER_LABELS,
    HereticConfig,
    HereticConfigError,
    PluginEntry,
    available_presets,
    load_preset,
)
from services.model_service import LocalModelRecord, ModelService
from services.storage_service import Severity, StorageService
from ui.components import Card, FieldRow, StoragePathRow
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


def _dataset_text(spec: dict) -> str:
    return str(spec.get("dataset", "") or "")


def _dataset_split(spec: dict) -> str:
    return str(spec.get("split", "") or "")


def _dataset_column(spec: dict) -> str:
    return str(spec.get("column", "") or "")


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
            "Modify a compatible model with the real Heretic 2.x workflow. Every engine "
            "setting is editable below; the original model files are left untouched.",
            parent,
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

        # --- Engine Status Card ---
        self.engine_card = EngineCard(
            "Heretic engine",
            "Heretic runs in its own Python environment and uses a separate worker process. "
            "It never shares the GPU with Unsloth.",
            self,
        )
        self.add(self.engine_card)

        # --- Model and setup card ---
        setup = Card(
            "Model and run profile",
            "Pick a model, choose a starting configuration, then refine any setting in the "
            "configuration tabs below.",
        )
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(QLabel("Target Model:", setup), 0, 0)
        self.model_combo = QComboBox(setup)
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText(
            "Select local model or enter HF repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
        )
        self.model_combo.addItem("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
        self.model_combo.setCurrentIndex(0)
        grid.addWidget(self.model_combo, 0, 1, 1, 3)

        grid.addWidget(QLabel("Preset:", setup), 1, 0)
        self.preset_combo = QComboBox(setup)
        for preset_id, label in available_presets():
            self.preset_combo.addItem(label, preset_id)
        self._select_preset("compact")
        grid.addWidget(self.preset_combo, 1, 1)
        self.load_preset_button = QPushButton("Load preset", setup)
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        grid.addWidget(self.load_preset_button, 1, 2)
        grid.addWidget(
            QLabel("Presets are upstream Heretic configs; edit anything after loading."), 1, 3
        )
        setup.add_layout(grid)

        self.outputs_path = StoragePathRow("outputs_dir", "Output folder", setup)
        self.outputs_path.open_requested.connect(self._open_outputs)
        setup.add(self.outputs_path)
        self.add(setup)

        # --- Full configuration card ---
        config_card = Card(
            "Heretic 2.x configuration",
            "These are the real Heretic config.toml settings. The Raw TOML tab can edit "
            "anything not represented by a widget, including custom scorer/modifier plugins.",
        )
        self.config_tabs = QTabWidget(config_card)
        self._build_model_tab()
        self._build_objectives_tab()
        self._build_optimization_tab()
        self._build_export_tab()
        self._build_toml_tab()
        config_card.add(self.config_tabs)
        self.add(config_card)

        # --- Control & live progress card ---
        control = Card(
            "Run control & live monitoring",
            "Heretic benchmarks the machine, optimizes the configured scores with Optuna, "
            "and reports each trial while it works.",
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
        self.extra_metric_label = QLabel("", control)
        self.extra_metric_label.setObjectName("StripValue")
        for label in (
            self.trial_metric_label,
            self.refusal_metric_label,
            self.kl_metric_label,
            self.extra_metric_label,
        ):
            metrics_layout.addWidget(label)
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

        log_label = QLabel("Engine Console", control)
        log_label.setObjectName("SectionLabel")
        control.add(log_label)
        self.console = ConsoleWidget(parent=control)
        control.add(self.console)
        self.last_model_row = FieldRow("Saved model", "None saved yet", caption_width=104)
        control.add(self.last_model_row)
        self.add(control)

        # --- Information Card ---
        notes = Card(
            "What to expect",
            "A preset with many trials can take hours. The compact preset uses 4-bit "
            "quantization, two trials and short prompt slices to fit a 6 GB GPU.",
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
        self.apply_config_to_form(self._config)
        self.refresh()

        self._state.job_changed.connect(self.apply_job)
        self._state.engines_changed.connect(self.apply_engines)
        self._state.models_changed.connect(self._update_model_choices)

    # -- form construction ------------------------------------------------

    def _build_model_tab(self) -> None:
        tab = QWidget(self.config_tabs)
        grid = QGridLayout(tab)
        grid.setSpacing(8)
        row = 0

        def add(label: str, widget: QWidget) -> None:
            nonlocal row
            grid.addWidget(QLabel(label, tab), row, 0)
            grid.addWidget(widget, row, 1, 1, 2)
            row += 1

        self.dtypes_edit = QLineEdit(tab)
        self.dtypes_edit.setPlaceholderText("auto, float16, bfloat16, float32")
        add("Dtypes to try:", self.dtypes_edit)

        self.quant_combo = QComboBox(tab)
        self.quant_combo.addItems(list(QUANTIZATIONS))
        add("Quantization:", self.quant_combo)

        self.device_map_edit = QLineEdit(tab)
        add("Device map:", self.device_map_edit)

        self.max_memory_edit = QLineEdit(tab)
        self.max_memory_edit.setPlaceholderText(
            'blank = automatic; e.g. {"0": "20GB", "cpu": "64GB"}'
        )
        add("Max memory (JSON):", self.max_memory_edit)

        self.offload_check = QCheckBox("Move analysis tensors to CPU", tab)
        add("Memory offload:", self.offload_check)

        self.batch_spin = QSpinBox(tab)
        self.batch_spin.setRange(0, 4096)
        self.batch_spin.setToolTip("0 lets Heretic benchmark an optimal batch size.")
        add("Batch size (0 = auto):", self.batch_spin)

        self.max_batch_spin = QSpinBox(tab)
        self.max_batch_spin.setRange(1, 4096)
        add("Max batch size:", self.max_batch_spin)

        self.max_response_spin = QSpinBox(tab)
        self.max_response_spin.setRange(1, 8192)
        add("Max response tokens:", self.max_response_spin)

        self.response_prefix_edit = QLineEdit(tab)
        self.response_prefix_edit.setPlaceholderText("blank = detect automatically")
        add("Response prefix:", self.response_prefix_edit)

        self.system_prompt_edit = QLineEdit(tab)
        add("System prompt:", self.system_prompt_edit)

        self.cot_edit = QTextEdit(tab)
        self.cot_edit.setMaximumHeight(64)
        self.cot_edit.setPlaceholderText("One 'initializer | finalizer' pair per line")
        add("CoT blocks:", self.cot_edit)

        self.debug_check = QCheckBox("Print torch/debug information", tab)
        add("Debug output:", self.debug_check)

        grid.setRowStretch(row, 1)
        self.config_tabs.addTab(tab, "Model & loading")

    def _build_objectives_tab(self) -> None:
        tab = QWidget(self.config_tabs)
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        modifier_row = QHBoxLayout()
        modifier_row.addWidget(QLabel("Modifier:", tab))
        self.modifier_combo = QComboBox(tab)
        self.modifier_combo.addItems([*_MODIFIER_LABELS_ORDER(), _CUSTOM])
        self.modifier_combo.currentTextChanged.connect(self._on_modifier_changed)
        modifier_row.addWidget(self.modifier_combo, 1)
        self.modifier_plugin_edit = QLineEdit(tab)
        self.modifier_plugin_edit.setPlaceholderText("Custom plugin import path or file.py:Class")
        self.modifier_plugin_edit.setEnabled(False)
        modifier_row.addWidget(self.modifier_plugin_edit, 2)
        layout.addLayout(modifier_row)

        layout.addWidget(QLabel("Scorers (co-optimized):", tab))
        self.scorers_layout = QVBoxLayout()
        layout.addLayout(self.scorers_layout)
        add_scorer = QPushButton("Add scorer", tab)
        add_scorer.clicked.connect(lambda: self._add_scorer_row())
        layout.addWidget(add_scorer)

        layout.addWidget(QLabel("Modifier prompt sources (for the selected modifier):", tab))
        self._prompt_fields = QGridLayout()
        self._prompt_fields.setSpacing(6)
        self.good_dataset_edit = QLineEdit(tab)
        self.good_split_edit = QLineEdit(tab)
        self.good_column_edit = QLineEdit(tab)
        self.bad_dataset_edit = QLineEdit(tab)
        self.bad_split_edit = QLineEdit(tab)
        self.bad_column_edit = QLineEdit(tab)
        self.kw_dataset_edit = QLineEdit(tab)
        self.kw_split_edit = QLineEdit(tab)
        self.kw_column_edit = QLineEdit(tab)
        self.kl_dataset_edit = QLineEdit(tab)
        self.kl_split_edit = QLineEdit(tab)
        self.kl_column_edit = QLineEdit(tab)
        sources = (
            ("Good prompts", self.good_dataset_edit, self.good_split_edit, self.good_column_edit),
            ("Bad prompts", self.bad_dataset_edit, self.bad_split_edit, self.bad_column_edit),
            ("Keyword scorer", self.kw_dataset_edit, self.kw_split_edit, self.kw_column_edit),
            ("KL scorer", self.kl_dataset_edit, self.kl_split_edit, self.kl_column_edit),
        )
        for index, (name, dataset, split, column) in enumerate(sources, start=1):
            self._prompt_fields.addWidget(QLabel(name, tab), index, 0)
            dataset.setPlaceholderText("dataset id or path")
            split.setPlaceholderText("split")
            column.setPlaceholderText("column")
            self._prompt_fields.addWidget(dataset, index, 1, 1, 2)
            self._prompt_fields.addWidget(split, index, 3)
            self._prompt_fields.addWidget(column, index, 4)
        layout.addLayout(self._prompt_fields)
        layout.addWidget(
            QLabel(
                "Scorer-specific options (marker lists, classifier model, benchmark task, "
                "prefixes) are edited in the Raw TOML tab.",
                tab,
            )
        )
        layout.addStretch(1)
        self.config_tabs.addTab(tab, "Objectives")

    def _build_optimization_tab(self) -> None:
        tab = QWidget(self.config_tabs)
        grid = QGridLayout(tab)
        grid.setSpacing(8)
        row = 0

        def add(label: str, widget: QWidget) -> None:
            nonlocal row
            grid.addWidget(QLabel(label, tab), row, 0)
            grid.addWidget(widget, row, 1, 1, 2)
            row += 1

        self.trials_spin = QSpinBox(tab)
        self.trials_spin.setRange(1, 10_000)
        add("Optimization trials:", self.trials_spin)

        self.startup_spin = QSpinBox(tab)
        self.startup_spin.setRange(0, 10_000)
        add("Startup (random) trials:", self.startup_spin)

        self.seed_edit = QLineEdit(tab)
        self.seed_edit.setPlaceholderText("blank = random seed")
        add("Seed:", self.seed_edit)

        self.checkpoint_combo = QComboBox(tab)
        self.checkpoint_combo.addItems(["Restart from scratch", "Continue previous run"])
        add("Existing checkpoint:", self.checkpoint_combo)

        self.trial_index_spin = QSpinBox(tab)
        self.trial_index_spin.setRange(0, 9999)
        self.trial_index_spin.setToolTip("Index in the sorted Pareto front to export.")
        add("Trial to export:", self.trial_index_spin)

        self.checkpoint_label = QLabel("(set by the output folder)", tab)
        add("Study checkpoint dir:", self.checkpoint_label)
        grid.setRowStretch(row, 1)
        self.config_tabs.addTab(tab, "Optimization")

    def _build_export_tab(self) -> None:
        tab = QWidget(self.config_tabs)
        grid = QGridLayout(tab)
        grid.setSpacing(8)
        row = 0

        def add(label: str, widget: QWidget) -> None:
            nonlocal row
            grid.addWidget(QLabel(label, tab), row, 0)
            grid.addWidget(widget, row, 1, 1, 2)
            row += 1

        self.export_combo = QComboBox(tab)
        self.export_combo.addItems(list(EXPORT_STRATEGIES))
        add("Export strategy:", self.export_combo)

        self.model_action_combo = QComboBox(tab)
        self.model_action_combo.addItems(list(MODEL_ACTIONS))
        add("After optimization:", self.model_action_combo)

        self.upload_repo_edit = QLineEdit(tab)
        self.upload_repo_edit.setPlaceholderText("username/model-heretic")
        add("HF repository:", self.upload_repo_edit)

        self.private_check = QCheckBox("Create a private repository", tab)
        add("Upload visibility:", self.private_check)

        self.repro_combo = QComboBox(tab)
        self.repro_combo.addItems(["(not set)", *REPRODUCIBILITY])
        add("Reproducibility info:", self.repro_combo)

        self.modified_type_edit = QLineEdit(tab)
        add("Modified model type:", self.modified_type_edit)

        self.shard_edit = QLineEdit(tab)
        add("Max shard size:", self.shard_edit)

        self.tags_edit = QLineEdit(tab)
        self.tags_edit.setPlaceholderText("comma-separated tags")
        add("Upload tags:", self.tags_edit)

        self.shell_check = QCheckBox("Offer in-context Python shell (unrestricted)", tab)
        add("Advanced:", self.shell_check)
        grid.setRowStretch(row, 1)
        self.config_tabs.addTab(tab, "Export")

    def _build_toml_tab(self) -> None:
        tab = QWidget(self.config_tabs)
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        self.toml_view = QTextEdit(tab)
        self.toml_view.setStyleSheet(
            "font-family: 'Cascadia Mono', Consolas, monospace; font-size: 8.5pt;"
        )
        layout.addWidget(self.toml_view, 1)

        buttons = QHBoxLayout()
        refresh = QPushButton("Refresh from form", tab)
        refresh.clicked.connect(self._refresh_toml_view)
        apply_toml = QPushButton("Apply TOML to form", tab)
        apply_toml.setObjectName("Primary")
        apply_toml.clicked.connect(self._apply_toml_to_form)
        validate = QPushButton("Validate", tab)
        validate.clicked.connect(self._validate_config)
        open_file = QPushButton("Open…", tab)
        open_file.clicked.connect(self._open_config_file)
        save_file = QPushButton("Save…", tab)
        save_file.clicked.connect(self._save_config_file)
        for button in (refresh, apply_toml, validate, open_file, save_file):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.toml_status = QLabel("", tab)
        self.toml_status.setWordWrap(True)
        layout.addWidget(self.toml_status)
        self.config_tabs.addTab(tab, "Raw TOML")

    # -- form <-> config --------------------------------------------------

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
        cfg.device_map = self.device_map_edit.text().strip() or "auto"
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
        cfg.model = self.model_combo.currentData() or self.model_combo.currentText().strip()
        return cfg

    def apply_config_to_form(self, cfg: HereticConfig) -> None:
        self._config = copy.deepcopy(cfg)
        self.dtypes_edit.setText(", ".join(cfg.dtypes))
        self.quant_combo.setCurrentText(cfg.quantization)
        self.device_map_edit.setText(cfg.device_map)
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

    # -- raw TOML actions -------------------------------------------------

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

    # -- setup / state ----------------------------------------------------

    def _open_outputs(self, label: str) -> None:
        open_storage_path(self._state, self._storage, label)

    def select_model(self, model_ref: str) -> None:
        idx = self.model_combo.findText(model_ref)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setEditText(model_ref)

    def _select_preset(self, preset_id: str) -> None:
        index = self.preset_combo.findData(preset_id)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)

    def _load_selected_preset(self) -> None:
        preset_id = self.preset_combo.currentData()
        if not preset_id:
            return
        try:
            cfg = load_preset(preset_id)
        except (HereticConfigError, OSError) as error:
            self._state.notify("error", f"Could not load Heretic preset: {error}")
            return
        self.apply_config_to_form(cfg)
        self._state.notify("info", f"Loaded Heretic preset: {self.preset_combo.currentText()}")

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
            self.model_combo.setCurrentIndex(0)

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
            if self._state.hardware is None:
                self.run_button.setToolTip(
                    "Run Heretic — GPU information unavailable; model compatibility is not verified."
                )
            else:
                self.run_button.setToolTip(
                    "Run Heretic — verify the selected model and settings fit in VRAM."
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

    # -- launch / protocol ------------------------------------------------

    def _start_heretic(self) -> None:
        model_name = self.model_combo.currentData() or self.model_combo.currentText().strip()
        if not model_name:
            self._state.notify("warning", "Please select or enter a model name to modify.")
            return

        try:
            cfg = self.config_from_form()
        except HereticConfigError as error:
            self._state.notify("error", f"Invalid Heretic configuration: {error}")
            return
        cfg.model = model_name
        problems = cfg.validate()
        if problems:
            self.toml_status.setText("Issues: " + "; ".join(problems))
            self._state.notify("error", "Heretic configuration is invalid: " + problems[0])
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

        config_data = self._build_config(job.job_id, model_name)
        if job.state is JobState.QUEUED:
            self._queued_jobs[job.job_id] = {
                "config": config_data,
                "interpreter": interpreter,
                "script": worker_script,
            }
            self._connect_queue_signal()
            self.console.append(
                f"[queue] Job {job.job_id} queued; it starts when the GPU is free."
            )
            self._state.notify("info", f"Heretic job {job.job_id} queued.")
            return

        self._launch_job(job.job_id, config_data, interpreter, worker_script)

    def _build_config(self, job_id: str, model_name: str) -> dict:
        outputs_dir = self._storage.path_for("outputs_dir") / f"heretic_{job_id}"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = self.config_from_form()
        except HereticConfigError:
            cfg = copy.deepcopy(self._config)
        cfg.model = model_name
        headless = cfg.headless_copy(outputs_dir / "model", outputs_dir / "checkpoints")
        return {
            "job_id": job_id,
            "model_name": model_name,
            "trials": headless.n_trials,
            "export_strategy": headless.export_strategy,
            "output_dir": str(outputs_dir / "model"),
            "toml": headless.to_toml(),
        }

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
        self.console.append(f"[queue] Job {snapshot.job_id} is starting now.")
        self._launch_job(snapshot.job_id, spec["config"], spec["interpreter"], spec["script"])

    def _launch_job(self, job_id: str, config_data: dict, interpreter, worker_script) -> None:
        try:
            self._active_job_id = job_id
            self._pending_completion = None
            self.run_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.console.start_session("Heretic modification", model_name)
            self.trial_metric_label.setText("Trial: 0")
            self.refusal_metric_label.setText("Refusal: —")
            self.kl_metric_label.setText("KL Div: —")
            self.extra_metric_label.setText("")

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
        except Exception as error:
            if self._active_worker is not None and self._active_worker.running:
                self._state.notify(
                    "error",
                    f"Heretic worker started but setup reported an error: {error}",
                )
            else:
                self._handle_setup_failure(job_id, error)

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
            self.console.append(f"[engine stderr] {line}")

    def _on_worker_protocol_error(self, error: object) -> None:
        self.console.append(f"[protocol warning] {error}")

    def _on_worker_message(self, msg: WorkerMessage) -> None:
        payload = msg.payload
        if msg.type is MessageType.HELLO:
            pid = payload.get("pid")
            detail = f" (PID {pid})" if pid else ""
            self.console.append(f"[worker] Started{detail}.")

        elif msg.type is MessageType.READY:
            engine = payload.get("engine", "Heretic engine")
            version = payload.get("version")
            detail = f" {version}" if version else ""
            self.console.append(f"[worker] {engine} ready{detail}.")

        elif msg.type is MessageType.ERROR:
            message = str(payload.get("message") or "Unknown worker error.")
            self.console.append(f"[engine error] {message}")

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
            if val is None:
                return
            if name == "refusal_rate":
                self.refusal_metric_label.setText(f"Refusal: {float(val) * 100:.1f}%")
            elif name == "kl_divergence":
                self.kl_metric_label.setText(f"KL Div: {float(val):.4f}")
            else:
                self.extra_metric_label.setText(f"{name}: {float(val):.4f}")

        elif msg.type is MessageType.LOG:
            self.console.append(str(payload.get("message", "")))

        elif msg.type is MessageType.RESULT:
            out_dir = payload.get("output_dir", "")
            self.last_model_row.set_value(out_dir)

        elif msg.type is MessageType.JOB_COMPLETE:
            success = bool(payload.get("success", False))
            message = str(
                payload.get("message")
                or (
                    "Heretic modification completed." if success else "Heretic finished with error."
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
                self._state.notify("info", "Heretic modification completed successfully!")
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


def _MODIFIER_LABELS_ORDER() -> tuple[str, ...]:
    return tuple(MODIFIER_LABELS)


def _SCORER_ORDER() -> tuple[str, ...]:
    return tuple(SCORER_LABELS)

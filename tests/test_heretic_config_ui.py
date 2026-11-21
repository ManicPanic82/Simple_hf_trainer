"""Heretic tab: presets, Raw TOML editing and headless config generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.credentials import MemoryCredentialStore
from services.heretic_config import HereticConfig
from ui.heretic_tab import HereticTab


def _tab(tmp_path: Path, *, credentials=None) -> HereticTab:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    storage = MagicMock()
    storage.validate.return_value = SimpleNamespace(path=outputs, issues=[], ok=True, exists=True)
    storage.path_for.return_value = outputs
    state = MagicMock()
    state.engines = {}
    state.models = []
    state.hardware = None
    return HereticTab(
        state,
        storage,
        job_manager=MagicMock(),
        registry=MagicMock(),
        model_service=MagicMock(),
        credentials=credentials,
    )


def test_loading_upstream_preset_updates_widgets(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab.preset_combo.setCurrentIndex(tab.preset_combo.findData("noslop"))
    tab.load_preset_button.click()

    assert tab.max_response_spin.value() == 300
    assert "no-slop" in tab.tags_edit.text()
    assert tab.modifier_combo.currentText().startswith("ARA")
    assert "writing-prompts" in tab.good_dataset_edit.text()
    assert tab._validate_config() == []


def test_raw_toml_apply_updates_form_and_headless_config(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    text = tab.toml_view.toPlainText().replace("n_trials = 2", "n_trials = 7")
    assert text != tab.toml_view.toPlainText()
    tab.toml_view.setPlainText(text)
    tab._apply_toml_to_form()

    assert tab.trials_spin.value() == 7
    assert "Configuration valid" in tab.toml_status.text()

    built = tab._build_config("heretic-9", "org/model")
    cfg = HereticConfig.from_toml(built["toml"])
    assert cfg.model == "org/model"
    assert cfg.n_trials == 7
    assert cfg.model_action == "save"
    assert cfg.export_strategy == "merge"
    assert cfg.trial_index == 0
    job_dir = tmp_path / "outputs" / "heretic_heretic-9"
    assert Path(cfg.save_directory) == job_dir / "model"
    assert Path(cfg.study_checkpoint_dir) == job_dir / "checkpoints"


def test_invalid_toml_is_reported_without_crashing(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab.toml_view.setPlainText("n_trials = [")
    tab._apply_toml_to_form()
    assert "Invalid TOML" in tab.toml_status.text()
    assert tab.trials_spin.value() == 2


def test_upload_config_uses_stored_token(qapp, tmp_path: Path) -> None:
    credentials = MemoryCredentialStore("hf_secret")
    tab = _tab(tmp_path, credentials=credentials)
    tab.model_action_combo.setCurrentText("upload")
    tab.upload_repo_edit.setText("user/model-heretic")

    built = tab._build_config("heretic-upload", "org/model")
    assert tab._worker_environment(built) == {"HF_TOKEN": "hf_secret"}

    tab.model_action_combo.setCurrentText("save")
    built = tab._build_config("heretic-save", "org/model")
    assert tab._worker_environment(built) == {}


def test_validation_blocks_a_run_without_objectives(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    for row in list(tab._scorer_rows):
        tab._remove_scorer_row(row, row["plugin"].parentWidget())
    tab._add_scorer_row()
    tab._scorer_rows[0]["optimization"].setCurrentText("none")
    tab.model_combo.setEditText("org/model")
    tab._start_heretic()
    problems = tab.toml_status.text()
    assert "at least one scorer" in problems

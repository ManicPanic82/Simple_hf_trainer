"""Heretic tab: presets, Raw TOML editing and headless config generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.credentials import MemoryCredentialStore
from services.heretic_config import HereticConfig, HereticConfigError
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


def test_device_map_table_is_editable_in_the_form(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab.device_map_edit.setText('{"0": 0, "cpu": "disk"}')

    cfg = tab.config_from_form()
    assert cfg.device_map == {"0": 0, "cpu": "disk"}
    assert HereticConfig.from_toml(cfg.to_toml()).device_map == {
        "0": 0,
        "cpu": "disk",
    }

    # Round-tripping through the form must preserve the table.
    tab.apply_config_to_form(cfg)
    assert tab.device_map_edit.text() == '{"0": 0, "cpu": "disk"}'

    tab.device_map_edit.setText('{"0": }')
    with pytest.raises(HereticConfigError):
        tab.config_from_form()


def test_profile_origin_distinguishes_upstream_and_bundled(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab.preset_combo.setCurrentIndex(tab.preset_combo.findData("noslop"))
    assert "Upstream Heretic preset" in tab.profile_origin_label.text()

    tab.preset_combo.setCurrentIndex(tab.preset_combo.findData("compact"))
    assert "Built-in low-VRAM profile" in tab.profile_origin_label.text()


def test_build_config_preserves_selected_trial_index(qapp, tmp_path: Path) -> None:
    tab = _tab(tmp_path)
    tab.model_combo.setEditText("org/model")
    config = tab._build_config("heretic-selected-trial", "org/model", trial_index=2)

    parsed = HereticConfig.from_toml(config["toml"])
    assert parsed.model == "org/model"
    assert parsed.trial_index == 2


def test_trial_selection_dialog_defaults_to_recommended_trial(qapp) -> None:
    from ui.heretic_tab import HereticTrialSelectionDialog

    dialog = HereticTrialSelectionDialog(
        results=[
            {"trial": 1, "refusals": "20%", "kl_divergence": "0.02"},
            {"trial": 2, "refusals": "5%", "kl_divergence": "0.01"},
        ],
        recommended_index=1,
    )

    assert dialog.selected_trial() == 1


def test_selected_trial_rerun_releases_gpu_and_launches_after_exit_cleanup(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from PySide6.QtCore import QTimer

    from services.engine_registry import HERETIC
    from workers.job_manager import JobState

    tab = _tab(tmp_path)
    tab.model_combo.setEditText("org/model")
    tab._active_job_id = "heretic-1"
    tab._pending_completion = (True, "initial run complete")
    tab._trial_ranked_results = lambda limit=3: ([{"trial": 1}, {"trial": 2}], 1)
    tab._show_trial_selection_dialog = lambda results, recommended_index: 2
    tab._verify_output = lambda: []
    tab._config.model = "org/model"

    queued = SimpleNamespace(job_id="heretic-2", state=JobState.STARTING)
    tab._job_manager.submit.return_value = queued
    tab._build_config = MagicMock(return_value={"job_id": "heretic-2"})
    tab._launch_job = MagicMock()
    scheduled: list[object] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda delay, callback: scheduled.append(callback))

    tab._on_worker_exited(0)

    tab._job_manager.mark_succeeded.assert_called_once_with(
        "heretic-1", message="Completed; selected trial queued to re-run."
    )
    tab._job_manager.submit.assert_called_once_with(HERETIC, model_ref="org/model")
    tab._build_config.assert_called_once_with(
        "heretic-2", "org/model", None, trial_index=1
    )
    assert tab._active_job_id is None
    assert len(scheduled) == 1
    scheduled[0]()
    tab._launch_job.assert_called_once_with(
        "heretic-2",
        {"job_id": "heretic-2"},
        None,
        Path(__file__).resolve().parents[1] / "src" / "workers" / "heretic_worker.py",
    )

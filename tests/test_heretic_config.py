"""Heretic 2.x config model: round trips, presets and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.heretic_config import (
    MODIFIER_ABLITERATION,
    SCORER_CLASSIFIER,
    DatasetSpec,
    HereticConfig,
    HereticConfigError,
    PluginEntry,
    available_presets,
    load_preset,
)


def test_full_config_round_trips_exactly() -> None:
    for cfg in (HereticConfig.default(), HereticConfig.compact()):
        text = cfg.to_toml()
        parsed = HereticConfig.from_toml(text)
        assert parsed.to_toml() == text
        assert parsed == cfg


def test_bundled_presets_parse_and_are_complete() -> None:
    presets = dict(available_presets())
    assert {"compact", "default", "noslop", "emotions", "terse"} <= presets.keys()
    for preset_id in presets:
        cfg = load_preset(preset_id)
        assert cfg.modifiers
        assert any(entry.optimization in ("minimize", "maximize") for entry in cfg.scorers)
        assert HereticConfig.from_toml(cfg.to_toml()).to_toml() == cfg.to_toml()
        # Only the model is supplied by the GUI at run time.
        assert [problem for problem in cfg.validate() if problem != "model is required"] == []


def test_partial_toml_is_layered_over_upstream_defaults() -> None:
    cfg = HereticConfig.from_toml(
        'n_trials = 3\n[scorer.KeywordRate]\nscore_name = "Cliches"\n',
        base=HereticConfig.default(),
    )
    assert cfg.n_trials == 3
    assert cfg.plugin_tables["scorer"]["KeywordRate"]["score_name"] == "Cliches"
    assert cfg.plugin_tables["scorer"]["KLDivergence"]["prompts"]["dataset"]


def test_multi_instance_classifier_preserves_plugin_tables() -> None:
    cfg = load_preset("emotions")
    assert len(cfg.scorers) == 3
    tables = cfg.plugin_tables["scorer"]
    assert "BERTClassifier_negative_emotions" in tables
    assert "BERTClassifier_positive_emotions" in tables
    assert tables["BERTClassifier_positive_emotions"]["match_labels"] == ["joy", "surprise"]
    assert HereticConfig.from_toml(cfg.to_toml()).plugin_tables == cfg.plugin_tables


def test_validation_reports_unrunnable_configurations() -> None:
    cfg = HereticConfig.compact()
    cfg.model = "org/model"
    assert cfg.validate() == []
    cfg.scorers = [PluginEntry(SCORER_CLASSIFIER, "none")]
    assert "at least one scorer" in " ".join(cfg.validate())
    cfg.export_strategy = "explode"
    assert "export_strategy" in " ".join(cfg.validate())
    cfg.export_strategy = "merge"
    cfg.model_action = "upload"
    assert "upload_repo_id" in " ".join(cfg.validate())


def test_headless_copy_pins_interactive_values(tmp_path: Path) -> None:
    cfg = HereticConfig.compact()
    cfg.model = "org/model"
    result = cfg.headless_copy(tmp_path / "out", tmp_path / "checkpoints")
    assert result.model_action == "save"
    assert result.export_strategy == "merge"
    assert result.checkpoint_action == "restart"
    assert result.trial_index == 0
    assert result.save_directory == str(tmp_path / "out")
    assert result.study_checkpoint_dir == str(tmp_path / "checkpoints")
    # The source configuration is not mutated.
    assert cfg.model_action is None


def test_upload_settings_are_pinned_for_headless_runs(tmp_path: Path) -> None:
    cfg = HereticConfig.compact()
    cfg.model = "org/model"
    cfg.model_action = "upload"
    cfg.upload_repo_id = "user/model-heretic"
    result = cfg.headless_copy(tmp_path / "out", tmp_path / "checkpoints")
    assert result.upload_repo_private is False
    assert result.upload_reproducibility_information == "none"
    assert 'model_action = "upload"' in result.to_toml()
    assert 'upload_repo_id = "user/model-heretic"' in result.to_toml()


def test_special_characters_survive_toml_serialization() -> None:
    cfg = HereticConfig.compact()
    cfg.model = 'org/"quoted"-model'
    cfg.system_prompt = 'Line one\nLine two with "quotes" and \\\nslashes'
    cfg.response_prefix_test_prompts = [
        DatasetSpec("org/data", "train[:3]", "text", prefix="A|B :: C\n")
    ]
    text = cfg.to_toml()
    parsed = HereticConfig.from_toml(text)
    assert parsed.system_prompt == cfg.system_prompt
    assert parsed.response_prefix_test_prompts[0].prefix == "A|B :: C\n"
    assert parsed.model == cfg.model


def test_malformed_or_unknown_toml_is_rejected() -> None:
    with pytest.raises(HereticConfigError, match="Invalid TOML"):
        HereticConfig.from_toml("n_trials = [")
    with pytest.raises(HereticConfigError, match="n_trials"):
        HereticConfig.from_toml('n_trials = "two"')
    with pytest.raises(HereticConfigError, match="unknown dataset field"):
        HereticConfig.from_toml("[[response_prefix_test_prompts]]\nbogus = 1\n")


def test_modifier_and_dataset_lists_are_typed() -> None:
    cfg = HereticConfig.from_toml(
        f'modifiers = [{{ plugin = "{MODIFIER_ABLITERATION}" }}]\n'
        'batch_size_test_prompts = { dataset = "org/data", split = "train[:2]" }\n'
    )
    assert cfg.modifiers[0].plugin == MODIFIER_ABLITERATION
    assert cfg.batch_size_test_prompts[0].split == "train[:2]"

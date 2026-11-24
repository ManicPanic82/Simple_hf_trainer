"""Main window startup, tab contract and cross-tab navigation tests."""

from __future__ import annotations

from ui.main_window import MainWindow


def test_main_window_starts_with_exactly_six_workflow_tabs(context) -> None:
    window = MainWindow(context)
    try:
        assert window.tab_names() == [
            "Models",
            "Datasets",
            "Training",
            "Analytics",
            "Heretic",
            "Ollama Export",
        ]
        assert window.tabs.count() == 6
        assert hasattr(window, "analytics_tab")
        assert hasattr(window, "ollama_export_tab")
    finally:
        window.close()


def test_model_and_dataset_handoffs_open_the_correct_workspace(context) -> None:
    window = MainWindow(context)
    try:
        window._on_use_model_in_training("org/model")
        assert window.tabs.currentIndex() == 2
        assert window.training_tab.model_combo.currentText() == "org/model"

        window._on_use_dataset_in_training("C:/datasets/train.jsonl")
        assert window.tabs.currentIndex() == 2
        assert window.training_tab.dataset_combo.currentText() == "C:/datasets/train.jsonl"

        window._on_use_model_in_heretic("org/model")
        assert window.tabs.currentIndex() == 4
        assert window.heretic_tab.model_combo.currentText() == "org/model"
    finally:
        window.close()

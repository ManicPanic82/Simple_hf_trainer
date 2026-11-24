"""End-to-end UI checks for the desktop window and its tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

from app.application import create_window


def test_ui_e2e_window_tabs_and_widgets(context) -> None:
    window = create_window(context)
    try:
        window.show()
        QApplication.processEvents()

        assert window.tabs.count() == 6
        assert window.tab_names() == [
            "Models",
            "Datasets",
            "Training",
            "Analytics",
            "Heretic",
            "Ollama Export",
        ]

        checks = [
            (0, "Models", lambda t: t.models_tab),
            (1, "Datasets", lambda t: t.datasets_tab.local_table),
            (2, "Training", lambda t: t.training_tab.model_combo),
            (2, "Training", lambda t: t.training_tab.start_button),
            (3, "Analytics", lambda t: t.analytics_tab.loss_chart),
            (4, "Heretic", lambda t: t.heretic_tab.run_button),
            (5, "Ollama Export", lambda t: t.ollama_export_tab.start_button),
        ]
        for index, name, getter in checks:
            window.tabs.setCurrentIndex(index)
            QApplication.processEvents()
            assert window.tabs.currentIndex() == index
            assert window.tabs.tabText(index) == name
            widgets = getter(window)
            if not isinstance(widgets, (list, tuple)):
                widgets = [widgets]
            assert widgets
            for widget in widgets:
                assert isinstance(widget, QWidget)
                assert widget.isVisible()
    finally:
        window.close()

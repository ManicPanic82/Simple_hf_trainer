"""Tests for the shared engine output console."""

from __future__ import annotations

from ui.console import ConsoleWidget


def test_append_records_each_multiline_line_once(qapp) -> None:
    console = ConsoleWidget()
    console.start_session("Training", "org/model")

    console.append("first line\nsecond line", level="warning", source="worker")

    appended = [entry for entry in console.history if entry.text in {"first line", "second line"}]
    assert [entry.text for entry in appended] == ["first line", "second line"]
    assert all(entry.level == "warning" for entry in appended)
    assert all(entry.source == "worker" for entry in appended)
    assert console.visible_text().count("first line") == 1
    assert console.visible_text().count("second line") == 1


def test_clear_display_preserves_history_and_saved_log(qapp, tmp_path) -> None:
    console = ConsoleWidget()
    console.start_session("Heretic", "org/model")
    console.append("trial complete")

    before = console.full_text()
    console.clear_display()

    assert console.visible_text() == ""
    assert console.full_text() == before
    target = console.save_log_to(tmp_path / "engine.log")
    assert target.read_text(encoding="utf-8").count("trial complete") == 1

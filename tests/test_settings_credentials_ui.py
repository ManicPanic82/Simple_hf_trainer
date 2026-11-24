"""Tests for Hugging Face token credential handling in SettingsDialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLineEdit

from services.credentials import MemoryCredentialStore
from services.settings_service import AppSettings, SettingsService
from ui.settings_dialog import SettingsDialog


def test_settings_dialog_token_roundtrip(qapp, tmp_path: Path):
    store = MemoryCredentialStore()
    settings_service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=Path(__file__).parents[1] / "config" / "defaults.json",
    )
    dialog = SettingsDialog(
        AppSettings(),
        settings_service=settings_service,
        credentials=store,
    )

    assert dialog.token_pill.text() == "No token stored"

    # Set and store token
    dialog.token_edit.setText("hf_testtoken12345")
    dialog.save_token_btn.click()

    assert store.get_token() == "hf_testtoken12345"
    assert dialog.token_pill.text() == "Token stored securely"

    # Clear token
    dialog.clear_token_btn.click()
    assert store.get_token() is None
    assert dialog.token_pill.text() == "No token stored"


def test_settings_dialog_toggle_token_visibility(qapp, tmp_path: Path):
    store = MemoryCredentialStore()
    settings_service = SettingsService(
        settings_file=tmp_path / "settings.json",
        defaults_file=Path(__file__).parents[1] / "config" / "defaults.json",
    )
    dialog = SettingsDialog(
        AppSettings(),
        settings_service=settings_service,
        credentials=store,
    )

    assert dialog.token_edit.echoMode() == QLineEdit.EchoMode.Password
    assert dialog.show_token_btn.text() == "Show"

    dialog.show_token_btn.click()
    assert dialog.token_edit.echoMode() == QLineEdit.EchoMode.Normal
    assert dialog.show_token_btn.text() == "Hide"

    dialog.show_token_btn.click()
    assert dialog.token_edit.echoMode() == QLineEdit.EchoMode.Password
    assert dialog.show_token_btn.text() == "Show"

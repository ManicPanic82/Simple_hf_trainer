"""Logging redaction and secure credential storage."""

from __future__ import annotations

import logging
import sys
import uuid

import pytest

from app.logging_setup import RedactingFilter, configure_logging, redact
from services.credentials import DEFAULT_TARGET, create_credential_store

# Obvious placeholder so the file can never be mistaken for a real token.
SECRET = "hf_" + "NOTAREALTOKEN" * 4


def test_redact_masks_hugging_face_tokens() -> None:
    text = f"Authorization: Bearer {SECRET}"
    result = redact(text)
    assert SECRET not in result
    assert "REDACTED" in result


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("Downloading Qwen/Qwen2.5-0.5B") == "Downloading Qwen/Qwen2.5-0.5B"


def test_log_filter_masks_message_and_arguments() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "token=%s", (SECRET,), None)
    assert RedactingFilter().filter(record) is True
    assert SECRET not in record.getMessage()


def test_file_log_never_contains_token(tmp_path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    path = configure_logging(tmp_path / "logs" / "app.log", to_stderr=False)
    try:
        assert path is not None
        logging.getLogger("test").info("using token %s", SECRET)
        for handler in root.handlers:
            handler.flush()
        content = path.read_text(encoding="utf-8")
        assert SECRET not in content
        assert "REDACTED" in content
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_configure_logging_survives_unusable_path(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file, not a directory", encoding="utf-8")
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        assert configure_logging(blocked / "app.log") is None
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Credential Manager only")
def test_windows_credential_manager_roundtrip() -> None:
    store = create_credential_store()
    if not store.available:
        pytest.skip("Windows Credential Manager is unavailable")

    target = f"SimpleHFTrainer/__pytest__/{uuid.uuid4()}"
    token = "hf_" + "PYTESTONLY" * 4
    try:
        assert store.get_token(target) is None
        store.set_token(token, target)
        assert store.get_token(target) == token
    finally:
        store.delete_token(target)
    assert store.get_token(target) is None


def test_default_credential_target_is_namespaced() -> None:
    assert DEFAULT_TARGET.startswith("SimpleHFTrainer/")

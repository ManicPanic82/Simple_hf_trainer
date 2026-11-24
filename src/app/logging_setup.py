"""Logging configuration with credential redaction.

Log files are local, rotated, and never contain Hugging Face tokens: any
``hf_...`` style token is replaced before a record reaches disk or stderr.
"""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")
BEARER_PATTERN = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{8,}")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def redact(text: str) -> str:
    """Return ``text`` with credential-shaped substrings masked."""
    text = TOKEN_PATTERN.sub("hf_****REDACTED****", text)
    return BEARER_PATTERN.sub(r"\1****REDACTED****", text)


class RedactingFilter(logging.Filter):
    """Masks credentials in the message and its arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_safe(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {key: _safe(value) for key, value in record.args.items()}
        return True


def _safe(value: object) -> object:
    return redact(value) if isinstance(value, str) else value


def configure_logging(
    path: Path,
    *,
    level: int = logging.INFO,
    to_stderr: bool = False,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
) -> Path | None:
    """Attach a rotating file handler to the root logger.

    Returns the log path, or ``None`` if the log directory could not be used.
    Logging failures never prevent the application from starting.
    """
    root = logging.getLogger()
    root.setLevel(level)

    redactor = RedactingFilter()
    formatter = logging.Formatter(LOG_FORMAT)

    if to_stderr:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        stream.addFilter(redactor)
        root.addHandler(stream)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    except OSError:
        logging.getLogger(__name__).warning("Could not open log file at %s", path)
        return None

    handler.setFormatter(formatter)
    handler.addFilter(redactor)
    root.addHandler(handler)
    return path

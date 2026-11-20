"""Locations owned by the application on disk.

All paths here are per-user and must never be committed to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Simple HF Trainer"
APP_DIR_NAME = "SimpleHFTrainer"
HOME_ENV_VAR = "SIMPLE_HF_TRAINER_HOME"


def _local_app_data() -> Path:
    raw = os.environ.get("LOCALAPPDATA")
    if raw:
        return Path(raw)
    return Path.home() / "AppData" / "Local"


def app_data_dir() -> Path:
    """Directory for settings, logs and other small per-user state."""
    return _local_app_data() / APP_DIR_NAME


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def log_dir() -> Path:
    return app_data_dir() / "logs"


def log_file() -> Path:
    return log_dir() / "simple-hf-trainer.log"


def default_home() -> Path:
    """Root for default model/dataset/output storage.

    Overridable with the ``SIMPLE_HF_TRAINER_HOME`` environment variable so
    large artefacts can live on a roomier drive.
    """
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / APP_DIR_NAME


def repository_root() -> Path:
    """Repository root derived from this file (``src/app/paths.py``)."""
    return Path(__file__).resolve().parents[2]

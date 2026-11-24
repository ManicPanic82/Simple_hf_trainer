"""Persistent catalog of locally discovered models and datasets.

This is *not* the engine registry (see ``services.engine_registry``): it stores
plain JSON-serialisable dicts keyed by their on-disk ``path``, so re-scanning a
library updates entries in place instead of duplicating them.

Loading is defensive - a missing, corrupt, or hand-edited file degrades to an
empty catalog rather than preventing startup. Writes are atomic (temporary file
plus ``os.replace``) so an interrupted save cannot leave a truncated catalog.
The module deliberately has no Qt dependency.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from app import paths

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"

__all__ = ["Registry", "to_record_dict"]


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable copy, stringifying paths and unknown types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_record_dict(record: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping record into a plain JSON-safe dict."""
    if is_dataclass(record) and not isinstance(record, type):
        record = asdict(record)
    if not isinstance(record, Mapping):
        return {}
    cleaned = _json_safe(record)
    return cleaned if isinstance(cleaned, dict) else {}


def _identity(record: Mapping[str, Any]) -> str:
    """Stable key for a record; an empty string means 'not a usable record'."""
    raw = record.get("path")
    if raw is None:
        return ""
    text = str(raw).strip()
    return str(Path(text)) if text else ""


class Registry:
    """On-disk catalog of local models and datasets keyed by path.

    ``add_model`` and ``add_dataset`` only touch memory; call :meth:`save` once
    after a batch of updates (as the scan services do) to persist the catalog.
    """

    def __init__(
        self,
        home: Path | str | None = None,
        *,
        path: Path | str | None = None,
    ) -> None:
        if path is not None:
            self._path = Path(path)
        else:
            base = Path(home) if home is not None else paths.app_data_dir()
            self._path = base / REGISTRY_FILENAME
        self._models: dict[str, dict[str, Any]] = {}
        self._datasets: dict[str, dict[str, Any]] = {}
        self.issues: list[str] = []
        self.load()

    @property
    def path(self) -> Path:
        """Location of the backing ``registry.json`` file."""
        return self._path

    # -- in-memory catalog ------------------------------------------------

    def add_model(self, record: Mapping[str, Any]) -> None:
        """Insert or replace a model record, keyed by its ``path``."""
        self._upsert(self._models, record)

    def add_dataset(self, record: Mapping[str, Any]) -> None:
        """Insert or replace a dataset record, keyed by its ``path``."""
        self._upsert(self._datasets, record)

    def get_models(self) -> list[dict[str, Any]]:
        """Return a copy of the current model records."""
        return [dict(item) for item in self._models.values()]

    def get_datasets(self) -> list[dict[str, Any]]:
        """Return a copy of the current dataset records."""
        return [dict(item) for item in self._datasets.values()]

    @staticmethod
    def _upsert(store: dict[str, dict[str, Any]], record: Mapping[str, Any]) -> None:
        data = to_record_dict(record)
        identity = _identity(data)
        if not identity:
            return
        data["path"] = identity
        store[identity] = data

    # -- persistence ------------------------------------------------------

    def load(self) -> None:
        """Load the catalog defensively; invalid entries are ignored."""
        self.issues = []
        self._models = {}
        self._datasets = {}
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except (OSError, UnicodeError) as error:
            self.issues.append(f"Registry could not be read ({error}); using an empty catalog.")
            return

        try:
            payload = json.loads(text)
        except ValueError as error:
            self.issues.append(f"Registry is not valid JSON ({error}); using an empty catalog.")
            return
        if not isinstance(payload, dict):
            self.issues.append("Registry is not a JSON object; using an empty catalog.")
            return

        for key, store in (("models", self._models), ("datasets", self._datasets)):
            entries = payload.get(key)
            if entries is None:
                continue
            if not isinstance(entries, list):
                self.issues.append(f"Registry '{key}' is not a list; ignoring it.")
                continue
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    self.issues.append(f"Ignoring invalid {key} entry #{index}.")
                    continue
                data = to_record_dict(entry)
                identity = _identity(data)
                if not identity:
                    self.issues.append(f"Ignoring {key} entry #{index} without a path.")
                    continue
                data["path"] = identity
                store[identity] = data

    def save(self) -> bool:
        """Write the catalog atomically. Returns ``False`` if writing failed."""
        payload = {"models": self.get_models(), "datasets": self.get_datasets()}
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=f"{self._path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, self._path)
            temporary = None
        except (OSError, TypeError, ValueError) as error:
            logger.warning("Could not write registry %s: %s", self._path, error)
            return False
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
        return True

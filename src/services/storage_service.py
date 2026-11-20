"""Validation and safe creation of user-configured storage directories.

Rules:
* All configured storage paths must be absolute.
* A path that exists but is a file (or a non-directory) is an error.
* Missing directories inside an existing parent are reported as informational
  and can be created on request.
* Nothing is ever deleted, moved, or "cleaned up".
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from services.settings_service import AppSettings


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class StorageIssue:
    severity: Severity
    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.severity.value}: {self.message}"


@dataclass(frozen=True)
class PathValidation:
    label: str
    path: Path
    exists: bool
    writable: bool
    issues: tuple[StorageIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)


@dataclass
class StorageValidation:
    results: list[PathValidation] = field(default_factory=list)
    created: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def issues(self) -> list[StorageIssue]:
        return [issue for result in self.results for issue in result.issues]


class StorageService:
    """Owns the model/dataset/output/cache directories from settings."""

    LABELS = ("models_dir", "datasets_dir", "outputs_dir", "hf_cache_dir")

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings

    def path_for(self, label: str) -> Path:
        if label not in self.LABELS:
            raise KeyError(f"Unknown storage label: {label}")
        return dict(self._settings.paths.items())[label]

    def all_paths(self) -> dict[str, Path]:
        return dict(self._settings.paths.items())

    def validate(self, label: str, *, probe_writable: bool = True) -> PathValidation:
        path = self.path_for(label)
        issues: list[StorageIssue] = []

        raw = str(path)
        if not raw or raw.strip() == "":
            issues.append(StorageIssue(Severity.ERROR, path, "Path is empty."))
            return PathValidation(label, path, False, False, tuple(issues))

        if not path.is_absolute():
            issues.append(
                StorageIssue(Severity.ERROR, path, "Path must be absolute (for example C:\\Data).")
            )
            return PathValidation(label, path, False, False, tuple(issues))

        exists = path.exists()
        is_directory = path.is_dir()
        writable = False

        if exists and not is_directory:
            issues.append(StorageIssue(Severity.ERROR, path, "Path exists but is not a directory."))
        elif exists:
            writable = self._can_write(path) if probe_writable else os.access(path, os.W_OK)
            if probe_writable and not writable:
                issues.append(
                    StorageIssue(Severity.ERROR, path, "Directory is not writable by this user.")
                )
        else:
            anchor = self._nearest_existing_parent(path)
            if anchor is None:
                issues.append(
                    StorageIssue(
                        Severity.ERROR,
                        path,
                        "No existing parent directory was found for this path.",
                    )
                )
            elif not anchor.is_dir():
                issues.append(
                    StorageIssue(Severity.ERROR, path, f"Parent path is not a directory: {anchor}")
                )
            else:
                issues.append(
                    StorageIssue(
                        Severity.WARNING,
                        path,
                        f"Directory does not exist yet; it will be created under {anchor}.",
                    )
                )
                writable = self._can_write(anchor) if probe_writable else os.access(anchor, os.W_OK)
                if probe_writable and not writable:
                    issues.append(
                        StorageIssue(
                            Severity.ERROR,
                            path,
                            f"Parent directory is not writable: {anchor}",
                        )
                    )

        return PathValidation(label, path, exists, writable, tuple(issues))

    def validate_all(self, *, probe_writable: bool = True) -> StorageValidation:
        return StorageValidation(
            results=[self.validate(label, probe_writable=probe_writable) for label in self.LABELS]
        )

    def ensure_directories(self) -> StorageValidation:
        """Create missing configured directories without touching existing data."""
        validation = StorageValidation()
        for label in self.LABELS:
            result = self.validate(label, probe_writable=False)
            path = result.path
            if result.ok and not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    validation.created.append(path)
                except OSError as error:
                    result = PathValidation(
                        label,
                        path,
                        path.exists(),
                        False,
                        result.issues
                        + (
                            StorageIssue(
                                Severity.ERROR,
                                path,
                                f"Could not create directory: {error}",
                            ),
                        ),
                    )
            validation.results.append(result)
        return validation

    @staticmethod
    def _nearest_existing_parent(path: Path) -> Path | None:
        for candidate in path.parents:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _can_write(directory: Path) -> bool:
        """Probe writability by creating and removing a temporary file."""
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(directory), prefix=".sht-write-probe-", delete=True
            ):
                return True
        except OSError:
            return False

"""Live, timestamped output console for engine worker processes.

The console keeps a complete in-memory history that is independent of the
visible widget.  ``Clear display`` only clears the rendered text; it never
touches the history, the worker process, or a log that was already saved.

Rendering is bounded through ``QPlainTextEdit.setMaximumBlockCount`` so a very
long run cannot make the GUI progressively slower, while the full history is
still available for ``Copy`` and ``Save log``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui import theme

LEVELS = frozenset({"info", "warning", "error", "system"})
_LEVEL_TAGS = {"warning": "[WARN]", "error": "[ERROR]"}
_LEVEL_COLORS = {
    "info": theme.TEXT,
    "warning": theme.WARN,
    "error": theme.ERROR,
    "system": theme.TEXT_MUTED,
}


@dataclass(frozen=True)
class LogEntry:
    """One timestamped line in a console session."""

    timestamp: datetime
    level: str
    text: str
    source: str | None = None
    session: int = 0


@dataclass
class LogSession:
    """All output produced by one engine run (or the idle session)."""

    index: int
    title: str
    started_at: datetime
    subtitle: str = ""
    entries: list[LogEntry] = field(default_factory=list)


class ConsoleWidget(QWidget):
    """Dark terminal-style output view with copy/clear/save controls."""

    saved = Signal(str)
    save_failed = Signal(str)

    def __init__(
        self,
        *,
        max_visible_blocks: int = 4000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_visible_blocks = max(200, int(max_visible_blocks))
        self._sessions: list[LogSession] = []
        self._current: LogSession | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.session_label = QLabel("No session yet", self)
        self.session_label.setObjectName("Muted")
        toolbar.addWidget(self.session_label)
        toolbar.addStretch(1)

        self.auto_scroll = QCheckBox("Auto-scroll", self)
        self.auto_scroll.setChecked(True)
        self.auto_scroll.setToolTip(
            "Follow new output automatically. Turn off to inspect earlier lines."
        )
        toolbar.addWidget(self.auto_scroll)

        self.copy_button = QPushButton("Copy", self)
        self.copy_button.setToolTip("Copy the selected text, or the complete retained history.")
        toolbar.addWidget(self.copy_button)

        self.clear_button = QPushButton("Clear display", self)
        self.clear_button.setToolTip(
            "Clear only the visible text. The run, its history, and saved logs are unaffected."
        )
        toolbar.addWidget(self.clear_button)

        self.save_button = QPushButton("Save log…", self)
        self.save_button.setToolTip(
            "Write the complete retained session history to a .log or .txt file."
        )
        toolbar.addWidget(self.save_button)
        layout.addLayout(toolbar)

        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setObjectName("Console")
        self.text_edit.setReadOnly(True)
        self.text_edit.setUndoRedoEnabled(False)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setMaximumBlockCount(self._max_visible_blocks)
        # The surrounding workstation gives the console a resizable splitter;
        # this minimum keeps roughly a dozen lines available even when the user
        # drags the monitor open.
        self.text_edit.setMinimumHeight(220)
        layout.addWidget(self.text_edit, 1)

        self.copy_button.clicked.connect(self.copy)
        self.clear_button.clicked.connect(self.clear_display)
        self.save_button.clicked.connect(lambda: self.save_log())

    # -- sessions --------------------------------------------------------

    @property
    def sessions(self) -> list[LogSession]:
        return list(self._sessions)

    @property
    def current_session(self) -> LogSession | None:
        return self._current

    def start_session(self, title: str, subtitle: str = "") -> LogSession:
        """Begin a distinct session without discarding earlier sessions."""
        session = LogSession(
            index=len(self._sessions) + 1,
            title=title or "Output",
            started_at=datetime.now(),
            subtitle=subtitle,
        )
        self._sessions.append(session)
        self._current = session
        self.session_label.setText(f"Session {session.index} · {session.title}")
        header = f"═══ Session {session.index} · {session.title} ═══"
        if subtitle:
            header = f"{header} — {subtitle}"
        self._record(session, LogEntry(datetime.now(), "system", header, None, session.index))
        # Add a separator line after session header for visual clarity
        self._record(
            session,
            LogEntry(
                datetime.now(),
                "system",
                "----------------------------------------",
                None,
                session.index,
            ),
        )
        return session

    # -- output ----------------------------------------------------------

    def append(
        self,
        text: object,
        *,
        level: str = "info",
        source: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append text, splitting multi-line messages into timestamped lines."""
        if level not in LEVELS:
            level = "info"
        message = "" if text is None else str(text)
        if not message:
            return

        session = self._current or self.start_session("Output")

        lines = message.splitlines()
        for i, line in enumerate(lines):
            # Add special formatting for session headers
            if "Session" in line:
                # Add spacing before session header (if there's a previous line with "Session" text)
                if i > 0 and "Session" in lines[i - 1]:
                    self._record(
                        session,
                        LogEntry(
                            timestamp or datetime.now(),
                            "info",
                            "=== " + lines[i - 1].strip() + " ===",
                            source,
                            session.index,
                        ),
                    )

            # Add spacing after session headers
            if "Session" in line:
                self._record(
                    session,
                    LogEntry(
                        timestamp or datetime.now(),
                        "info",
                        "----------------------------------------",
                        source,
                        session.index,
                    ),
                )

            # Add all lines to the log
            for line in lines:
                self._record(
                    session,
                    LogEntry(
                        timestamp or datetime.now(),
                        level,
                        line,
                        source,
                        session.index,
                    ),
                )

    def append_many(self, entries: list[tuple[str, str, str | None]]) -> None:
        """Append ``(text, level, source)`` triples in one call."""
        for text, level, source in entries:
            self.append(text, level=level, source=source)

    def flush(self) -> None:
        """Compatibility hook; entries are rendered immediately."""

    def _record(self, session: LogSession, entry: LogEntry) -> None:
        session.entries.append(entry)
        self._render(entry)

    def _render(self, entry: LogEntry) -> None:
        scroll = self.text_edit.verticalScrollBar()
        at_bottom = scroll.value() >= scroll.maximum() - 4
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text_edit.document().isEmpty():
            cursor.insertBlock()
        cursor.setCharFormat(self._format_for(entry.level))
        cursor.insertText(self._format_entry(entry))
        if self.auto_scroll.isChecked() and at_bottom:
            scroll.setValue(scroll.maximum())

    @staticmethod
    def _format_for(level: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_LEVEL_COLORS.get(level, theme.TEXT)))
        return fmt

    @staticmethod
    def _format_entry(entry: LogEntry) -> str:
        parts = [f"[{entry.timestamp:%H:%M:%S}]"]
        tag = _LEVEL_TAGS.get(entry.level)
        if tag:
            parts.append(tag)
        if entry.source:
            parts.append(f"[{entry.source}]")
        parts.append(entry.text)
        return " ".join(parts)

    # -- history ---------------------------------------------------------

    @property
    def history(self) -> list[LogEntry]:
        return [entry for session in self._sessions for entry in session.entries]

    def full_text(self) -> str:
        """Every retained line, including lines no longer visible."""
        return "\n".join(self._format_entry(entry) for entry in self.history)

    def visible_text(self) -> str:
        return self.text_edit.toPlainText()

    # -- actions ---------------------------------------------------------

    def copy(self) -> None:
        """Copy the selection, or the complete retained history when none."""
        cursor = self.text_edit.textCursor()
        if cursor.hasSelection():
            self.text_edit.copy()
            return
        QApplication.clipboard().setText(self.full_text())

    def clear_display(self) -> None:
        """Clear only the rendered text. History and the worker are untouched."""
        self.text_edit.clear()

    def save_log_to(self, path: str | Path) -> Path:
        """Write the complete retained history to ``path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Simple HF Trainer — Heretic console log",
            f"# Saved: {datetime.now().isoformat(timespec='seconds')}",
            f"# Sessions: {len(self._sessions)}",
            "",
        ]
        if not self._sessions:
            lines.append("(no output yet)")
        for session in self._sessions:
            lines.append(
                f"=== Session {session.index}: {session.title} "
                f"(started {session.started_at.isoformat(timespec='seconds')}) ==="
            )
            if session.subtitle:
                lines.append(f"# {session.subtitle}")
            lines.extend(self._format_entry(entry) for entry in session.entries)
            lines.append("")
        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
        return target

    def save_log(self, parent: QWidget | None = None) -> Path | None:
        """Ask for a destination and save the complete history."""
        default_name = f"heretic-console-{datetime.now():%Y%m%d-%H%M%S}.log"
        path, _ = QFileDialog.getSaveFileName(
            parent or self,
            "Save Heretic console log",
            default_name,
            "Log files (*.log *.txt);;All files (*.*)",
        )
        if not path:
            return None
        try:
            target = self.save_log_to(path)
        except OSError as error:
            self.save_failed.emit(str(error))
            return None
        self.saved.emit(str(target))
        return target


__all__ = ["ConsoleWidget", "Console", "LogEntry", "LogSession"]

# Convenience alias for users who expect "Console" (like the main_window.py imports it)
Console = ConsoleWidget

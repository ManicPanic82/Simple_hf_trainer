"""Shared page scaffold for the four workflow tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.components import Pill


class TabPage(QWidget):
    """Title, one-line purpose, optional state pill, and a scrollable body."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("TabPage")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 14)
        outer.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(12)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("TabTitle")
        header.addWidget(self.title_label)

        self.status_pill = Pill("Initialising…", "info", self)
        header.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        self.header = header
        outer.addLayout(header)

        self.subtitle_label = QLabel(subtitle, self)
        self.subtitle_label.setObjectName("Muted")
        self.subtitle_label.setWordWrap(True)
        outer.addWidget(self.subtitle_label)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget(self._scroll)
        self._container.setObjectName("TabContent")
        self.content = QVBoxLayout(self._container)
        self.content.setContentsMargins(0, 4, 4, 4)
        self.content.setSpacing(12)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, 1)

    def add(self, widget: QWidget) -> QWidget:
        self.content.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self.content.addLayout(layout)

    def finish(self) -> None:
        """Append a trailing stretch once all cards are added."""
        self.content.addStretch(1)

    def set_status(self, text: str, level: str = "info") -> None:
        """Show the current tab readiness without implying unfinished UI is complete."""
        self.status_pill.set_state(level, text)

    def add_header_widget(self, widget: QWidget) -> QWidget:
        self.header.addWidget(widget)
        return widget

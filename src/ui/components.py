"""Reusable presentation widgets shared by the four workflow tabs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

PILL_STATES = frozenset({"ok", "warn", "error", "info", "off"})


class Pill(QLabel):
    """Small state badge. ``state`` drives colour through the stylesheet."""

    def __init__(self, text: str = "", state: str = "off", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Pill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        if state not in PILL_STATES:
            state = "off"
        if text is not None:
            self.setText(text)
        self.setProperty("state", state)
        style = self.style()
        style.unpolish(self)
        style.polish(self)


class Card(QFrame):
    """Bordered surface with a title, optional subtitle, and body layout."""

    def __init__(
        self, title: str, subtitle: str | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("CardTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.header = header
        layout.addLayout(header)

        self.subtitle_label = QLabel(subtitle or "", self)
        self.subtitle_label.setObjectName("CardSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.body.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.body)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def add_header_widget(self, widget: QWidget) -> QWidget:
        self.header.addWidget(widget)
        return widget


class FieldRow(QWidget):
    """``Caption   Value`` row with a selectable value."""

    def __init__(
        self,
        caption: str,
        value: str = "",
        *,
        caption_width: int = 116,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.caption = QLabel(caption, self)
        self.caption.setObjectName("Muted")
        self.caption.setMinimumWidth(caption_width)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.caption)

        self.value = QLabel(value, self)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.value.setWordWrap(True)
        layout.addWidget(self.value, 1)

    def set_value(self, text: str) -> None:
        self.value.setText(text)


class ElidedPathLabel(QLabel):
    """Path label that elides in the middle while keeping the full tooltip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PathValue")
        self._full_text = ""

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._refresh()

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def minimumSizeHint(self):  # noqa: N802 - Qt API
        # The elided text must never force the surrounding form wider than the
        # available column; it simply elides further.
        hint = super().minimumSizeHint()
        hint.setWidth(80)
        return hint

    def _refresh(self) -> None:
        metrics = QFontMetrics(self.font())
        available = max(80, self.width() - 4)
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, available)
        super().setText(elided)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self.set_full_text(text)


class StoragePathRow(QWidget):
    """Shows one configured storage path with its validation state."""

    open_requested = Signal(str)

    def __init__(self, label: str, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self._note = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        caption_label = QLabel(caption, self)
        caption_label.setObjectName("Muted")
        caption_label.setMinimumWidth(116)
        layout.addWidget(caption_label)

        self.path_label = ElidedPathLabel(self)
        layout.addWidget(self.path_label, 1)

        self.pill = Pill("Not checked", "off", self)
        layout.addWidget(self.pill)

        self.button = QPushButton("Open folder", self)
        self.button.clicked.connect(lambda: self.open_requested.emit(self.label))
        layout.addWidget(self.button)

    def set_path(
        self,
        path: Path,
        *,
        ok: bool,
        exists: bool,
        note: str = "",
        severity: str = "off",
    ) -> None:
        self.path_label.set_full_text(str(path))
        self._note = note
        self.setToolTip(note)
        if ok and exists:
            self.pill.set_state("ok", "Ready")
            self.button.setText("Open folder")
        elif ok:
            self.pill.set_state("info", "To be created")
            self.button.setText("Create folder")
        else:
            self.pill.set_state("error" if severity != "warn" else "warn", "Problem")
            self.button.setText("Open folder")
        self.button.setEnabled(bool(str(path).strip()))

    def note(self) -> str:
        return self._note


class CollapsibleSection(QWidget):
    """Titled section whose body can be expanded or collapsed."""

    def __init__(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("SectionToggle")
        # QToolButton treats '&' as a mnemonic marker, so escape it to keep
        # titles such as "Hardware & execution" readable.
        self.toggle.setText(title.replace("&", "&&"))
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.clicked.connect(self.set_expanded)
        outer.addWidget(self.toggle)

        self.subtitle_label: QLabel | None = None
        if subtitle:
            self.subtitle_label = QLabel(subtitle, self)
            self.subtitle_label.setObjectName("Muted")
            self.subtitle_label.setWordWrap(True)
            self.subtitle_label.setVisible(expanded)
            outer.addWidget(self.subtitle_label)

        self._body = QWidget(self)
        self.body = QVBoxLayout(self._body)
        self.body.setContentsMargins(8, 2, 0, 4)
        self.body.setSpacing(8)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._body.setVisible(expanded)
        if self.subtitle_label is not None:
            self.subtitle_label.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self._body.isVisible()

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)


__all__ = [
    "Card",
    "CollapsibleSection",
    "ElidedPathLabel",
    "FieldRow",
    "Pill",
    "StoragePathRow",
]

"""Card that reports one optional engine's detection state."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from services.engine_registry import EngineState, EngineStatus
from ui.components import Card, FieldRow, Pill

_STATE_TO_PILL = {
    EngineState.AVAILABLE: ("ok", "Engine available"),
    EngineState.NOT_INSTALLED: ("warn", "Engine not installed"),
    EngineState.INTERPRETER_MISSING: ("warn", "Environment not set up"),
    EngineState.ERROR: ("error", "Engine check failed"),
    EngineState.UNCHECKED: ("off", "Not checked yet"),
}


class EngineCard(Card):
    """Shows interpreter, install state, version, and detail for one engine."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(title, subtitle, parent)
        self.pill = Pill("Not checked yet", "off", self)
        self.add_header_widget(self.pill)

        self.interpreter_row = FieldRow("Interpreter", "Not configured", caption_width=104)
        self.state_row = FieldRow("Install state", "Not checked yet", caption_width=104)
        self.version_row = FieldRow("Version", "—", caption_width=104)
        self.add(self.interpreter_row)
        self.add(self.state_row)
        self.add(self.version_row)

    def apply_status(self, status: EngineStatus) -> None:
        state, text = _STATE_TO_PILL[status.state]
        self.pill.set_state(state, text)
        self.interpreter_row.set_value(
            str(status.interpreter) if status.interpreter else "Not configured"
        )
        self.state_row.set_value(status.detail or text)
        self.version_row.set_value(status.version or "—")
        self.setToolTip(status.detail)

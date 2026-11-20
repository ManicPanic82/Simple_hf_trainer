"""Launch the real main window and save a screenshot for manual verification.

Uses the application's own settings/log locations, exactly like `main.py`.

    .\\environments\\app\\Scripts\\python.exe scripts\\screenshot_app.py [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.application import build_context, create_application, create_window  # noqa: E402
from ui.theme import apply_theme  # noqa: E402


def main(argv: list[str]) -> int:
    output = Path(argv[1]) if len(argv) > 1 else ROOT / "docs/evidence/ui-main-window.png"
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    app = create_application(argv[:1])
    apply_theme(app)

    context = build_context()
    window = create_window(context)
    window.resize(1280, 820)
    window.show()
    window.start()

    def capture() -> None:
        image = window.grab()
        image.save(str(output))
        # Also capture each tab so the UI can be reviewed without interaction.
        for index, name in enumerate(window.TAB_NAMES):
            window.tabs.setCurrentIndex(index)
            QApplication.processEvents()
            shot = window.grab()
            shot.save(str(output.with_name(f"{output.stem}-{name.lower()}{output.suffix}")))
        window.close()
        app.quit()

    QTimer.singleShot(4500, capture)
    QApplication.processEvents()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

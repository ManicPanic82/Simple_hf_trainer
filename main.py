"""Entry point for the Simple HF Trainer desktop application.

Run from the repository root with the application environment:

    .\\environments\\app\\Scripts\\python.exe main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from app.application import run  # noqa: E402  (path bootstrap must run first)

if __name__ == "__main__":
    raise SystemExit(run(sys.argv))

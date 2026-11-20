"""Restrained dark theme for the desktop workstation."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BACKGROUND = "#12161a"
SURFACE = "#1a1f25"
SURFACE_RAISED = "#20262d"
BORDER = "#2a323b"
TEXT = "#e7eaee"
TEXT_MUTED = "#98a3b2"
ACCENT = "#4f9cf9"
OK = "#3fb950"
WARN = "#d29922"
ERROR = "#f85149"

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 9.5pt;
}}
QMainWindow, QDialog {{ background-color: {BACKGROUND}; }}
QWidget#TabPage, QWidget#TabContent {{ background-color: {BACKGROUND}; }}
QScrollArea, QScrollArea QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QMenuBar {{ background-color: {SURFACE}; border-bottom: 1px solid {BORDER}; padding: 2px 6px; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background-color: {SURFACE_RAISED}; border-radius: 4px; }}

QFrame#Header {{
    background-color: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}
QLabel#AppTitle {{ font-size: 15pt; font-weight: 600; }}
QLabel#AppSubtitle {{ color: {TEXT_MUTED}; }}

QTabWidget#MainTabs::pane {{
    border: none;
    background-color: {BACKGROUND};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 9px 20px;
    margin-right: 2px;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QTabBar::tab:focus {{ outline: none; }}

QFrame#Card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#CardTitle {{ font-size: 11pt; font-weight: 600; }}
QLabel#TabTitle {{ font-size: 16pt; font-weight: 600; }}
QLabel#Muted, QLabel#CardSubtitle {{ color: {TEXT_MUTED}; }}
QLabel#PathValue {{
    font-family: "Cascadia Mono", "Consolas", monospace;
    color: {TEXT_MUTED};
}}
QLabel#SectionLabel {{ color: {TEXT_MUTED}; font-weight: 600; }}

QLabel#Pill {{
    border-radius: 9px;
    padding: 2px 9px;
    font-size: 8.5pt;
    font-weight: 600;
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QLabel#Pill[state="ok"] {{
    color: {OK}; border-color: rgba(63, 185, 80, 0.45);
    background-color: rgba(63, 185, 80, 0.10);
}}
QLabel#Pill[state="warn"] {{
    color: {WARN}; border-color: rgba(210, 153, 34, 0.45);
    background-color: rgba(210, 153, 34, 0.10);
}}
QLabel#Pill[state="error"] {{
    color: {ERROR}; border-color: rgba(248, 81, 73, 0.45);
    background-color: rgba(248, 81, 73, 0.10);
}}
QLabel#Pill[state="info"] {{
    color: {ACCENT}; border-color: rgba(79, 156, 249, 0.45);
    background-color: rgba(79, 156, 249, 0.10);
}}

QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: #273039; }}
QPushButton:pressed {{ background-color: #1d242b; }}
QPushButton:disabled {{ color: #5b6573; background-color: #191e23; border-color: #232a32; }}
QPushButton#Primary {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #08101c;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: #6aabfb; }}
QPushButton#Primary:disabled {{ background-color: #26313d; color: #6b7887; border-color: #2a3541; }}

QLineEdit, QComboBox, QSpinBox {{
    background-color: #0e1216;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled {{ color: #6b7887; background-color: {SURFACE}; border-color: #242b33; }}

QProgressBar {{
    background-color: #0e1216;
    border: none;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}

QFrame#StatusStrip {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
}}
QLabel#StripLabel {{ color: {TEXT_MUTED}; font-size: 8.5pt; }}
QLabel#StripValue {{ font-size: 8.5pt; }}
QFrame#StripSep {{ background-color: {BORDER}; max-width: 1px; min-width: 1px; }}

QFrame#Notice {{
    border-radius: 6px;
    border: 1px solid {BORDER};
    background-color: {SURFACE_RAISED};
}}
QFrame#Notice[level="info"] {{ border-color: rgba(79, 156, 249, 0.5); }}
QFrame#Notice[level="warning"] {{ border-color: rgba(210, 153, 34, 0.5); }}
QFrame#Notice[level="error"] {{ border-color: rgba(248, 81, 73, 0.5); }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #333c46; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {BORDER}; }}

QMenu {{ background-color: {SURFACE_RAISED}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background-color: {ACCENT}; color: #08101c; }}
QToolTip {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 6px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the palette and stylesheet to ``app``."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0e1216"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#08101c"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(palette)

    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(STYLESHEET)

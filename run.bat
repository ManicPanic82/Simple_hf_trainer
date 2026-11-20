@echo off
setlocal
cd /d "%~dp0"

if exist "environments\app\Scripts\python.exe" (
    start "" "environments\app\Scripts\python.exe" main.py
) else (
    echo [ERROR] Application virtual environment not found at environments\app.
    echo Please refer to docs\INSTALLATION.md to create environments\app.
    pause
)

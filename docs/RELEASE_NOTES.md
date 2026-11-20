# Release v1.0.0

## Highlights
- **One‑click launch** via `run.bat` or PowerShell `launch.ps1`.
- **GPU‑exclusive job manager** with FIFO queue, allowing multiple jobs to be submitted without immediate rejection.
- **Download progress UI** – `DownloadProgressBar` widget integrated into the Models tab (and ready for Datasets tab).
- **Persistent registry** (`src/services/registry.py`) storing discovered models and datasets in `registry.json`.
- Updated **Installation docs** with a dedicated *Launch* section.
- Comprehensive test suite (`123 tests`) and lint‑free code (`ruff` clean).
- Git tag `v1.0.0` created and pushed.

## Details
- `JobState` now includes `QUEUED` and the manager automatically starts the next queued job after the current one finishes.
- `ModelsTab` now uses `DownloadProgressBar` for model downloads, exposing `update_progress` and `complete` signals.
- `Registry` provides `add_model`, `add_dataset`, `get_models`, `get_datasets` and persists to `registry.json` under the app home directory.
- Documentation (`docs/INSTALLATION.md`) now contains a **Launch** subsection.
- Release notes added and the repository is ready for publishing.

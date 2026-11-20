# Simple HF Trainer

A local-first **Windows desktop application** for four connected workflows:

1. **Models** — discover, download, and manage Hugging Face models.
2. **Datasets** — find, import, preview, and prepare Hugging Face datasets.
3. **Training** — fine-tune compatible models with the real **Unsloth** engine (LoRA/QLoRA).
4. **Heretic** — modify compatible models with the real **Heretic** abliteration tool.

It is deliberately narrow: no chat, no RAG, no agents, no inference server, no cloud
training, no plugin framework. The GPU (a 6 GB GTX 1660 in the reference setup) is treated
as the scarce resource it is.

## Current status

The `feature/desktop-foundation` development branch contains the four-tab desktop
application and working implementations for model discovery, dataset preparation,
Unsloth fine-tuning, and Heretic abliteration. This is a development branch, not a
release-ready claim: the initial project plan called for a desktop foundation first,
and the engine workflows on this branch are ahead of that original phase boundary.

| Area | Current implementation |
| --- | --- |
| Four-tab PySide6 desktop shell | Implemented |
| Hardware status, settings, storage paths | Implemented; hardware values degrade gracefully when probes are unavailable |
| Hugging Face token storage | Windows Credential Manager integration |
| Worker protocol and process runner | JSON-lines protocol, validation, background QProcess execution |
| GPU-exclusive job lifecycle | Centralised lock; completion is reconciled after the child process exits |
| Models | Hub search, background metadata lookup, download, local library and qualified memory estimates |
| Datasets | Hub search, local inspection, preview, preparation; JSON, JSONL, CSV, Parquet and Arrow IPC |
| Training | Unsloth LoRA/QLoRA worker with progress, metrics, dataset validation and cooperative cancellation |
| Heretic | Headless Heretic worker with trial metrics and cancellation interrupt |

The compatibility report records small, real engine smoke tests on the reference GTX 1660.
Those tests prove that the selected engine versions can execute minimal workloads; they do
not prove every GUI workflow, model architecture, dataset format, export strategy, or
long-running training configuration. See
[`docs/COMPATIBILITY_REPORT.md`](docs/COMPATIBILITY_REPORT.md) for the exact evidence
and its limitations.

### Previously recorded validation

The branch author recorded 123 passing tests and a clean Ruff run before the review repairs.
Those results are historical and must not be treated as validation of the current repair
branch until the Windows test and lint commands below have been rerun.


## Requirements

- Windows 10/11 64-bit.
- Python 3.11–3.13 (3.12.10 was used for the verified environments).
- NVIDIA GPU. Unsloth requires compute capability ≥ 7.0; Turing (7.5) has **no bfloat16**,
  so training uses float16.
- 6 GB VRAM is the design constraint: expect small models (≈0.5B–1.5B with 4-bit QLoRA) and
  short sequence lengths.

## Quick start (PowerShell)

```powershell
git clone https://github.com/ManicPanic82/Simple_hf_trainer.git
cd Simple_hf_trainer

py -3.12 -m venv environments\app
.\environments\app\Scripts\python.exe -m pip install -r requirements-dev.txt

.\launch.ps1   # Or double-click run.bat in Windows Explorer
```

The window opens even when Unsloth and Heretic are not installed; their status is reported
honestly in the header and tab cards. Engine setup is documented in
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Tests and linting

```powershell
.\environments\app\Scripts\python.exe -m pytest
.\environments\app\Scripts\python.exe -m ruff check .
```

The suite runs offscreen (`QT_QPA_PLATFORM=offscreen` is set by `tests/conftest.py`) and does
not download models or datasets.

## Repository layout

```text
main.py                     entry point
config/defaults.json        default settings (no secrets)
src/app/                    bootstrap, paths, logging, shared state
src/ui/                     main window, four tabs, widgets, theme
src/services/               settings, storage, hardware monitor, engines, credentials
src/workers/                protocol, QProcess worker runner, GPU-exclusive job manager
scripts/                    environment probe, engine verification, screenshots
docs/                       compatibility report, install and development notes
tests/                      pytest suite
```

Engine environments live under `environments/` and are **never** committed.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SIMPLE_HF_TRAINER_HOME` | Root for default models/datasets/outputs/hf-cache folders |
| `SIMPLE_HF_TRAINER_UNSLOTH_PYTHON` | Interpreter for the Unsloth environment |
| `SIMPLE_HF_TRAINER_HERETIC_PYTHON` | Interpreter for the Heretic environment |
| `HF_HOME` | Hugging Face cache location (honoured, not modified) |

## Known limitations

- Only the desktop foundation exists; the four workflows are still placeholders with clear
  labels.
- The job manager rejects a second engine job instead of queueing it.
- No model or dataset registry is persisted yet.
- GPU monitoring uses `nvidia-smi`; on machines without it the strip shows *Unavailable*
  instead of inventing values.
- Training/Heretic are invoked per-job in their own environments; those workflows enter the
  GUI in the next phase.

## Licence notes

Heretic is AGPL-3.0. This application installs and invokes it as a separate tool in its own
environment; it is not bundled or linked into this codebase.

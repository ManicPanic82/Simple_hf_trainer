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

This repository contains the complete implementation across all four workflows,
built Windows-first with an isolated architecture and exclusive GPU resource management.

| Area | Status |
| --- | --- |
| Four-tab PySide6 desktop shell | Implemented and tested |
| Shared hardware / job status strip | Implemented (degrades gracefully without a driver or psutil) |
| Settings, path validation, storage creation | Implemented and tested |
| Hugging Face token storage | Implemented and integrated with Windows Credential Manager |
| Worker protocol (JSON lines) | Implemented and tested, including malformed input |
| GPU-exclusive job manager | Implemented and tested (rejects conflicting engine, releases on crash/cancel) |
| Model search / download / library | Implemented and tested (HF Hub query, snapshot download, local model index, 6 GB VRAM compatibility assessment) |
| Dataset search / preview / preparation | Implemented and tested (HF Hub dataset query, sample preview, schema validation, Alpaca template preparation) |
| Unsloth training workflow in the GUI | Implemented and tested (QProcess worker, live steps, loss metrics, log stream, adapter save) |
| Heretic workflow in the GUI | Implemented and tested (QProcess worker with DummyOutput headless patch, live trials, refusal & KL metrics) |

See [`docs/COMPATIBILITY_REPORT.md`](docs/COMPATIBILITY_REPORT.md) for the evidence behind
engine claims, including what was actually executed on this machine.

### Verified on this machine (not just installed)

| Check | Result |
| --- | --- |
| GUI launch from the app environment | Window opens with exactly four tabs, live GPU/RAM status |
| `python -m pytest` | 123 passed (offscreen Qt) |
| `python -m ruff check .` | All checks passed |
| Unsloth 4-bit LoRA training (5 steps, Qwen2.5-0.5B) | Completed, loss 3.01 → 2.73, adapter saved, 0.58 GB peak VRAM |
| Heretic abliteration | Completed on Qwen2.5-0.5B (4-bit, 2 trials): refusals 5/5 → 4/5, KL 0.0093; merged model saved and generating |
| Windows Credential Manager token round-trip | Written, read back and deleted by the test suite |

Screenshots of each tab are in [`docs/evidence`](docs/evidence).

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

.\environments\app\Scripts\python.exe main.py
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

# Simple HF Trainer

A local-first **Windows desktop application** for six connected workflows:

1. **Models** — discover, download, and manage Hugging Face models.
2. **Datasets** — find, import, preview, and prepare Hugging Face datasets.
3. **Training** — fine-tune compatible models with the real **Unsloth** engine (LoRA/QLoRA).
4. **Analytics** — inspect live training/evaluation metrics, model architecture, hardware history and persisted job outcomes.
5. **Heretic** — modify compatible models with the real **Heretic 2.x** tool (ARA, abliteration, scorer plugins and presets).
6. **Ollama Export** — convert merged Hugging Face models to GGUF with llama.cpp, quantise them, and import/verify them in Ollama.

It is deliberately narrow: no chat, no RAG, no agents, no built-in inference server, no cloud
training, no plugin framework. The GPU (a 6 GB GTX 1660 in the reference setup) is treated
as the scarce resource it is.

## Current status

The `fix/restore-four-tab-contract` development branch contains the six-tab desktop
application and workflows for model discovery, dataset preparation, Unsloth fine-tuning,
live analytics, Heretic abliteration, and GGUF/Ollama export. This is a development branch, not a
release-ready claim: the initial project plan called for a desktop foundation first,
and the engine workflows on this branch are ahead of that original phase boundary.

| Area | Current implementation |
| --- | --- |
| Six-tab PySide6 desktop shell | Implemented; Windows validation pending |
| Analytics | Live loss/evaluation/LR/performance charts, hardware history, architecture view and persisted job history |
| Ollama Export | llama.cpp HF-to-GGUF conversion, optional quantisation, Ollama import and verification; requires user-configured tool paths |
| Hardware status, settings, storage paths | Implemented; hardware values degrade gracefully when probes are unavailable |
| Hugging Face token storage | Windows Credential Manager integration |
| Worker protocol and process runner | JSON-lines protocol, validation, background QProcess execution |
| GPU-exclusive job lifecycle | Centralised lock; completion is reconciled after the child process exits |
| Models | Hub search, background metadata lookup, download, local library and qualified memory estimates |
| Datasets | Hub search, local inspection, preview, preparation; JSON, JSONL, CSV, Parquet and Arrow IPC |
| Training | Unsloth LoRA/QLoRA worker with max-step or epoch-based runs, warmup, optimizer selection, checkpoint strategy/interval, optional evaluation dataset/strategy/interval, progress, metrics, cooperative cancellation and merged-16-bit or adapter-only export |
| Heretic | Full 2.x config UI (modifiers, scorers, presets, Raw TOML), headless worker with trial metrics and cancellation |

The compatibility report records small, real engine smoke tests on the reference GTX 1660.
Those tests prove that the selected engine versions can execute minimal workloads; they do
not prove every GUI workflow, model architecture, dataset format, export strategy, or
long-running training configuration. See
[`docs/COMPATIBILITY_REPORT.md`](docs/COMPATIBILITY_REPORT.md) for the exact evidence
and its limitations. The staged completion and Windows acceptance plan is in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

### Previously recorded validation

Validation is in progress on the repair branch. Historical test counts from earlier branches
are not evidence that this branch passes. Run the Windows test and lint commands below before
accepting or releasing a build.


## Requirements

- Windows 10/11 64-bit.
- Python 3.11–3.13 (3.12.10 was used for the verified environments).
- NVIDIA GPU. Unsloth requires compute capability ≥ 7.0; Turing (7.5) has **no bfloat16**,
  so training uses float16.
- 6 GB VRAM is the design constraint: expect small models (≈0.5B–1.5B with 4-bit QLoRA) and
  short sequence lengths.

## Quick start (PowerShell)

```powershell
git clone --branch fix/restore-four-tab-contract https://github.com/ManicPanic82/Simple_hf_trainer.git
cd Simple_hf_trainer

py -3.12 -m venv environments\app
.\environments\app\Scripts\python.exe -m pip install -r requirements-dev.txt

.\launch.ps1   # Or double-click run.bat in Windows Explorer
```

The window opens even when Unsloth and Heretic are not installed; their status is reported
honestly in the header and tab cards. Engine setup is documented in
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

### Heretic configuration

The Heretic tab exposes the real Heretic 2.x `config.toml` surface:

- **Presets** loaded from the bundled upstream files (`config/heretic_presets/`),
  plus a compact 6 GB-friendly starter profile.
- **Model & loading**: dtypes, 4-bit quantization, device map, memory limits,
  batch size, response length, response prefix and chain-of-thought blocks.
- **Objectives**: ARA or classic abliteration, any number of built-in or custom
  scorer plugins with `minimize` / `maximize` / `none`, and prompt sources.
- **Optimization / Export**: trials, seed, checkpoint behaviour, trial choice,
  merge vs adapter export, local save or Hugging Face upload.
- **Raw TOML**: refresh from the form, apply edits back to it, validate, and
  open/save `config.toml` files. Anything not covered by a widget (keyword
  marker lists, classifier settings, benchmark tasks, custom plugins) is edited here.

See [`docs/HERETIC_CONFIG.md`](docs/HERETIC_CONFIG.md) for details and limitations.

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
src/ui/                     main window, six tabs, widgets, theme
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

- The engine workflows have only been smoke-tested with small reference workloads; production
  runs, broad model-family coverage, large datasets and all export paths remain unverified.
- The GPU job manager runs one engine job at a time and queues further jobs FIFO; queued jobs start automatically when the GPU is free.
- VRAM assessments are rough indicators. Disk footprint is not GPU memory usage; actual
  requirements depend on quantization, context length, activations, offload settings and
  other GPU processes.
- Cancellation is cooperative first. Unsloth stops at a training-step boundary and Heretic
  receives an interrupt; an unresponsive worker is terminated after a grace period.
- Heretic's interactive-only post-run actions (chat and benchmarks) and special modes
  (reproduce, collect-reproducibles, evaluate-model) are not exposed in the GUI. Local
  save and Hugging Face upload are fully headless; uploads use the token stored in
  Windows Credential Manager via `HF_TOKEN`.
- Model and dataset libraries are catalogued in `registry.json`, and terminal job records
  are kept in `jobs.json` under the app home. Both are best-effort local caches: a corrupt
  file is ignored and the library is rebuilt by the next scan. In-flight job state is not
  restored after a restart.
- GPU monitoring uses `nvidia-smi`; without it the status strip reports *Unavailable*.


## Licence notes

Heretic is AGPL-3.0. This application installs and invokes it as a separate tool in its own
environment; it is not bundled or linked into this codebase.

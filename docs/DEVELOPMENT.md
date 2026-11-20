# Development

## Principles

1. **Nothing simulated.** A feature is either implemented and verified, or labelled as
   missing (`Not implemented yet`, `Compatibility testing required`, `Engine unavailable`).
   No fake progress bars, canned search results, or invented metrics.
2. **The GUI never touches heavy ML dependencies.** Unsloth and Heretic run in separate
   interpreters and separate processes.
3. **The GUI thread never blocks.** Monitoring and engine checks run on `QThreadPool`
   workers; engine jobs will run in `QProcess` workers.
4. **Small surface.** Exactly four tabs, one window, no speculative abstractions.

## Layout

```text
main.py                       adds src/ to sys.path, calls app.application.run()
config/defaults.json          committed defaults, ${APP_HOME}/${HF_HOME} placeholders
src/app/
    application.py            AppContext, create_application/create_window/run
    state.py                  AppState signals (hardware, engines, job, notice)
    paths.py                  per-user settings/log locations
    logging_setup.py          rotating file log + credential redaction
src/services/
    settings_service.py       JSON settings load/validate/atomic save
    storage_service.py        path validation + safe directory creation
    hardware_monitor.py       nvidia-smi + psutil probes, off-thread polling
    engine_registry.py        out-of-process engine interpreter/package detection
    credentials.py            Windows Credential Manager token storage
src/workers/
    protocol.py               JSON-lines message schema, strict decode
    worker_process.py         QProcess runner for one engine worker
    job_manager.py            GPU-exclusive lock and job lifecycle
src/ui/
    main_window.py            header, four tabs, notice banner, status strip
    models_tab.py datasets_tab.py training_tab.py heretic_tab.py
    tab_page.py components.py engine_card.py status_strip.py theme.py
    settings_dialog.py storage_action.py
```

`AppState` is the only channel between services and widgets: services emit, state
republishes, widgets subscribe. A widget never calls an engine or reads settings directly.

## Running

```powershell
.\environments\app\Scripts\python.exe main.py
```

Optional developer helpers:

```powershell
# Launch, save screenshots of every tab, exit
.\environments\app\Scripts\python.exe scripts\screenshot_app.py docs\evidence\ui-main-window.png

# Read-only platform evidence dump
py -3.12 scripts\collect_environment.py
```

## Tests and lint

```powershell
.\environments\app\Scripts\python.exe -m pytest
.\environments\app\Scripts\python.exe -m ruff check .
```

- `tests/conftest.py` forces `QT_QPA_PLATFORM=offscreen`, provides a session `QApplication`,
  isolated settings/storage fixtures, and a `wait_for` helper that spins the Qt event loop.
- No test downloads models or datasets; the only child processes are the fake worker and the
  engine-probe stubs.
- The Windows Credential Manager test writes a uniquely named credential and deletes it in a
  `finally` block.

## Worker protocol

One JSON object per line over stdin/stdout, encoded as UTF-8. Envelope:

```json
{"protocol": 1, "type": "progress", "job_id": "unsloth-3", "seq": 12,
 "payload": {"fraction": 0.42, "step": 21, "total": 50}}
```

| Type | Direction | Required payload | Notes |
| --- | --- | --- | --- |
| `hello` | worker → app | — | optional `role` |
| `ready` | worker → app | — | optional `engine`, `version` |
| `job_start` | app → worker | — | optional `model`, `config` |
| `progress` | worker → app | `fraction` 0..1 | optional `step`, `total` |
| `log` | worker → app | `message` | optional `level` in debug/info/warning/error |
| `metric` | worker → app | `name`, numeric `value` | optional `step` |
| `result` | worker → app | free-form object | e.g. output paths |
| `error` | worker → app | `message` | optional `recoverable` |
| `cancel` | app → worker | optional `reason` | worker should stop and report `job_complete` |
| `job_complete` | worker → app | — | optional `success`, `exit_code` |

Rules enforced by `workers/protocol.py`:

- `job_id` is required for every type except `hello`/`ready`.
- Unknown types, unknown protocol versions, non-object payloads, oversized lines (> 64 KiB),
  invalid JSON, and type-specific field errors raise `ProtocolError` with a stable `code`.
- Unknown payload keys are preserved (forward compatibility), so a newer worker does not break
  an older GUI.
- `WorkerProcess` turns malformed lines into a `protocol_error` signal and keeps reading;
  a hostile or broken worker cannot crash the window.

## GPU exclusivity

`workers/job_manager.py` owns a single `GpuLock`.

- `submit(engine)` acquires the lock or raises `GpuBusyError` naming the current owner. There
  is no queue yet; rejection is explicit and visible.
- `request_cancel(job_id)` moves the job to `cancelling` but **keeps** the lock until the
  worker process has actually exited.
- `mark_succeeded` / `mark_failed` / `mark_cancelled` release the lock.
- `handle_worker_exit(job_id, exit_code)` reconciles reality: cancellation wins, a non-zero
  exit is a failure, and exit code 0 **without** a completion message is also a failure
  ("exited without reporting completion"). It is safe to call twice and for unknown jobs.
- Terminal states cannot be overwritten by a late transition.

The UI only reflects this state; it cannot bypass the lock.

## Engine environments

| Environment | Contents | Why separate |
| --- | --- | --- |
| `environments/app` | PySide6, huggingface_hub, datasets, psutil, pytest, ruff | Must stay small and stable |
| `environments/unsloth` | torch+cu126, unsloth, triton-windows, bitsandbytes, xformers, trl, peft | Pins torch/transformers tightly |
| `environments/heretic` | torch+cu126, heretic-llm | Its own torch/transformers constraints |

Detection runs `python -c "<probe>"` in the engine interpreter, which reports
`{"found": bool, "version": str|null}`. The app never imports those packages itself.

## Adding the engine workflows (next boundaries)

1. Write `src/workers/unsloth_worker.py` that speaks the protocol and wraps the verified
   Unsloth training call (see `scripts/verify_unsloth_training.py`).
2. Launch it via `WorkerProcess(interpreter, script)` from a controller that first calls
   `JobManager.submit("unsloth")`; forward progress/log/metric messages into tab widgets.
3. On `exited`, always call `JobManager.handle_worker_exit` and surface the result.
4. Add registry records for produced adapters so Training/Heretic can reuse them.

Keep the worker scripts free of Qt: they must remain runnable by hand for debugging.

Engine invocation details that already bite:

- Unsloth writes an `unsloth_compiled_cache/` directory into the process working directory, so
  workers are started with an explicit `cwd` (`WorkerProcess(..., cwd=...)`) pointing at
  application-owned scratch space, never the repository root.
- Heretic has no `__main__` module. Launch
  `environments\heretic\Scripts\heretic.exe`, or
  `python -c "from heretic.main import main; main()"`. Drive it with
  `HERETIC_CONFIG_FILE` plus a TOML file that sets `export_strategy`, `checkpoint_action`,
  `trial_index`, `model_action` and `save_directory`; otherwise it blocks on prompts.
- Heretic builds its `questionary` prompts even for values that are already pinned
  (`ask_if_unset(settings.trial_index, questionary.select(...))`), and on Windows that
  constructs a `prompt_toolkit` `Win32Output`, which needs a real console screen buffer.
  With stdout on a pipe — i.e. any `QProcess` or `Tee-Object` — it raises
  `NoConsoleScreenBufferError` right after the trials finish. `scripts/run_heretic_unattended.py`
  patches the output factory to a `DummyOutput` (the prompt is never run) and then calls
  Heretic's own entry point; that is the shape the Heretic worker must take.
- Heretic 2.1.0 offers pytest-style CLI flags and a `--help` that lists every settings key,
  which is a cheap way to confirm a pin before wiring it up.

## Status and UI conventions

- `Pill` states: `ok`, `warn`, `error`, `info`, `off`.
- Every unfinished control is disabled **and** labelled; tooltips explain the blocker.
- Paths render through `StoragePathRow` (validation pill + open/create button).
- New colours belong in `ui/theme.py`, not inline in widgets.

## Known gaps

- No persistent model/dataset registry, no download manager, no HF search UI.
- No settings UI for the Hugging Face token (the secure store exists; the dialog is pending).
- Job manager has no queue and no restart/recovery across app restarts.
- Engine workers for Unsloth/Heretic are not written yet; only the protocol/process
  foundation exists and is tested.

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
- The GUI sends a protocol cancellation request asynchronously. Unsloth stops at the next
  training-step boundary; Heretic receives an interrupt and unwinds its cleanup path. If a
  worker does not exit within 10 seconds, the process runner requests termination and then
  kills it after a further 2 seconds. The GUI thread never waits synchronously for this path.
- The UI records `job_complete` as a pending result and only releases the GPU lock after the
  process-exited signal. A process that exits without a completion message is a failure.
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

## Engine workflows implementation

1. `src/workers/unsloth_worker.py` speaks the JSON-lines protocol and executes Unsloth
   fine-tuning (LoRA/QLoRA in 4-bit) with live step progress and loss metrics streaming.
2. `src/workers/heretic_worker.py` speaks the JSON-lines protocol and runs Heretic abliteration
   with the `DummyOutput` headless patch, streaming trial scores (Refusals, KL divergence).
3. Both workers are executed out-of-process via `WorkerProcess` with exclusive GPU access enforced
   by `JobManager`.
4. Resulting LoRA adapters and abliterated models are automatically scanned and indexed by
   `ModelService`, enabling seamless reuse across the tabs.

## Status and UI conventions

- `Pill` states: `ok`, `warn`, `error`, `info`, `off`.
- Paths render through `StoragePathRow` (validation pill + open/create button).
- Hardware and engines degrade gracefully when dependencies are absent.
- All four tabs communicate reactively through `AppState` signals.

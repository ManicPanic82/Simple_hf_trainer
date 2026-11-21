# Release notes

## Tag `v1.0.0`

`v1.0.0` is a lightweight tag pointing at commit `640b932` ("Add PowerShell launch
script"), 25 commits behind the current branch head (`0c280bb`). It marks the
one-click Windows launchers (`run.bat` and `launch.ps1`) and the foundational
four-tab PySide6 desktop app that existed at that commit:

- **One-click launch** — `run.bat` / `launch.ps1` start the app with the `environments/app` interpreter.
- **Four-tab desktop foundation** — Models, Datasets, Training and Heretic tabs, a shared `AppState`, settings/storage/hardware services, and the out-of-process worker protocol.

The tag is a historical marker for that foundation, **not** a claim that every
engine workflow was validated. It did not include a job queue, a persistent
model/dataset registry, or a custom download-progress widget.

## Current development state — `fix/desktop-foundation-review`

The development branch carries working implementations ahead of the original
"foundation first" phase boundary: model discovery and downloads, dataset
inspection/preparation/download, an Unsloth LoRA/QLoRA training worker, and a
headless Heretic abliteration worker, all executed out-of-process with a
GPU-exclusive job manager.

Validation at this point:

- **168 tests pass** and `ruff check` is clean (Windows, offscreen Qt; no test downloads models or datasets). Two pre-existing files are not Ruff-`format` clean.
- A real **Unsloth 5-step 4-bit LoRA training run** was validated end-to-end on the reference GTX 1660: Unsloth imported at 20.6 s, model loaded at 32.1 s, all 5 steps completed, adapter and tokenizer saved, `JOB_COMPLETE success=True`, zero protocol errors.
- A real **Heretic abliteration run** was validated through the same QProcess launcher: 2 trials, refusal/KL metrics, merged model saved with 6 artifacts, `JOB_COMPLETE success=True`, zero protocol errors.
- Both workers had the same Windows startup defect: a daemon stdin-cancellation listener blocked C-extension imports. The listener is now attached only after engine setup, immediately before the engine runs; regression tests guard the ordering. The Heretic worker also had a protocol-stream recursion defect (its interceptor re-emitted through the replaced `sys.stdout`), now fixed by capturing a private protocol stream.
- Completed LoRA adapters and abliterated models are published to the Models library automatically; the Models tab subscribes to `models_changed`.
- Dataset Hub download/import and honest per-file model-download progress with between-files cancellation were added. Compatibility checks are advisory only — unknown model/GPU combinations are never silently blocked.

## Known limitations

- **No job queue.** The GPU job manager owns a single lock and rejects a second engine job with `GpuBusyError`. There is no FIFO queue and no `JobState.QUEUED`; rejection is explicit and visible.
- **No persistent registry.** `src/services/engine_registry.py` performs out-of-process engine *detection* only. Discovered models and datasets are not catalogued, and there is no `registry.json`.
- **No persistent job history.** Job state is tracked in memory for the active lifecycle only; past jobs are not retained across restarts.
- **Cancellation during startup is limited.** To avoid the Windows import deadlock, the worker's stdin cancellation listener is attached after engine/model setup, immediately before training. Cancel requests issued during that startup/loading window are not observed by the worker until training begins; the process runner's grace-period termination is the backstop.
- **No GUI click-through acceptance run.** Both engines were validated through the same `WorkerProcess` launcher the tabs use, and unattended CLI/headless runs are recorded in the compatibility report. Manual clicking through the four tabs (including normal close and active-job close) has not been performed in this session, and long-running workloads remain unverified.
- **Engine evidence is smoke-test only.** Production runs, broad model-family and dataset-format coverage, and all export paths remain unverified; refer to `docs/COMPATIBILITY_REPORT.md` for the exact evidence levels.

# Coding Agent Prompt: Simple HF Trainer

You are the senior software engineer responsible for establishing the first working foundation of **Simple HF Trainer**.

Repository: https://github.com/ManicPanic82/Simple_hf_trainer  
Target branch: `feature/desktop-foundation`  
Default branch: `main`

Read this prompt fully before making changes. Work carefully, verify assumptions, and report evidence rather than claiming functionality that has not been tested.

---

## 1. Product goal

Build a **native Windows desktop application** that provides a straightforward, reliable interface for:

1. Discovering, downloading, inspecting, and managing Hugging Face models.
2. Discovering, downloading, importing, previewing, validating, and preparing Hugging Face datasets.
3. Fine-tuning compatible models using the **real Unsloth engine**, including supported LoRA/QLoRA workflows.
4. Modifying compatible models using the **real Heretic project** for abliteration.

Models and outputs should be reusable between the workflows. The application is local-first and Windows-first. The interface should make advanced functionality accessible without becoming cluttered or requiring users to work directly in command-line tools for routine tasks.

The product must remain deliberately focused. Do not add unrelated functionality or expand the product beyond the four workflows below.

## 2. Exactly four tabs

The main application must contain **exactly these four tabs**, in this order:

1. **Models**
2. **Datasets**
3. **Training**
4. **Heretic**

There must be no additional tabs. Shared application status, hardware information, settings access, notifications, and job state belong in the main window or appropriate existing tab, not in extra tabs.

### Models

Intended responsibilities:

- Search the Hugging Face Hub.
- Authenticate with Hugging Face securely when authentication is needed.
- Inspect model metadata and available files.
- Download models and show download progress, cancellation, and errors.
- Maintain a local model library and show local model locations.
- Reuse locally available models in Training and Heretic.
- Validate model paths and identify supported/unsupported formats where possible.

### Datasets

Intended responsibilities:

- Search Hugging Face datasets.
- Download or import local datasets.
- Preview dataset rows and available columns/splits.
- Validate dataset structure and report useful errors.
- Prepare supported data for training, with transparent configuration and output locations.
- Reuse prepared datasets in Training.

### Training

Intended responsibilities:

- Configure and launch real Unsloth fine-tuning jobs.
- Support LoRA/QLoRA only where the selected model, GPU, software versions, and Unsloth actually support the configuration.
- Expose useful training settings without overwhelming the normal workflow.
- Show job state, logs, progress, metrics, output location, and errors.
- Support cancellation and safe cleanup.
- Register completed outputs so they can be reused by the application.

### Heretic

Intended responsibilities:

- Select a compatible local or downloaded model.
- Configure and launch the real Heretic abliteration workflow.
- Show the actual engine output, progress, logs, and errors.
- Save results to a clearly identified output directory.
- Make resulting models available to the Models library and other compatible workflows.

Do not implement simulated versions of these features. Until a feature is genuinely implemented and verified, label it clearly as not implemented or awaiting compatibility work.

---

## 3. Explicit scope exclusions

Do **not** add any of the following:

- Chat interface or chatbot.
- RAG, document question answering, or vector databases.
- Agents or autonomous agent workflows.
- Inference server or model-serving platform.
- Reinforcement learning.
- Image generation.
- Cloud training or remote GPU execution.
- Multi-GPU orchestration.
- Plugin marketplace or plugin framework.
- An unrelated benchmarking platform.
- Extra tabs or unrelated dashboard sections.
- Any other feature that is not necessary for the four defined workflows.

Avoid feature creep. Prefer a small, maintainable foundation over a large collection of placeholders and abstractions.

---

## 4. Hardware and operating environment

The initial target machine is:

- OS: Windows 10 Pro, 64-bit.
- CPU: Intel Core i7-6700K.
- Memory: 32 GB RAM.
- GPU: NVIDIA GeForce GTX 1660.
- GPU memory: 6 GB VRAM.

The 6 GB VRAM limit is a critical design constraint. Do not imply that arbitrary Hugging Face models, quantization modes, Unsloth configurations, or Heretic workloads will fit.

Requirements:

- Windows is the primary platform.
- Use Windows-compatible paths and process handling.
- Document commands in **PowerShell**.
- Do not make Unix shell commands the primary setup or operating instructions.
- Do not automatically upgrade NVIDIA drivers, system CUDA, or unrelated system components.
- Avoid unnecessary dependency churn and system-wide package changes.
- Use isolated Python environments for the application and the ML engines.
- Detect missing hardware/software capabilities and explain them clearly.
- The GUI must remain usable when an optional engine environment is missing or incompatible.

Do not assume the local machine's current installed versions. Inspect and record actual versions during compatibility investigation.

---

## 5. Technology direction

Use the following technologies unless the compatibility investigation demonstrates a concrete blocker:

- **Python** for the application and orchestration.
- **PySide6 / Qt** for the native desktop GUI.
- **Hugging Face Hub** for model discovery, authentication, and downloads.
- **Hugging Face Datasets** for dataset discovery and dataset workflows.
- **Unsloth** for actual model fine-tuning.
- **Heretic**: https://github.com/p-e-w/heretic for actual abliteration/model modification.
- **PyTorch and CUDA** for GPU-backed workloads where supported.
- **pytest** for automated tests.
- **Ruff** for linting.

Keep the GUI/application environment separate from the Unsloth and Heretic environments. These engines may have conflicting Python or ML dependency requirements. Use separate worker processes and separate virtual environments rather than trying to force all dependencies into one environment.

Suggested environment layout:

```text
environments/
├── app/
├── unsloth/
└── heretic/
```

The exact environment setup must be based on verified compatibility, not guesswork.

### GPU exclusivity

Unsloth and Heretic must not load heavyweight models onto the GPU simultaneously.

Design a central job manager/resource lock that grants exclusive GPU ownership to one heavyweight engine job at a time. It must handle:

- A job already owning the GPU.
- Queued or rejected requests.
- Cancellation.
- Worker failure and process termination.
- Releasing the lock reliably.
- Preventing a second engine from starting while the first is still using GPU memory.

Do not rely on the UI alone to enforce this rule.

### Worker communication

Use a documented, structured protocol between the GUI and worker processes. JSON messages over standard input/output or another suitable local process protocol are acceptable.

The protocol should support, at minimum:

- Worker startup and readiness.
- Job identification.
- Progress and status updates.
- Log messages.
- Metrics or structured results.
- Error messages.
- Cancellation requests.
- Completion and exit status.

Validate incoming messages. Malformed or unexpected worker output must not crash the GUI.

### Configuration and credentials

- Use local JSON or TOML configuration for ordinary application settings.
- Do not introduce a database unless a demonstrated requirement makes one necessary.
- Store Hugging Face credentials in Windows Credential Manager or another suitable secure credential store.
- Never write access tokens or secrets in plaintext to settings JSON, logs, source files, or documentation.
- Ensure credentials are not included in exception messages or diagnostic reports.

---

## 6. Phase 0: compatibility investigation first

**Do this before attempting to build the complete application or installing large dependency stacks.**

The first deliverable is an evidence-based compatibility report at:

`docs/COMPATIBILITY_REPORT.md`

Investigate the actual Windows/GPU environment and the current official requirements and capabilities of both engines.

### 6.1 Inspect the machine

Record, where available:

- Windows edition and build.
- Python versions.
- NVIDIA GPU model.
- GPU compute capability.
- Installed NVIDIA driver.
- CUDA runtime/toolkit information.
- PyTorch version and CUDA availability.
- Total and available VRAM.
- Relevant installed packages and their versions.

Do not change drivers, system CUDA, or other global settings as part of this investigation.

### 6.2 Unsloth investigation

Use official Unsloth documentation and project information to establish:

- Current Windows support status and supported installation method.
- Supported Python, PyTorch, CUDA, and dependency combinations.
- Whether the GTX 1660 architecture is supported by the relevant stack.
- Whether the GPU's compute capability is sufficient for required kernels and quantization paths.
- Whether 4-bit quantization and QLoRA are supported on this hardware/software combination.
- Which model architectures are supported.
- Practical memory constraints on a 6 GB GPU.
- Any known Windows limitations or workarounds.

Do not assume that a package installing successfully means the engine works.

Where practical and safe, create an isolated Unsloth environment and run a **real, minimal training workload** using a suitably small compatible model/dataset. Record the exact configuration and outcome. Do not download huge models or run workloads likely to exhaust system resources.

If a real training test is not feasible, say exactly why and identify what remains unverified.

### 6.3 Heretic investigation

Use the official project at https://github.com/p-e-w/heretic and its current documentation/source to establish:

- Windows support and installation requirements.
- Supported Python, PyTorch, CUDA, and dependency combinations.
- Supported model architectures and formats.
- Whether the GTX 1660 and its 6 GB VRAM are sufficient for any useful supported workflow.
- Quantization support and limitations.
- Expected memory requirements and practical model-size constraints.
- Actual command-line/API invocation, configuration, input requirements, and output format.
- Whether the tool can be run reliably in an isolated worker process on this Windows machine.

Where practical and safe, run a **real minimal Heretic model-modification workflow** using a suitably small compatible model. Verify that an output artifact is actually produced and can be inspected. An import test or successful installation is not proof of working model modification.

If the workload cannot be run, record the precise blocker and leave the capability unverified.

### 6.4 Report evidence honestly

For each finding, distinguish clearly between:

- **Verified working**: tested on the target environment with an actual workload.
- **Installed but untested**: installation/import succeeded, but the actual workload was not run.
- **Documented/theoretical**: supported by upstream documentation, but not tested locally.
- **Unsupported**: evidence shows the configuration is not supported.
- **Uninvestigated**: not enough evidence yet.

Include:

- Exact versions and commands used.
- Actual test configuration.
- Relevant console output or concise evidence.
- Errors and blockers.
- Memory/VRAM observations where available.
- Links to authoritative upstream documentation.
- Recommended known-good isolated environment pins, if established.

Never fabricate successful tests, performance numbers, support claims, or compatibility.

Do not proceed by silently relaxing requirements when an engine is incompatible. Document the limitation and keep the architecture honest.

---

## 7. Phase 1: desktop foundation

After the compatibility investigation, implement only the desktop foundation. Do not attempt to build the complete model, dataset, training, or Heretic workflows in this phase.

### 7.1 Main application

Create a working PySide6 application with:

- A professional, restrained dark interface.
- Exactly four tabs in the specified order.
- A compact, coherent layout suitable for a desktop workstation.
- Sensible resizing and responsive layouts.
- Clear status and error presentation.
- No blocking long-running work on the GUI thread.

The application should launch even if Unsloth and Heretic are not installed in the application environment.

### 7.2 Shared status strip

Include a compact status area in the main window showing available information such as:

- GPU name.
- GPU utilization.
- VRAM usage/total.
- System RAM usage.
- Active job and engine.
- GPU availability/ownership state.

Hardware APIs may be unavailable. Handle this gracefully and show an unavailable/unknown state instead of crashing or inventing values.

This is not a fifth tab.

### 7.3 Four real tab shells

Create the four actual navigable tabs:

- Models
- Datasets
- Training
- Heretic

Each should have a sensible initial layout and concise explanation of its intended role. Clearly mark unfinished operations with states such as:

- `Not implemented yet`
- `Compatibility testing required`
- `Engine unavailable`
- `Not supported by detected hardware`

Do not show fake progress bars, fake downloads, fake model cards presented as live search results, simulated training metrics, or pretend Heretic output.

### 7.4 Settings and storage

Implement:

- Persistent application settings.
- Sensible defaults.
- Validated user-configurable storage paths for models, datasets, and outputs.
- Clear handling of missing, invalid, or inaccessible directories.
- Safe creation of application-owned directories where appropriate.
- No destructive cleanup of user data.

Keep settings local and do not store secrets in plaintext.

### 7.5 Hardware monitor

Implement a small hardware-monitoring service with a clean interface. It should report available GPU/RAM information and degrade gracefully when optional monitoring libraries or APIs are missing.

Do not make the GUI depend on a specific optional hardware-monitoring package being installed.

### 7.6 Worker protocol foundation

Implement and test the structured worker-message protocol, including parsing, validation, malformed messages, and error handling.

Do not implement the full Unsloth or Heretic workers in this phase.

### 7.7 Job manager foundation

Define and test the job manager/resource-lock interface needed for exclusive GPU use. Implement only the foundation required for safe orchestration; do not build a large speculative scheduling framework.

---

## 8. Suggested starting project structure

Use this as a guide, not as a requirement to create unused files or abstractions. Keep the actual structure proportionate to the code.

```text
Simple_hf_trainer/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── defaults.json
│   └── settings.json
├── docs/
│   ├── COMPATIBILITY_REPORT.md
│   ├── INSTALLATION.md
│   └── DEVELOPMENT.md
├── src/
│   ├── app/
│   │   ├── application.py
│   │   └── state.py
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── models_tab.py
│   │   ├── datasets_tab.py
│   │   ├── training_tab.py
│   │   └── heretic_tab.py
│   ├── services/
│   │   ├── hardware_monitor.py
│   │   ├── settings_service.py
│   │   └── storage_service.py
│   └── workers/
│       └── protocol.py
├── environments/
│   ├── app/
│   ├── unsloth/
│   └── heretic/
└── tests/
```

Do not commit virtual environments, downloaded model weights, datasets, caches, credentials, or generated training artifacts.

---

## 9. Testing requirements

Add meaningful automated tests for:

- Application startup.
- Exactly four tabs and their order/names.
- Tab navigation.
- Settings persistence and invalid settings handling.
- Storage path validation and failure cases.
- Hardware detection when available.
- Graceful hardware-monitor failure/unavailability.
- Worker protocol valid-message parsing.
- Worker protocol malformed/unknown-message handling.
- Missing optional engine dependencies.
- Job manager GPU exclusivity and lock release behavior, to the extent implemented.

Tests must not require downloading large models or datasets. Keep ordinary tests deterministic and suitable for local Windows execution.

Use pytest and Ruff.

Expected local commands:

```powershell
python -m pytest
python -m ruff check .
python .\main.py
```

If a project virtual environment is used, document its activation and use its Python executable consistently.

Do not rely on GitHub Actions for validation. The user will run tests locally.

---

## 10. Documentation

Create or update:

### `README.md`

Explain:

- What the application is.
- The four workflows.
- Current implementation status.
- Supported/target platform.
- Quick start.
- Known hardware and compatibility limitations.

### `docs/COMPATIBILITY_REPORT.md`

Provide the Phase 0 evidence-based compatibility findings described above.

### `docs/INSTALLATION.md`

Provide Windows PowerShell-first installation and setup instructions, including isolated environments and how to install the application dependencies without modifying system Python unnecessarily.

### `docs/DEVELOPMENT.md`

Explain:

- Project structure.
- How to run the application.
- How to run tests and Ruff.
- Worker protocol basics.
- How engine environments are separated.
- Current limitations and next implementation boundaries.

Do not document functionality as available until it actually works.

---

## 11. Git and change-control rules

These rules are mandatory:

1. Before changing anything, inspect the current branch, working tree, and latest commit.
2. Work only on `feature/desktop-foundation`.
3. If that branch already exists, inspect it and its contents before using it. Do not overwrite or reset it.
4. Never commit changes to `main`.
5. Never merge into `main`.
6. Never force-push.
7. Never delete branches.
8. Never rewrite history or perform destructive Git operations.
9. Do not push until the user explicitly authorizes pushing.
10. Keep commits small, descriptive, and reviewable.
11. Do not commit secrets, local settings containing credentials, virtual environments, model weights, datasets, caches, or generated artifacts.
12. If the repository state prevents safe work on the requested feature branch, stop and explain the blocker rather than changing `main` to work around it.

The user does not want GitHub Actions used for validation. Run the available checks locally and report their results.

---

## 12. Work sequence

Follow this order:

1. Inspect repository and Git state.
2. Inspect the actual Windows/Python/GPU environment.
3. Research current official Unsloth and Heretic compatibility.
4. Perform safe, minimal real engine tests where feasible.
5. Write `docs/COMPATIBILITY_REPORT.md` with evidence and unresolved questions.
6. Establish the isolated environment strategy and document it.
7. Implement the PySide6 desktop foundation and exactly four tabs.
8. Implement settings, storage validation, hardware-monitor fallback, worker protocol, and job-manager foundation.
9. Add automated tests.
10. Run pytest and Ruff locally.
11. Run the application and verify the four-tab UI on Windows.
12. Review the complete diff for scope, secrets, generated files, and accidental changes.
13. Report the results. Do not push unless the user has explicitly authorized it.

Do not skip compatibility investigation and jump straight into installing every ML dependency.

---

## 13. Completion criteria for this task

This initial task is complete only when:

- The application launches on the target Windows environment.
- The main window contains exactly four tabs in the correct order.
- The GUI remains responsive.
- Shared hardware/job status is displayed when information is available and degrades gracefully otherwise.
- Settings persist and storage paths are validated.
- Missing optional ML engines do not prevent the GUI from starting.
- The worker protocol foundation is implemented and tested.
- The GPU-exclusive job-manager foundation is present and tested to the extent implemented.
- The compatibility report distinguishes verified results from assumptions and untested claims.
- Windows setup and development instructions are documented.
- pytest passes.
- Ruff passes.
- The Git diff contains only intentional, in-scope files.
- No changes have been made to `main`.

Do not claim completion if any criterion is not met. List incomplete criteria and blockers plainly.

---

## 14. Final report format

When finished, provide a concise but detailed engineering report containing:

1. **Repository state**
   - Current branch.
   - Starting and final commit hashes.
   - Whether the working tree is clean.
   - Whether anything was pushed (normally no, unless explicitly authorized).

2. **Compatibility findings**
   - Windows/Python/PyTorch/CUDA/GPU details.
   - Unsloth findings and actual test outcome.
   - Heretic findings and actual test outcome.
   - Practical GTX 1660 6 GB limitations.
   - Verified vs untested vs unsupported items.

3. **Implementation**
   - What was actually built.
   - Confirmation of exactly four tabs.
   - Worker and job-manager foundation status.

4. **Files changed**
   - List the important added/modified files and their purpose.

5. **Validation**
   - Exact pytest command and result.
   - Exact Ruff command and result.
   - Application launch/manual verification result.

6. **Outstanding issues**
   - Known bugs, unsupported configurations, incomplete work, and next safe steps.

Be precise. Do not claim a command ran unless it did. Do not claim an engine works unless a real workload was completed and its output verified.

**Priority:** a small, honest, tested Windows desktop foundation and a trustworthy compatibility report. Do not build beyond Phase 0 and Phase 1 in this task.
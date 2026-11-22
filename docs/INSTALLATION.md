# Installation (Windows / PowerShell)

Everything here is PowerShell-first and uses **isolated environments**. Nothing is installed
into the system Python, and no driver, CUDA toolkit, or global package is modified.

Verified on: Windows 10 Pro 19045, Python 3.12.10, NVIDIA GTX 1660 6 GB, driver 617.42.
Exact package versions from that run are in
[`COMPATIBILITY_REPORT.md`](COMPATIBILITY_REPORT.md).

---

## 1. Prerequisites

- **Windows 10 or 11, 64-bit.**
- **Python 3.11–3.13** from python.org with the `py` launcher available. Check:

  ```powershell
  py --list
  ```

- **Git**:

  ```powershell
  git --version
  ```

- **NVIDIA driver** already installed and working. Confirm the GPU is visible:

  ```powershell
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  ```

  Expected on the reference machine: `NVIDIA GeForce GTX 1660, 617.42, 6144 MiB, 7.5`.

- Do **not** upgrade the NVIDIA driver or system CUDA as part of installing this app. PyTorch
  wheels carry their own CUDA runtime; the driver only needs to be recent enough, which it
  already is if `nvidia-smi` works.

---

## 2. Get the repository

```powershell
git clone --branch fix/restore-four-tab-contract https://github.com/ManicPanic82/Simple_hf_trainer.git
cd Simple_hf_trainer
```

The application source currently lives on the development branch shown above. The repository's
`main` branch is not the application checkout; do not use a plain default-branch clone for this
validation.

## 3. Application environment

The GUI environment contains PySide6 and the Hugging Face client libraries only. It must not
contain torch or the engines.

```powershell
py -3.12 -m venv environments\app
.\environments\app\Scripts\python.exe -m pip install --upgrade pip
.\environments\app\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` adds pytest and ruff to the runtime requirements.

Verify:

```powershell
.\environments\app\Scripts\python.exe -c "import PySide6, huggingface_hub, psutil; print('app env ok')"
```

## 4. Run the application

```powershell
.\environments\app\Scripts\python.exe main.py
```

Or activate the environment first:

```powershell
.\environments\app\Scripts\Activate.ps1
python main.py
python -m pytest
python -m ruff check .
```

The application starts without any engine installed. Missing engines are reported as
*not set up* / *not installed*; they never block startup.

---

## 5. Unsloth environment (optional, only for training)

Unsloth lives in `environments\unsloth`. The versions below are the ones verified to train on
this GTX 1660. Install torch from the CUDA 12.6 wheel index first so pip cannot pull a CPU
build.

```powershell
py -3.12 -m venv environments\unsloth
.\environments\unsloth\Scripts\python.exe -m pip install --upgrade pip
.\environments\unsloth\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
.\environments\unsloth\Scripts\python.exe -m pip install unsloth
```

Verify CUDA and a real (tiny) training run:

```powershell
.\environments\unsloth\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

$env:SHT_UNSLOTH_OUT = "docs\evidence\unsloth-lora-run"
.\environments\unsloth\Scripts\python.exe scripts\verify_unsloth_training.py
```

The verification script downloads `unsloth/Qwen2.5-0.5B-Instruct` (about 1 GB), trains five
LoRA steps with 4-bit weights, saves the adapter, and prints a JSON summary. Delete the cache
and output folder afterwards if you do not want to keep them — the application never deletes
user data itself.

Notes:

- Turing (compute 7.5) has no bfloat16. `dtype=None` makes Unsloth pick float16; do not force
  `bf16=True`.
- Unsloth requires compute capability ≥ 7.0. Pre-Volta cards fall back to eager mode and are
  slow; older cards are not supported.
- Use `SHT_UNSLOTH_STEPS` / `SHT_UNSLOTH_SEQ` to make the check even smaller.

## 6. Heretic environment (optional, only for model modification)

Heretic lives in `environments\heretic`. It needs Python 3.10+ and PyTorch 2.6+.

```powershell
py -3.12 -m venv environments\heretic
.\environments\heretic\Scripts\python.exe -m pip install --upgrade pip
.\environments\heretic\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\environments\heretic\Scripts\python.exe -m pip install heretic-llm
```

Verify:

```powershell
.\environments\heretic\Scripts\heretic.exe --help
```

`python -m heretic` does **not** work: `heretic-llm` ships only a console entry point
(`heretic = "heretic.main:main"`) and has no `__main__` module. Always call the `heretic.exe`
in the environment, or run
`python -c "from heretic.main import main; main()"`.

Heretic is interactive by design (it asks what to do with the modified model). For an
unattended run, pin every prompt in a TOML file and select it with `HERETIC_CONFIG_FILE` —
`config/heretic-verify.toml` in this repository is a working example:

```powershell
$env:HERETIC_CONFIG_FILE = (Resolve-Path config\heretic-verify.toml).Path
.\environments\heretic\Scripts\heretic.exe
```

The desktop application includes a Heretic workflow that launches the configured isolated
interpreter through its headless wrapper. The direct compatibility smoke test above proves the
engine can modify and export a small model; it does not replace acceptance testing through the
graphical workflow.

## 7. Runtime acceptance through the desktop app

After both engine environments are installed and detected, launch the app and perform these
checks with the small reference model `Qwen/Qwen2.5-0.5B-Instruct`:

1. In **Training**, select a small local JSONL dataset, choose a short run (5 steps, batch 1,
   sequence length 128 or 256), and save an adapter to a new output folder.
2. Confirm the console shows the configured Unsloth interpreter and worker script, progress
   advances, the run reaches a terminal success state, and adapter files exist in the output.
3. In **Heretic**, load the compact verification configuration, keep the run to two trials and
   short prompt slices, and choose a new output folder. Do not point it at an existing model
   directory.
4. Confirm the Heretic console shows the isolated interpreter/wrapper, trial progress appears,
   and the selected trial is exported. Confirm the output contains a model config and weight
   files.
5. While one engine is running, queue a job for the other engine. Confirm the second job waits
   and does not start until the first process exits and releases GPU ownership.
6. Cancel a short run. Confirm the job reaches Cancelled (or Failed if the engine exits
   abnormally), the worker process exits, and the queued job can then acquire the GPU.

Treat this as a destructive boundary: use disposable output folders, never overwrite a model
or dataset, and do not test upload unless you explicitly intend to publish a model.

Record the model, dataset, settings, engine versions, exit status, and output paths when
reporting acceptance. A passing unit suite does not substitute for these real-process checks.

## 8. Pointing the application at the engines

Two equivalent options:

1. **Settings dialog** — *Settings* button → *Unsloth python.exe* / *Heretic python.exe* →
   browse to `environments\unsloth\Scripts\python.exe` and
   `environments\heretic\Scripts\python.exe`. The paths are stored in
   `%LOCALAPPDATA%\SimpleHFTrainer\settings.json`.
2. **Environment variables**:

   ```powershell
   $env:SIMPLE_HF_TRAINER_UNSLOTH_PYTHON = "F:\path\to\Simple_hf_trainer\environments\unsloth\Scripts\python.exe"
   $env:SIMPLE_HF_TRAINER_HERETIC_PYTHON = "F:\path\to\Simple_hf_trainer\environments\heretic\Scripts\python.exe"
   ```

If both are unset, the app looks for `environments\<engine>\Scripts\python.exe` inside the
repository.

Detection is metadata-only (`importlib.util.find_spec` in the engine interpreter). It never
imports torch into the GUI process.

---

## 9. Storage

Defaults (override with the `SIMPLE_HF_TRAINER_HOME` environment variable):

| Purpose | Default |
| --- | --- |
| Models | `%USERPROFILE%\SimpleHFTrainer\models` |
| Datasets | `%USERPROFILE%\SimpleHFTrainer\datasets` |
| Outputs | `%USERPROFILE%\SimpleHFTrainer\outputs` |
| HF cache | `%HF_HOME%`, or `%USERPROFILE%\SimpleHFTrainer\hf-cache` |

Settings live in `%LOCALAPPDATA%\SimpleHFTrainer\settings.json`; logs in
`%LOCALAPPDATA%\SimpleHFTrainer\logs\simple-hf-trainer.log`.

Storage paths must be absolute. On save, the app validates them, offers to create missing
folders, and reports unwritable or file-in-the-way paths. It never deletes or moves data.

## 10. Credentials

Hugging Face tokens are stored in **Windows Credential Manager** under
`SimpleHFTrainer/HuggingFace`. They are never written to `settings.json`, logs, or the
repository. A `HF_TOKEN` environment variable is honoured by the Hugging Face libraries
themselves; the app does not read or store it in plaintext.

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Unsloth · not set up` | Interpreter path wrong, or the venv was not created. Run the probe: `.\environments\unsloth\Scripts\python.exe -c "import unsloth; print(unsloth.__version__)"` |
| `Unsloth · check failed` | The interpreter starts but cannot import the package; read the tooltip/detail, usually a broken dependency install. |
| GPU shows *Unavailable* | `nvidia-smi` missing or driver broken. The rest of the app still works. |
| CUDA out of memory on 6 GB | Use a smaller model, 4-bit loading, shorter sequence length, batch size 1, and close other GPU consumers (browsers, Visual Studio). |
| `torch.cuda.is_available() == False` | The wrong torch wheel was installed (CPU build). Reinstall with `--index-url https://download.pytorch.org/whl/cu126`. |
| Long path errors | Keep the repository path short, e.g. `C:\dev\Simple_hf_trainer`, and avoid deep output folders. |
| Two engines at once | Not possible by design: the job manager owns a single GPU lock and rejects the second request. |

## 12. Uninstalling / cleaning up

The application never deletes user data. To remove everything manually:

```powershell
Remove-Item -Recurse -Force environments
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\SimpleHFTrainer"
Remove-Item -Recurse -Force "$env:USERPROFILE\SimpleHFTrainer"
```

Only do this if you do not need the downloaded models or training outputs.

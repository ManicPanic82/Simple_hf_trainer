# Compatibility report (Phase 0)

Evidence-based investigation of **Unsloth** and **Heretic** on the reference Windows machine,
performed before building the desktop foundation.

Every finding below is tagged with one of the levels the task requires:

| Tag | Meaning |
| --- | --- |
| **Verified** | A real workload was executed on this machine and its output was inspected. |
| **Installed but untested** | Installation/import succeeded; the real workload was not run. |
| **Documented** | Supported by upstream documentation/source, not exercised locally. |
| **Unsupported** | Evidence shows the combination does not work. |
| **Uninvestigated** | Not enough evidence collected. |

Nothing in this report is inferred from "the package installed, therefore it works".

Environment evidence dump: [`evidence/environment-py312.txt`](evidence/environment-py312.txt)
(produced by `scripts/collect_environment.py`).

---

## 1. Machine

| Item | Value | Evidence |
| --- | --- | --- |
| OS | Microsoft Windows 10 Pro, version 10.0.19045, build 19045 (64-bit) | `Win32_OperatingSystem` |
| CPU | Intel Core i7-6700K @ 4.00 GHz, 4 cores / 8 threads | `Win32_Processor` |
| RAM | 31.94 GB total, 19.76 GB free at probe time | `Win32_ComputerSystem` |
| GPU | NVIDIA GeForce GTX 1660, WDDM, 6144 MiB | `nvidia-smi` |
| Compute capability | **7.5** (Turing) | `nvidia-smi --query-gpu=compute_cap`, `torch.cuda.get_device_capability` |
| Driver | 617.42 (KMD), CUDA UMD 13.4, `nvidia-smi` 621.24 | `nvidia-smi` |
| System CUDA toolkit | 13.3 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`, `nvcc` V13.3.73) | `nvcc --version` |
| Python | 3.11.9 and 3.12.10 (both 64-bit, `py` launcher) | interpreter probe |
| Free disk | C: 96.2 GB, D: 563.8 GB, E: 442.5 GB, F: 1627.9 GB | `Get-PSDrive` |

No driver, system CUDA installation, or global Python package was modified during this
investigation. All work happened in throw-away virtual environments under `environments/`.

### PyTorch / CUDA

The isolated Unsloth environment installed `torch 2.14.1+cu126` and reports:

```text
torch: 2.14.1+cu126
torch.version.cuda: 12.6
torch.cuda.is_available(): True
device: NVIDIA GeForce GTX 1660
capability: (7, 5)
vram_total: 6.0 GB
```

**Verified**: a CUDA 12.6 PyTorch wheel runs on this driver/GPU without installing a system
CUDA runtime. The pre-existing CUDA 13.3 toolkit is irrelevant to the wheels; nothing needs to
be downgraded.

### bfloat16 nuance (matters for training defaults)

`torch.cuda.is_bf16_supported()` returns **True** on this machine with torch 2.14.1, and
`torch.randn(..., dtype=torch.bfloat16, device="cuda") @ ...` executes. That is not evidence of
Ampere-class bfloat16 hardware: compute capability 7.5 has no native bf16 matrix units, and
current PyTorch implements these operations through conversion/emulation.

Unsloth's own startup banner is explicit about the same hardware:

```text
NVIDIA GeForce GTX 1660. Num GPUs = 1. Max memory: 6.0 GB. Platform: Windows.
Torch: 2.14.1+cu126. CUDA: 7.5. CUDA Toolkit: 12.6. Triton: 3.9.0
Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
```

**Conclusion**: treat bf16 as unavailable on this GPU and use fp16. `is_bf16_supported()` alone
must not be used to decide training precision.

---

## 2. Unsloth

### 2.1 Upstream facts (Documented)

Sources: <https://github.com/unslothai/unsloth>, <https://docs.unsloth.ai>,
<https://docs.unsloth.ai/get-started/install-and-update/windows-installation>.

- Windows is supported natively (no WSL required); the documented path is a virtual
  environment plus `pip install unsloth`.
- Unsloth requires an NVIDIA GPU with compute capability **≥ 7.0** (Volta or newer). The
  GTX 1660 at 7.5 is inside that range.
- The Unsloth matrix prescribes a specific `torch`/`xformers`/`triton` combination per CUDA
  version; installing a newer PyTorch than the one Unsloth was built against degrades some
  fast paths but is not fatal.
- 4-bit quantization and QLoRA are supported on Turing (bitsandbytes 4-bit path).
- Supported model families include Llama, Qwen (2/2.5/3), Gemma, Mistral, Phi, DeepSeek and
  many others; the GUI must not promise a model family that upstream does not patch.

### 2.2 What was installed here

`environments/unsloth` (Python 3.12.10), created for this investigation:

```powershell
py -3.12 -m venv environments\unsloth
.\environments\unsloth\Scripts\python.exe -m pip install -U pip
.\environments\unsloth\Scripts\python.exe -m pip install unsloth
```

pip resolved and installed (excerpt, full log kept locally):

```text
torch 2.14.1+cu126          unsloth 2026.11.7        unsloth_zoo 2026.11.7
transformers 5.17.0         trl 1.15.0               peft 0.22.0
bitsandbytes 0.50.2         triton-windows 3.9.0.post30
xformers 0.0.36             datasets 4.8.5           huggingface_hub 1.33.0
```

Install exit code: `0`.

Observed at runtime (not fatal):

```text
Unsloth: xformers is installed but cannot load with torch 2.14.1+cu126,
so PyTorch attention is used instead.
```

So the xformers dependency version pulled by pip does not match this torch build. Training
still works; attention falls back to PyTorch's implementation. This is a real performance
limitation on this machine, not a correctness problem.

### 2.3 Real minimal training run — **Verified**

Script: `scripts/verify_unsloth_training.py` (kept in the repository; also used as the
reference for the future training worker).

Configuration actually executed:

| Setting | Value |
| --- | --- |
| Model | `unsloth/Qwen2.5-0.5B-Instruct` (4-bit, `dtype=None`) |
| LoRA | r=8, alpha=16, dropout=0, gradient checkpointing `unsloth` |
| Dataset | 64 synthetic arithmetic prompt/completion rows (no download beyond the model) |
| Sequence length | 256 |
| Steps | 5 (`max_steps`) |
| Batch | per-device 1, grad-accum 1, optimizer `adamw_8bit` |
| Precision | `fp16=True`, `bf16=False` |
| Seed | 42 |

Result (JSON printed by the script, exit code `0`):

```text
"torch": "2.14.1+cu126", "cuda_available": true,
"gpu": "NVIDIA GeForce GTX 1660", "capability": [7, 5],
"unsloth": "2026.11.7",
"load_seconds": 89.1, "train_seconds": 7.0, "train_loss": 2.856731986999512,
"peak_vram_gb": 0.58,
"artifacts": ["README.md", "adapter_config.json", "adapter_model.safetensors",
              "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "trainer-state"],
"sample_output": "### Question: What is 21 + 21?\n### Answer: 42\n\n# This code will",
"status": "verified", "total_seconds": 102.9
```

Training loss over the five steps: 3.012 → 2.774 → 2.911 → 2.852 → 2.734
(`train_loss` average 2.857). Console excerpt preserved in
[`evidence/unsloth-training-console.txt`](evidence/unsloth-training-console.txt).

What this proves:

- CUDA training works on the GTX 1660 through Unsloth on Windows.
- 4-bit loading + LoRA + 8-bit optimizer works on 6 GB.
- An adapter artifact is produced (`adapter_model.safetensors`, 4.4 M trainable parameters,
  0.88 % of the model) and can be reloaded for generation.
- Peak allocated VRAM for this configuration was **0.58 GB**, i.e. VRAM is not the binding
  constraint for ~0.5B work; it will be for 7B+ models.

What this does not prove:

- No real user dataset was trained on (synthetic rows only, 5 steps). Loss behaviour on a real
  corpus and for many steps is unverified.
- No throughput claim: 5 steps took 7 s after a 89 s first-run load; that is not a benchmark.
- No evaluation, merging, GGUF export, or multi-GPU path was exercised.
- `unsloth install-kernels xformers` was **not** run, so the "matching kernels" fast path is
  unverified on this machine.

### 2.4 Practical 6 GB constraints

- Adapters keep optimizer state small, but activations scale with
  `batch × seq_len × hidden`. The 0.5B run used 0.58 GB; expect roughly linear growth with
  parameter count for the 4-bit weights plus growing activation memory with sequence length.
- 7B models in 4-bit (~4.5 GB of weights) leave very little room for activations on a 6 GB
  card; 1.5B–3B is the realistic range at short sequence lengths. This extrapolation is
  **Documented/deduced**, not measured here.
- Turing has no bf16 and no FlashAttention-2: training is slower per step than on Ampere.
- Chrome, the NVIDIA app and the desktop compositor already occupy ~0.8 GB of VRAM on this
  machine at idle (see `nvidia-smi` in the environment dump), which reduces headroom further.

---

## 3. Heretic

### 3.1 Upstream facts (Documented)

Sources: <https://github.com/p-e-w/heretic> (`pyproject.toml`, `src/heretic/config.py`,
`src/heretic/main.py`, `README.md`).

- Package name `heretic-llm`, licence **AGPL-3.0-or-later**, `requires-python >= 3.10`.
- Runtime dependencies are pinned tightly by the project, e.g. `accelerate~=1.15`,
  `bitsandbytes~=0.50`, `datasets~=5.0`, `huggingface-hub~=1.33`, `peft~=0.21`,
  `optuna~=5.0`, `lm-eval[hf]~=0.4`, `transformers~=4.57`-era pins, plus `torch`.
- It is a console application (`heretic = "heretic.main:main"`), configured by pydantic
  settings with three sources: CLI flags (kebab-case), environment variables prefixed
  `HERETIC_`, and a TOML file selected by `HERETIC_CONFIG_FILE` (default `config.toml`).
- Workflow: load model → capture residual directions from "good" vs "bad" prompt sets →
  optimize modification weights with Optuna → score trials → export.
- Interactive prompts would block a GUI worker. Every prompt has a settings key:
  `export_strategy` (`merge`/`adapter`), `checkpoint_action`, `trial_index`, `model_action`
  (`save`/`upload`/`chat`), `save_directory`. Setting all of them makes a run unattended.
- Two modifiers exist: `heretic.modifiers.abliteration.Abliteration` (classic abliteration)
  and `heretic.modifiers.ara.ARA` (default in the current version).
- Quantization options are `none` and `bnb_4bit`; `dtypes` defaults to
  `["auto", "float16", "bfloat16", "float32"]`, so it degrades from bf16 to fp16 on pre-Ampere
  hardware by itself.
- Prompt data defaults to `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors` from the
  Hugging Face Hub; the project's own tests pin commits and use ≤5-row slices for speed.

### 3.2 Installation on this machine

`environments/heretic` (Python 3.12.10):

```powershell
py -3.12 -m venv environments\heretic
.\environments\heretic\Scripts\python.exe -m pip install -U pip
.\environments\heretic\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.\environments\heretic\Scripts\python.exe -m pip install heretic-llm
```

pip installed `heretic-llm 2.1.0` (the current release; `master` is `2.2.0.dev0`) with:

```text
torch 2.14.1+cu126   torchvision 0.29.2   transformers 5.23.0   tokenizers 0.23.3
optuna 5.1.0         peft 0.22.0          accelerate 1.16.0     datasets 5.2.0
huggingface-hub 1.33.0   bitsandbytes 0.50.2   lm-eval 0.4.13    scikit-learn 1.9.1
```

Install exit code: `0`.

**Invocation facts discovered here:**

- `heretic.exe --help` works and lists every settings key as a CLI flag.
- `python -m heretic` **fails**: the package has no `__main__` module
  (`'heretic' is a package and cannot be directly executed`), so automation must call the
  console script or `heretic.main.main()`.
- Running `heretic.exe` with stdout redirected to a pipe crashes as soon as the first
  `questionary` prompt is *constructed*, even with `trial_index` pinned:
  `NoConsoleScreenBufferError: Found xterm-256color, while expecting a Windows console`.
  See section 3.3 for the workaround.

### 3.3 Real minimal abliteration run — **Verified**

Config: [`config/heretic-verify.toml`](../config/heretic-verify.toml), selected with
`HERETIC_CONFIG_FILE`. Launcher: `scripts/run_heretic_unattended.py` (see "console problem"
below). Console transcript: [`evidence/heretic-console.txt`](evidence/heretic-console.txt).

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Quantization | `bnb_4bit` (loaded as 4-bit, 0.43 GB allocated VRAM) |
| Modifier | `heretic.modifiers.abliteration.Abliteration` |
| Trials | 2 (`n_startup_trials` 1), seed 42 |
| Prompts | 5-row slices of `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors` |
| Max response length | 32 tokens, batch 2 |
| Export | `merge`, `model_action = "save"` into `docs/evidence/heretic-run/model` |

Observed results:

```text
Baseline Refusals: 5/5, KL divergence 0 (by definition)
Trial 1: Refusals 4/5, KL divergence 0.0093   (elapsed 6s)
Trial 2: Refusals 5/5, KL divergence 0.0079   (elapsed 12s cumulative)
Restoring model from trial 1 ...
Saving merged model ... Model saved to docs/evidence/heretic-run/model.
exit code 0
```

RAM stayed at 1.48–1.67 GB resident; GPU memory never exceeded 0.46 GB reserved during the
trials. The merge step loads the base model on CPU, so system RAM (not VRAM) is the constraint
at export time.

Artifacts actually produced (inspected):

```text
docs/evidence/heretic-run/model/config.json                 Qwen2ForCausalLM, 24 layers
docs/evidence/heretic-run/model/model.safetensors           942.32 MB
docs/evidence/heretic-run/model/tokenizer.json              10.89 MB
docs/evidence/heretic-run/model/tokenizer_config.json
docs/evidence/heretic-run/model/generation_config.json
docs/evidence/heretic-run/model/chat_template.jinja
docs/evidence/heretic-run/checkpoints/Qwen--Qwen2--5-0--5B-Instruct.jsonl   Optuna journal
```

The merged model was then loaded and used for generation in the same environment
(`transformers` `AutoModelForCausalLM`, fp16, `device_map="cuda"`):

```text
load_seconds 5.2, vram 0.93 GB, param_dtype torch.float16
prompt:     Explain in one sentence why the sky is blue.
completion: ... The Earth's atmosphere scatters sunlight, causing it to be scattered
            and refracted by water droplets or ice crystals in
```

So Heretic performs real model modification on this GPU, writes a loadable model, and that
model runs on the GTX 1660.

Honest limits of this evidence:

- 5 evaluation prompts and 2 trials are enough to prove the pipeline works, **not** enough to
  claim a useful abliteration. A real run uses the upstream defaults (100 trials, 100+ prompts)
  and takes far longer.
- Only one architecture (Qwen2) and one modifier (classic Abliteration) were exercised. The
  default modifier in 2.x is `ARA`; it was not run here.
- `export_strategy = "adapter"` and the upload path were not exercised.
- The merged file was written as `bfloat16` (`config.json` `"dtype": "bfloat16"`) even though
  the GPU has no native bf16. It loads fine as fp16, but users should expect the file to be
  bf16 on output.

### 3.3.1 Windows console problem (important for the GUI)

`heretic.exe` crashed *after* the trials with:

```text
questionary.select("Which trial do you want to use?", ...)
NoConsoleScreenBufferError: Found xterm-256color, while expecting a Windows console.
```

Two separate facts cause this:

1. Heretic constructs `questionary.select(...)` **eagerly**, as an argument to
   `ask_if_unset(settings.trial_index, ...)`. Setting `trial_index` avoids *running* the prompt
   but not *building* it.
2. Building a prompt on Windows creates a `prompt_toolkit` `Win32Output`, which calls
   `GetConsoleScreenBufferInfo()` on stdout. With stdout redirected to a pipe (QProcess,
   `Tee-Object`, CI) that fails.

Workaround implemented and verified: `scripts/run_heretic_unattended.py` replaces
`prompt_toolkit.output.defaults.create_output` with a `DummyOutput` before importing
`heretic.main`, then calls Heretic's own `main()`. With every interactive value pinned in the
config, no prompt is ever run, so a headless output is sufficient. The full run above completed
with exit code `0` and stdout piped.

Alternatives that were *not* verified: allocating a real console for the child process, or
running Heretic under a ConPTY/`winpty` pseudo-terminal. Either would keep prompt construction
working but complicates output capture and quoting on Windows.

### 3.4 Practical constraints for 6 GB

- The 4-bit trial loop is cheap: 0.46 GB reserved VRAM and ~6 s per trial for a 0.5B model.
  VRAM is not the limit for models in this class; **system RAM** matters because the merge step
  loads the base model on CPU (1.7 GB here, growing with model size).
- Extrapolating the same ratios (**Documented/deduced**, not measured): a 3B model in 4-bit
  needs roughly 2–3 GB of VRAM for weights plus activations for batched evaluation, and its
  merge step needs ~6 GB of free system RAM. 7B models are close to the limit of a 6 GB card
  once evaluation batches are included.
- Two trials in ~12 s is not a benchmark of a real run: with upstream defaults (100 trials,
  larger prompt sets, `max_response_length` 100) expect hours, not minutes.
- Heretic spawns its own model load per phase; keeping the model in the HF cache (it is already
  shared with Unsloth on this machine, `E:\HuggingFaceCache`) avoids repeated downloads.

---

## 4. Recommended known-good environments

| Environment | Pins verified on this machine |
| --- | --- |
| `environments/app` | Python 3.12.10, PySide6 6.12.0, huggingface_hub 2.4.0, datasets 5.2.0, psutil 7.2.2, pytest 9.2.0, ruff 0.17.6 |
| `environments/unsloth` | Python 3.12.10, torch 2.14.1+cu126, unsloth 2026.11.7, unsloth_zoo 2026.11.7, transformers 5.17.0, trl 1.15.0, peft 0.22.0, bitsandbytes 0.50.2, triton-windows 3.9.0.post30 |
| `environments/heretic` | Python 3.12.10, torch 2.14.1+cu126, heretic-llm 2.1.0, transformers 5.23.0, optuna 5.1.0, peft 0.22.0, accelerate 1.16.0, bitsandbytes 0.50.2, tip: launch via `heretic.exe` (there is no `__main__`) |

Notes:

- The three environments deliberately do **not** share packages: the app environment must stay
  free of torch, and Unsloth and Heretic pin conflicting `transformers`/`peft` versions.
- The app never imports torch or the engines. It resolves an interpreter path and shells out
  (`importlib.util.find_spec` in the engine interpreter) to report status.

---

## 5. Summary of evidence levels

| Claim | Level |
| --- | --- |
| Windows 10 19045 + GTX 1660 (7.5) + driver 617.42 | Verified |
| CUDA 12.6 PyTorch wheels run on this driver | Verified |
| Unsloth installs on Windows with Python 3.12 | Verified |
| Unsloth trains 4-bit LoRA on the GTX 1660 and saves an adapter | Verified |
| Unsloth xformers kernel matches this torch build | **Unsupported here** (falls back to PyTorch attention) |
| bf16 is natively available on compute 7.5 | **Unsupported** — use fp16 |
| 1.5B–3B models train comfortably in 6 GB at short sequence lengths | Documented/deduced, not measured |
| Heretic installs in an isolated Python 3.12 environment | Verified |
| Heretic produces a merged modified model on this GPU | Verified (0.5B, 4-bit, 2 trials, artifact loaded and generating) |
| Heretic can be driven unattended (no prompts) | Verified, with the `DummyOutput` shim in `scripts/run_heretic_unattended.py` |
| Heretic works unmodified with stdout piped | **Unsupported** — `NoConsoleScreenBufferError` during prompt construction |
| Heretic with upstream default workload (100 trials, full prompt sets) on 6 GB | Uninvestigated |
| GUI-driven Heretic worker inside 6 GB | Worker implementation is present on the development branch; full GUI-driven acceptance and long-running workload validation remain unverified |

---

## 6. Upstream links

- Unsloth: <https://github.com/unslothai/unsloth> · <https://docs.unsloth.ai/get-started/install-and-update/windows-installation>
- Heretic: <https://github.com/p-e-w/heretic>
- bitsandbytes: <https://github.com/bitsandbytes-foundation/bitsandbytes>
- PyTorch wheel index used: <https://download.pytorch.org/whl/cu126>

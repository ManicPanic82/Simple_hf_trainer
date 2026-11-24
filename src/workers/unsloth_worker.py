"""Unsloth LoRA fine-tuning worker speaking the Simple HF Trainer JSON-lines protocol.

This script runs in the isolated Unsloth environment. It communicates exclusively
over stdout using UTF-8 JSON-lines according to protocol version 1.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    from .package_metadata import snapshot_training_datasets, write_training_package
except ImportError:  # Running this file directly as an isolated worker script.
    from package_metadata import snapshot_training_datasets, write_training_package

# stdout is reserved for the JSON-lines protocol. Engine libraries print their
# own banners and progress to sys.stdout, so keep a private handle to the real
# protocol stream before any engine import can replace or wrap sys.stdout.
_PROTOCOL_STREAM = sys.stdout


def emit(
    msg_type: str,
    job_id: str | None = None,
    payload: dict | None = None,
    seq: int | None = None,
) -> None:
    """Emit one protocol line to stdout."""
    data = {
        "protocol": 1,
        "type": msg_type,
        "payload": payload or {},
    }
    if job_id is not None:
        data["job_id"] = job_id
    if seq is not None:
        data["seq"] = seq
    line = json.dumps(data, ensure_ascii=False)
    _PROTOCOL_STREAM.write(line + "\n")
    _PROTOCOL_STREAM.flush()


def validate_dataset_path(dataset_path: str | Path | None) -> Path:
    """Require an existing supported dataset; never substitute synthetic examples."""
    if dataset_path is None or not str(dataset_path).strip():
        raise ValueError("Select a prepared training dataset before starting fine-tuning.")
    path = Path(dataset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset file not found: {path}")
    if path.suffix.lower() not in {".jsonl", ".json", ".csv", ".parquet", ".arrow"}:
        raise ValueError(f"Unsupported training dataset format: {path.suffix or 'no extension'}")
    return path.resolve()


def start_cancel_listener(job_id: str) -> threading.Event:
    """Read protocol cancellation messages from stdin without blocking training."""
    cancelled = threading.Event()

    def listen() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                continue
            if message.get("type") == "cancel" and message.get("job_id") == job_id:
                cancelled.set()
                return

    threading.Thread(target=listen, name="worker-cancel-listener", daemon=True).start()
    return cancelled


def load_dataset_file(file_path: Path):
    """Load training dataset into HuggingFace Dataset."""
    from datasets import Dataset

    suffix = file_path.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number}: {error.msg}"
                    ) from error
    elif suffix == ".json":
        with open(file_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
    elif suffix == ".csv":
        import csv

        with open(file_path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)
        rows = table.to_pylist()
    elif suffix == ".arrow":
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.memory_map(str(file_path), "r") as source:
            try:
                reader = ipc.open_file(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_stream(source)
            rows = reader.read_all().to_pylist()
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    if not rows:
        raise ValueError(f"No records found in dataset: {file_path}")

    # Ensure each row has a 'text' field
    formatted_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index + 1} is not an object.")
        if row.get("text"):
            formatted_rows.append({"text": str(row["text"])})
        elif "instruction" in row and "output" in row:
            inst = str(row.get("instruction", "") or "")
            inp = str(row.get("input", "") or "")
            out = str(row.get("output", "") or "")
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            formatted_rows.append({"text": text})
        elif "prompt" in row and ("response" in row or "completion" in row):
            response = row.get("response", row.get("completion", ""))
            formatted_rows.append(
                {"text": f"### Instruction:\n{row['prompt']}\n\n### Response:\n{response}"}
            )
        else:
            raise ValueError(
                f"Dataset row {index + 1} has no supported training fields. "
                "Expected 'text', 'instruction'+'output', or 'prompt'+'response'/'completion'."
            )

    return Dataset.from_list(formatted_rows)


def create_sft_config(config_class, config_kwargs: dict, max_seq_length: int):
    """Build SFTConfig across TRL versions or raise a useful compatibility error."""
    errors = []
    last_error = None
    for length_key in ("max_length", "max_seq_length"):
        try:
            return config_class(**config_kwargs, **{length_key: max_seq_length})
        except TypeError as error:
            last_error = error
            errors.append(f"{length_key}: {error}")

    details = "; ".join(errors)
    raise RuntimeError(
        "Unable to construct installed TRL SFTConfig using either supported sequence-length argument "
        f"('max_length' or 'max_seq_length'). Details: {details}"
    ) from last_error


def evaluation_strategy_argument(config_class) -> str:
    """Select the evaluation-strategy spelling supported by this Transformers version."""
    try:
        parameters = inspect.signature(config_class).parameters
    except (TypeError, ValueError):
        return "eval_strategy"
    if "eval_strategy" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return "eval_strategy"
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    raise RuntimeError(
        "Installed TRL SFTConfig exposes neither eval_strategy nor evaluation_strategy."
    )


def estimate_causal_lm_parameters(config) -> int | None:
    """Conservatively estimate decoder-only parameter count from a HF config."""
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    intermediate = getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None)
    vocab = getattr(config, "vocab_size", None)
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    values = (hidden, layers, intermediate, vocab, heads, kv_heads)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        return None
    kv_width = hidden * kv_heads // heads
    attention = 2 * hidden * hidden + 2 * hidden * kv_width
    activation = str(getattr(config, "hidden_act", "")).lower()
    mlp = 3 * hidden * intermediate if activation in {"silu", "swiglu", "gelu_new"} else 2 * hidden * intermediate
    embeddings = vocab * hidden
    if not bool(getattr(config, "tie_word_embeddings", False)):
        embeddings += vocab * hidden
    return int((embeddings + layers * (attention + mlp)) * 1.12)


def estimate_full_finetune_vram_bytes(
    parameter_count: int, total_vram_bytes: int, precision: str, optimizer: str
) -> int:
    """Estimate weights, gradients, optimizer state and a modest activation reserve."""
    parameter_bytes = 8 if precision == "fp32" else 4
    optimizer_bytes = 8 if optimizer == "adamw_torch" else 3
    activation_reserve = max(1024**3, int(total_vram_bytes * 0.15))
    return parameter_count * (parameter_bytes + optimizer_bytes) + activation_reserve


def run_worker(cfg: dict) -> int:
    # Unsloth/torch/transformers write human-readable banners and progress to
    # stdout. Route that chatter to stderr (the host's log channel) so stdout
    # carries protocol JSON only; emit() uses the captured _PROTOCOL_STREAM.
    sys.stdout = sys.stderr

    job_id = cfg.get("job_id", "unsloth-job")
    model_name = cfg.get("model_name", "unsloth/Qwen2.5-0.5B-Instruct")
    dataset_path_str = cfg.get("dataset_path", "")
    output_dir = Path(cfg.get("output_dir", "outputs/lora_adapter")).resolve()
    max_steps = int(cfg.get("max_steps", 5))
    num_train_epochs = float(cfg.get("num_train_epochs", 3))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    max_seq_length = int(cfg.get("max_seq_length", 256))
    learning_rate = float(cfg.get("learning_rate", 2e-4))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    lora_r = int(cfg.get("lora_r", 8))
    lora_alpha = int(cfg.get("lora_alpha", 16))
    optim = str(cfg.get("optim", "adamw_8bit"))
    training_method = str(cfg.get("training_method", "qlora"))
    precision = str(cfg.get("precision", "fp16"))
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    lr_scheduler_type = str(cfg.get("lr_scheduler_type", "cosine"))
    resume_from_checkpoint = str(cfg.get("resume_from_checkpoint") or "").strip() or None
    save_strategy = str(cfg.get("save_strategy", "no"))
    save_steps = int(cfg.get("save_steps", 50))
    eval_strategy = str(cfg.get("eval_strategy", "no"))
    eval_steps = int(cfg.get("eval_steps", 50))
    eval_dataset_path_str = cfg.get("eval_dataset_path")
    # Full fine-tuning exports the complete model; adapter methods can merge or
    # export only their PEFT weights.
    default_export = "full_model" if training_method == "full_finetune" else "merged_16bit"
    export_strategy = str(cfg.get("export_strategy", default_export))
    if training_method not in {"full_finetune", "lora", "qlora"}:
        emit("error", job_id, {"message": f"Unsupported training method: {training_method!r}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if precision not in {"fp16", "bf16", "fp32"}:
        emit("error", job_id, {"message": f"Unsupported training precision: {precision!r}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if (training_method == "full_finetune") != (export_strategy == "full_model"):
        emit("error", job_id, {"message": "Full fine-tuning requires full_model export; adapter methods require merged_16bit or adapter export."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if export_strategy not in {"full_model", "merged_16bit", "adapter"}:
        emit(
            "error",
            job_id,
            {"message": f"Unsupported export_strategy: {export_strategy!r}"},
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    if max_steps < 0 or num_train_epochs <= 0 or warmup_steps < 0:
        emit(
            "error",
            job_id,
            {
                "message": "Max steps must be zero or greater, epochs must be positive, and warmup steps cannot be negative."
            },
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if batch_size < 1 or grad_accum < 1 or max_seq_length < 64:
        emit(
            "error",
            job_id,
            {
                "message": "Batch size, gradient accumulation and sequence length must be positive supported values."
            },
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if not 0 < learning_rate < 1:
        emit("error", job_id, {"message": "Learning rate must be greater than 0 and less than 1."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if (
        not math.isfinite(weight_decay)
        or not math.isfinite(max_grad_norm)
        or weight_decay < 0
        or max_grad_norm < 0
    ):
        emit("error", job_id, {"message": "Weight decay and max gradient norm must be finite and non-negative."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if lr_scheduler_type not in {"cosine", "linear", "constant", "constant_with_warmup"}:
        emit("error", job_id, {"message": f"Unsupported learning-rate scheduler: {lr_scheduler_type!r}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if optim not in {"adamw_8bit", "adamw_torch", "paged_adamw_8bit"}:
        emit("error", job_id, {"message": f"Unsupported optimizer: {optim!r}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if save_strategy not in {"no", "steps", "epoch"} or eval_strategy not in {
        "no",
        "steps",
        "epoch",
    }:
        emit(
            "error",
            job_id,
            {"message": "Save and evaluation strategies must be no, steps, or epoch."},
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if save_steps < 1 or eval_steps < 1:
        emit(
            "error", job_id, {"message": "Save and evaluation intervals must be at least one step."}
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if resume_from_checkpoint and not Path(resume_from_checkpoint).is_dir():
        emit("error", job_id, {"message": f"Resume checkpoint folder not found: {resume_from_checkpoint}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if eval_strategy != "no" and not eval_dataset_path_str:
        emit(
            "error",
            job_id,
            {"message": "An evaluation dataset is required when evaluation is enabled."},
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    print("[checkpoint] Before HELLO", file=sys.stderr, flush=True)
    emit("hello", payload={"role": "unsloth-worker", "pid": os.getpid()})

    try:
        print("[checkpoint] Before dataset validation", file=sys.stderr, flush=True)
        dataset_path = validate_dataset_path(dataset_path_str)
        eval_dataset_path = None
        if eval_strategy != "no":
            eval_dataset_path = validate_dataset_path(eval_dataset_path_str)
            if eval_dataset_path.resolve() == dataset_path.resolve():
                raise ValueError("Training and evaluation datasets must be different files.")
        original_dataset_path = dataset_path
        original_eval_dataset_path = eval_dataset_path
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path, eval_dataset_path = snapshot_training_datasets(
            output_dir,
            dataset_path,
            eval_dataset_path,
        )
        print(
            f"[checkpoint] Dataset snapshot ready: {dataset_path}",
            file=sys.stderr,
            flush=True,
        )
    except (ValueError, OSError) as error:
        emit("error", job_id, {"message": str(error)})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    try:
        print("[checkpoint] Before importing unsloth", file=sys.stderr, flush=True)
        import unsloth

        print("[checkpoint] Unsloth imported", file=sys.stderr, flush=True)

        from unsloth import FastLanguageModel
        import torch
        from transformers import AutoConfig

        print("[checkpoint] FastLanguageModel and torch imported", file=sys.stderr, flush=True)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 was selected, but the active CUDA GPU does not support BF16.")
        if training_method == "full_finetune":
            if not torch.cuda.is_available():
                raise RuntimeError("Full fine-tuning requires a CUDA GPU; no CUDA device is available.")
            model_config_for_preflight = AutoConfig.from_pretrained(
                model_name, trust_remote_code=False
            )
            parameter_count = estimate_causal_lm_parameters(model_config_for_preflight)
            if parameter_count is None:
                raise RuntimeError(
                    "Cannot safely estimate this model's full fine-tuning memory. "
                    "Full fine-tuning was stopped before model loading."
                )
            total_vram = int(torch.cuda.get_device_properties(0).total_memory)
            estimated_vram = estimate_full_finetune_vram_bytes(
                parameter_count, total_vram, precision, optim
            )
            safe_budget = int(total_vram * 0.85)
            emit(
                "log",
                job_id,
                {"message": (
                    f"Full fine-tuning preflight: approximately {parameter_count:,} parameters; "
                    f"estimated {estimated_vram / 1024**3:.2f} GiB required, "
                    f"{total_vram / 1024**3:.2f} GiB GPU memory available."
                ), "level": "info"},
            )
            if estimated_vram > safe_budget:
                raise RuntimeError(
                    "Full fine-tuning blocked by the GPU memory preflight. "
                    f"Estimated requirement is {estimated_vram / 1024**3:.2f} GiB, "
                    f"but the safe budget is {safe_budget / 1024**3:.2f} GiB. "
                    "Choose a smaller model, reduce precision/sequence length, or use LoRA/QLoRA."
                )

        engine_version = getattr(unsloth, "__version__", "unknown")
        emit("ready", payload={"engine": "unsloth", "version": engine_version})
        print("[checkpoint] READY emitted", file=sys.stderr, flush=True)
    except Exception as error:
        emit("error", job_id, {"message": f"Failed to initialize Unsloth or validate the training hardware: {error}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    emit(
        "log",
        job_id,
        {"message": (
            f"Loading {model_name} for {training_method} ({precision}; "
            f"{'4-bit base' if training_method == 'qlora' else 'full-precision base'})..."
        ), "level": "info"},
    )
    emit("progress", job_id, {"fraction": 0.05, "step": 0, "total": max(1, max_steps)})

    try:
        print("[checkpoint] Before model loading", file=sys.stderr, flush=True)
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype_map[precision],
            load_in_4bit=training_method == "qlora",
        )
        print("[checkpoint] Model loaded successfully", file=sys.stderr, flush=True)
        emit("progress", job_id, {"fraction": 0.15, "step": 0, "total": max_steps})

        if training_method in {"lora", "qlora"}:
            emit("log", job_id, {"message": "Preparing LoRA adapters...", "level": "info"})
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
                random_state=42,
            )
            print("[checkpoint] LoRA configured successfully", file=sys.stderr, flush=True)
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False
            emit("log", job_id, {"message": "Full fine-tuning enabled: all model weights are trainable; no adapters are attached.", "level": "info"})
        model_config = getattr(model, "config", None)

        def config_value(name):
            value = getattr(model_config, name, None)
            return value if isinstance(value, (str, int, float, bool)) else None

        peft_configs = getattr(model, "peft_config", {}) or {}
        active_peft = (
            next(iter(peft_configs.values()), None) if hasattr(peft_configs, "values") else None
        )
        target_modules = getattr(active_peft, "target_modules", None)
        if isinstance(target_modules, (set, tuple, list)):
            target_modules = ", ".join(sorted(str(item) for item in target_modules))
        elif target_modules is not None:
            target_modules = str(target_modules)
        architecture = {
            "model_type": config_value("model_type"),
            "num_hidden_layers": config_value("num_hidden_layers") or config_value("n_layer"),
            "hidden_size": config_value("hidden_size") or config_value("n_embd"),
            "num_attention_heads": config_value("num_attention_heads") or config_value("n_head"),
            "num_key_value_heads": config_value("num_key_value_heads"),
            "intermediate_size": config_value("intermediate_size"),
            "vocab_size": config_value("vocab_size"),
            "max_position_embeddings": config_value("max_position_embeddings"),
            "lora_rank": lora_r if training_method in {"lora", "qlora"} else None,
            "lora_alpha": lora_alpha if training_method in {"lora", "qlora"} else None,
            "target_modules": target_modules,
            "training_method": training_method,
            "precision": precision,
        }
        emit("architecture", job_id, {"architecture": architecture})
        emit(
            "log",
            job_id,
            {"message": "LoRA adapters configured. Loading dataset...", "level": "info"},
        )

        print("[checkpoint] Before dataset loading", file=sys.stderr, flush=True)
        train_dataset = load_dataset_file(dataset_path)
        print(
            f"[checkpoint] Dataset loaded: {len(train_dataset)} samples",
            file=sys.stderr,
            flush=True,
        )
        if len(train_dataset) == 0:
            raise ValueError(f"Training dataset is empty: {dataset_path}")
        eval_dataset = None
        if eval_strategy != "no":
            eval_dataset = load_dataset_file(eval_dataset_path)
            if len(eval_dataset) == 0:
                raise ValueError(f"Evaluation dataset is empty: {eval_dataset_path}")
            emit(
                "log",
                job_id,
                {
                    "message": f"Loaded {len(eval_dataset)} evaluation samples from {eval_dataset_path}",
                    "level": "info",
                },
            )
        progress_total_steps = max_steps or max(
            1, math.ceil(len(train_dataset) / (batch_size * grad_accum) * num_train_epochs)
        )
        emit(
            "log",
            job_id,
            {
                "message": f"Loaded {len(train_dataset)} training samples from {dataset_path}",
                "level": "info",
            },
        )

        print("[checkpoint] Before Trainer/TRL imports", file=sys.stderr, flush=True)
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        metrics_history: list[dict] = []

        class ProtocolCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                metric_record = {"step": int(state.global_step)}
                for key, value in logs.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric_value):
                        metric_record[str(key)] = numeric_value
                if len(metric_record) > 1:
                    metrics_history.append(metric_record)
                step = int(state.global_step)
                for metric_name, metric_value in metric_record.items():
                    if metric_name == "step":
                        continue
                    emit(
                        "metric",
                        job_id,
                        {"name": metric_name, "value": metric_value, "step": step},
                    )
                if "loss" in metric_record:
                    loss_val = metric_record["loss"]
                    emit(
                        "log",
                        job_id,
                        {
                            "message": f"Step {step}/{progress_total_steps} · loss: {loss_val:.4f}",
                            "level": "info",
                        },
                    )
                if "eval_loss" in metric_record:
                    eval_loss = metric_record["eval_loss"]
                    emit(
                        "log",
                        job_id,
                        {
                            "message": f"Evaluation at step {step} · loss: {eval_loss:.4f}",
                            "level": "info",
                        },
                    )

            def on_step_end(self, args, state, control, **kwargs):
                if cancel_event.is_set():
                    control.should_training_stop = True
                    emit(
                        "log",
                        job_id,
                        {
                            "message": "Cancellation acknowledged; stopping after the current training step.",
                            "level": "warning",
                        },
                    )
                step = int(state.global_step)
                frac = min(0.95, max(0.15, 0.15 + 0.8 * (step / max(1, progress_total_steps))))
                emit(
                    "progress",
                    job_id,
                    {
                        "fraction": round(frac, 3),
                        "training_fraction": min(1.0, step / max(1, progress_total_steps)),
                        "step": step,
                        "total": progress_total_steps,
                        "epoch": float(state.epoch) if state.epoch is not None else None,
                    },
                )

        print("[checkpoint] Trainer/TRL imports complete", file=sys.stderr, flush=True)
        sft_kwargs = {
            "output_dir": str(output_dir / "temp_checkpoints"),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_steps": max_steps if max_steps > 0 else -1,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "logging_steps": 1,
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "save_total_limit": 2,
            "report_to": "none",
            "fp16": precision == "fp16",
            "bf16": precision == "bf16",
            "gradient_checkpointing": gradient_checkpointing,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "lr_scheduler_type": lr_scheduler_type,
            "dataset_text_field": "text",
            "optim": optim,
            "seed": 42,
        }
        sft_kwargs[evaluation_strategy_argument(SFTConfig)] = eval_strategy
        if eval_strategy != "no":
            sft_kwargs["eval_steps"] = eval_steps
        print("[checkpoint] Before SFTConfig construction", file=sys.stderr, flush=True)
        config = create_sft_config(SFTConfig, sft_kwargs, max_seq_length)
        print("[checkpoint] SFTConfig constructed", file=sys.stderr, flush=True)

        print("[checkpoint] Before SFTTrainer construction", file=sys.stderr, flush=True)
        trainer = None
        for extra in (
            {"tokenizer": tokenizer, "processing_class": tokenizer},
            {"processing_class": tokenizer},
            {"tokenizer": tokenizer},
        ):
            try:
                print(
                    f"[checkpoint] Trying SFTTrainer arguments: {list(extra)}",
                    file=sys.stderr,
                    flush=True,
                )
                trainer_kwargs = {
                    "model": model,
                    "train_dataset": train_dataset,
                    "args": config,
                    "callbacks": [ProtocolCallback()],
                    **extra,
                }
                if eval_dataset is not None:
                    trainer_kwargs["eval_dataset"] = eval_dataset
                trainer = SFTTrainer(**trainer_kwargs)
                print(
                    "[checkpoint] SFTTrainer constructed successfully", file=sys.stderr, flush=True
                )
                break
            except TypeError as error:
                print(
                    f"[checkpoint] SFTTrainer argument combination rejected: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

        if trainer is None:
            raise RuntimeError("Could not initialize SFTTrainer with installed TRL.")

        emit(
            "log",
            job_id,
            {"message": f"Beginning fine-tuning for {max_steps} steps...", "level": "info"},
        )

        # The cancellation listener must not block on stdin during the heavyweight
        # engine imports. On Windows a daemon thread reading an open QProcess stdin
        # pipe deadlocks the numpy/torch C-extension imports performed by
        # ``import unsloth``. Cancel messages sent during startup stay buffered in
        # the pipe and are consumed as soon as this listener starts.
        print("[checkpoint] Before cancellation listener", file=sys.stderr, flush=True)
        cancel_event = start_cancel_listener(job_id)
        print("[checkpoint] Cancellation listener started", file=sys.stderr, flush=True)

        start_time = time.time()
        print("[checkpoint] Before trainer.train()", file=sys.stderr, flush=True)
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("[checkpoint] trainer.train() returned", file=sys.stderr, flush=True)
        elapsed = round(time.time() - start_time, 1)

        final_loss = float(getattr(train_result, "training_loss", float("nan")))
        was_cancelled = cancel_event.is_set()
        status_message = (
            "Training cancelled after the current step."
            if was_cancelled
            else f"Training completed in {elapsed}s with final loss {final_loss:.4f}"
        )
        emit(
            "log",
            job_id,
            {"message": status_message, "level": "warning" if was_cancelled else "info"},
        )
        emit(
            "progress",
            job_id,
            {
                "fraction": 0.98,
                "step": int(
                    getattr(getattr(trainer, "state", None), "global_step", progress_total_steps)
                ),
                "total": progress_total_steps,
            },
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        if export_strategy == "full_model":
            emit("log", job_id, {"message": "Saving the complete fine-tuned model checkpoint...", "level": "info"})
            model.save_pretrained(str(output_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(output_dir))
            print("[checkpoint] Full model checkpoint saved", file=sys.stderr, flush=True)
        elif export_strategy == "merged_16bit":
            # Unsloth dequantizes the 4-bit base and folds in the LoRA weights,
            # producing a normal fp16 checkpoint. That checkpoint is discovered
            # as a base model, so it can be selected as the starting point for a
            # subsequent continued-training job.
            print("[checkpoint] Before merged model save", file=sys.stderr, flush=True)
            emit(
                "log",
                job_id,
                {
                    "message": "Merging LoRA adapters into the model and saving a 16-bit checkpoint...",
                    "level": "info",
                },
            )
            merge_saver = getattr(model, "save_pretrained_merged", None)
            if not callable(merge_saver):
                raise RuntimeError(
                    "This Unsloth build does not expose save_pretrained_merged; "
                    "cannot produce a merged model."
                )
            merge_saver(str(output_dir), tokenizer, save_method="merged_16bit")
            print("[checkpoint] Merged model saved", file=sys.stderr, flush=True)
        else:
            print("[checkpoint] Before saving adapter", file=sys.stderr, flush=True)
            model.save_pretrained(str(output_dir))
            print("[checkpoint] Adapter saved", file=sys.stderr, flush=True)
            tokenizer.save_pretrained(str(output_dir))
            print("[checkpoint] Tokenizer saved", file=sys.stderr, flush=True)

        weight_files = [
            path
            for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
        ]
        if not weight_files:
            raise RuntimeError(
                "Training export finished without producing a model weight file; "
                "the output package will not be reported as successful."
            )

        package_summary = {
            "status": "cancelled" if was_cancelled else "completed",
            "final_loss": final_loss if math.isfinite(final_loss) else None,
            "elapsed_seconds": elapsed,
            "global_step": int(getattr(getattr(trainer, "state", None), "global_step", 0)),
            "training_rows": len(train_dataset),
            "evaluation_rows": len(eval_dataset) if eval_dataset is not None else None,
            "engine": "unsloth",
            "engine_version": engine_version,
            "training_method": training_method,
            "precision": precision,
            "optimizer": optim,
        }
        write_training_package(
            output_dir=output_dir,
            config=cfg,
            metrics=metrics_history,
            summary=package_summary,
            train_dataset=dataset_path,
            eval_dataset=eval_dataset_path,
            train_dataset_source=original_dataset_path,
            eval_dataset_source=original_eval_dataset_path,
        )
        emit(
            "log",
            job_id,
            {
                "message": "Archived training/evaluation datasets and wrote model provenance files.",
                "level": "info",
            },
        )

        package_metadata_files = {
            "README.md",
            "training_config.json",
            "training_summary.json",
            "training_metrics.json",
            "package_manifest.json",
        }
        artifacts = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and path.name not in package_metadata_files
        )
        emit(
            "result",
            job_id,
            {
                "output_dir": str(output_dir),
                "final_loss": final_loss,
                "elapsed_s": elapsed,
                "artifacts": artifacts,
                "export_strategy": export_strategy,
            },
        )
        if was_cancelled:
            emit("job_complete", job_id, {"success": False, "exit_code": 130, "cancelled": True})
            return 130
        emit(
            "progress",
            job_id,
            {"fraction": 1.0, "step": progress_total_steps, "total": progress_total_steps},
        )
        emit("job_complete", job_id, {"success": True, "exit_code": 0})
        return 0

    except Exception as error:
        tb = traceback.format_exc()
        emit("error", job_id, {"message": f"{error}\n{tb[-1000:]}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1


def main(argv: list[str] | None = None) -> int:
    print("[startup] Entered main()", file=sys.stderr, flush=True)

    parser = argparse.ArgumentParser(description="Unsloth training worker")
    parser.add_argument("--config", help="Path to JSON config or inline JSON string")
    args = parser.parse_args(argv)

    print(
        f"[startup] Arguments parsed; config supplied: {bool(args.config)}",
        file=sys.stderr,
        flush=True,
    )

    cfg = {}
    if args.config:
        text = args.config.strip()
        if text.startswith("{"):
            cfg = json.loads(text)
            print(
                "[startup] Inline JSON configuration parsed",
                file=sys.stderr,
                flush=True,
            )
        elif Path(text).is_file():
            with open(text, encoding="utf-8") as f:
                cfg = json.load(f)
            print(
                "[startup] Configuration file loaded",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                "[startup] Config argument is neither inline JSON nor an existing file",
                file=sys.stderr,
                flush=True,
            )

    if not cfg and not sys.stdin.isatty():
        print(
            "[startup] No configuration loaded; checking stdin",
            file=sys.stderr,
            flush=True,
        )
        with contextlib.suppress(Exception):
            line = sys.stdin.readline().strip()
            if line:
                cfg = json.loads(line)
                print(
                    "[startup] Configuration loaded from stdin",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"[startup] Calling run_worker(); config loaded: {bool(cfg)}",
        file=sys.stderr,
        flush=True,
    )
    return run_worker(cfg)


if __name__ == "__main__":
    print("[startup] Worker script entry point reached", file=sys.stderr, flush=True)
    raise SystemExit(main())

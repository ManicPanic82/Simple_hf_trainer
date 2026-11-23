"""Unsloth LoRA fine-tuning worker speaking the Simple HF Trainer JSON-lines protocol.

This script runs in the isolated Unsloth environment. It communicates exclusively
over stdout using UTF-8 JSON-lines according to protocol version 1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import inspect
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

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
    return path


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
    save_strategy = str(cfg.get("save_strategy", "no"))
    save_steps = int(cfg.get("save_steps", 50))
    eval_strategy = str(cfg.get("eval_strategy", "no"))
    eval_steps = int(cfg.get("eval_steps", 50))
    eval_dataset_path_str = cfg.get("eval_dataset_path")
    # "merged_16bit" bakes the LoRA weights into a full fp16 model that can be
    # selected as the base for a later continued-training job. "adapter" keeps
    # the small PEFT adapter only (the previous behaviour).
    export_strategy = str(cfg.get("export_strategy", "merged_16bit"))
    if export_strategy not in {"merged_16bit", "adapter"}:
        emit(
            "error",
            job_id,
            {"message": f"Unsupported export_strategy: {export_strategy!r}"},
        )
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    if max_steps < 0 or num_train_epochs <= 0 or warmup_steps < 0:
        emit("error", job_id, {"message": "Max steps must be zero or greater, epochs must be positive, and warmup steps cannot be negative."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if batch_size < 1 or grad_accum < 1 or max_seq_length < 64:
        emit("error", job_id, {"message": "Batch size, gradient accumulation and sequence length must be positive supported values."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if not 0 < learning_rate < 1:
        emit("error", job_id, {"message": "Learning rate must be greater than 0 and less than 1."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if optim not in {"adamw_8bit", "adamw_torch", "paged_adamw_8bit"}:
        emit("error", job_id, {"message": f"Unsupported optimizer: {optim!r}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if save_strategy not in {"no", "steps", "epoch"} or eval_strategy not in {"no", "steps", "epoch"}:
        emit("error", job_id, {"message": "Save and evaluation strategies must be no, steps, or epoch."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if save_steps < 1 or eval_steps < 1:
        emit("error", job_id, {"message": "Save and evaluation intervals must be at least one step."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1
    if eval_strategy != "no" and not eval_dataset_path_str:
        emit("error", job_id, {"message": "An evaluation dataset is required when evaluation is enabled."})
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
        print(
            f"[checkpoint] Dataset validated: {dataset_path}",
            file=sys.stderr,
            flush=True,
        )
    except (ValueError, FileNotFoundError) as error:
        emit("error", job_id, {"message": str(error)})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    try:
        print("[checkpoint] Before importing unsloth", file=sys.stderr, flush=True)
        import unsloth

        print("[checkpoint] Unsloth imported", file=sys.stderr, flush=True)

        from unsloth import FastLanguageModel

        print("[checkpoint] FastLanguageModel imported", file=sys.stderr, flush=True)

        engine_version = getattr(unsloth, "__version__", "unknown")
        emit("ready", payload={"engine": "unsloth", "version": engine_version})
        print("[checkpoint] READY emitted", file=sys.stderr, flush=True)
    except Exception as error:
        emit("error", job_id, {"message": f"Failed to import Unsloth: {error}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    emit(
        "log",
        job_id,
        {"message": f"Loading model {model_name} in 4-bit precision...", "level": "info"},
    )
    emit("progress", job_id, {"fraction": 0.05, "step": 0, "total": max(1, max_steps)})

    try:
        print("[checkpoint] Before model loading", file=sys.stderr, flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        print("[checkpoint] Model loaded successfully", file=sys.stderr, flush=True)
        emit(
            "log",
            job_id,
            {"message": "Model loaded successfully. Preparing LoRA adapters...", "level": "info"},
        )
        emit("progress", job_id, {"fraction": 0.15, "step": 0, "total": max_steps})

        print("[checkpoint] Before LoRA configuration", file=sys.stderr, flush=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        print("[checkpoint] LoRA configured successfully", file=sys.stderr, flush=True)
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
                raise ValueError(f"Evaluation dataset is empty: {eval_path}")
            emit(
                "log",
                job_id,
                {"message": f"Loaded {len(eval_dataset)} evaluation samples from {eval_path}", "level": "info"},
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

        class ProtocolCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                step = int(state.global_step)
                if "loss" in logs:
                    loss_val = float(logs["loss"])
                    emit("metric", job_id, {"name": "loss", "value": loss_val, "step": step})
                    emit(
                        "log",
                        job_id,
                        {
                            "message": f"Step {step}/{progress_total_steps} · loss: {loss_val:.4f}",
                            "level": "info",
                        },
                    )
                if "eval_loss" in logs:
                    eval_loss = float(logs["eval_loss"])
                    emit(
                        "metric",
                        job_id,
                        {"name": "eval_loss", "value": eval_loss, "step": step},
                    )
                    emit(
                        "log",
                        job_id,
                        {"message": f"Evaluation at step {step} · loss: {eval_loss:.4f}", "level": "info"},
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
                    {"fraction": round(frac, 3), "step": step, "total": progress_total_steps},
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
            "report_to": "none",
            "fp16": True,
            "bf16": False,
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
        train_result = trainer.train()
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
                "step": int(getattr(getattr(trainer, "state", None), "global_step", progress_total_steps)),
                "total": progress_total_steps,
            },
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        if export_strategy == "merged_16bit":
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

        artifacts = [p.name for p in output_dir.iterdir() if p.is_file()]
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
        emit("progress", job_id, {"fraction": 1.0, "step": progress_total_steps, "total": progress_total_steps})
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

"""Unsloth LoRA fine-tuning worker speaking the Simple HF Trainer JSON-lines protocol.

This script runs in the isolated Unsloth environment. It communicates exclusively
over stdout using UTF-8 JSON-lines according to protocol version 1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


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
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def load_dataset_file(file_path: Path):
    """Load training dataset into HuggingFace Dataset."""
    from datasets import Dataset

    suffix = file_path.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        rows.append(json.loads(line))
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
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    if not rows:
        raise ValueError(f"No records found in dataset: {file_path}")

    # Ensure each row has a 'text' field
    formatted_rows = []
    for r in rows:
        if "text" in r and r["text"]:
            formatted_rows.append({"text": str(r["text"])})
        elif "instruction" in r:
            inst = r.get("instruction", "")
            inp = r.get("input", "")
            out = r.get("output", "")
            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            formatted_rows.append({"text": text})
        else:
            first_val = next(iter(r.values())) if r else ""
            formatted_rows.append({"text": str(first_val)})

    return Dataset.from_list(formatted_rows)


def run_worker(cfg: dict) -> int:
    job_id = cfg.get("job_id", "unsloth-job")
    model_name = cfg.get("model_name", "unsloth/Qwen2.5-0.5B-Instruct")
    dataset_path_str = cfg.get("dataset_path", "")
    output_dir = Path(cfg.get("output_dir", "outputs/lora_adapter")).resolve()
    max_steps = int(cfg.get("max_steps", 5))
    max_seq_length = int(cfg.get("max_seq_length", 256))
    learning_rate = float(cfg.get("learning_rate", 2e-4))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    lora_r = int(cfg.get("lora_r", 8))
    lora_alpha = int(cfg.get("lora_alpha", 16))

    emit("hello", payload={"role": "unsloth-worker", "pid": os.getpid()})

    try:
        import unsloth
        from unsloth import FastLanguageModel

        engine_version = getattr(unsloth, "__version__", "unknown")
        emit("ready", payload={"engine": "unsloth", "version": engine_version})
    except Exception as error:
        emit("error", job_id, {"message": f"Failed to import Unsloth: {error}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    emit("log", job_id, {"message": f"Loading model {model_name} in 4-bit precision...", "level": "info"})
    emit("progress", job_id, {"fraction": 0.05, "step": 0, "total": max_steps})

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        emit("log", job_id, {"message": "Model loaded successfully. Preparing LoRA adapters...", "level": "info"})
        emit("progress", job_id, {"fraction": 0.15, "step": 0, "total": max_steps})

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        emit("log", job_id, {"message": "LoRA adapters configured. Loading dataset...", "level": "info"})

        if dataset_path_str and Path(dataset_path_str).is_file():
            train_dataset = load_dataset_file(Path(dataset_path_str))
            emit("log", job_id, {"message": f"Loaded {len(train_dataset)} samples from {dataset_path_str}", "level": "info"})
        else:
            emit("log", job_id, {"message": "No dataset specified; using fallback training data", "level": "warning"})
            from datasets import Dataset

            rows = [{"text": f"### Question: What is {i}+{i}?\n### Answer: {i+i}"} for i in range(32)]
            train_dataset = Dataset.from_list(rows)

        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        class ProtocolCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    loss_val = float(logs["loss"])
                    step = int(state.global_step)
                    emit("metric", job_id, {"name": "loss", "value": loss_val, "step": step})
                    emit("log", job_id, {"message": f"Step {step}/{max_steps} · loss: {loss_val:.4f}", "level": "info"})

            def on_step_end(self, args, state, control, **kwargs):
                step = int(state.global_step)
                frac = min(0.95, max(0.15, 0.15 + 0.8 * (step / max(1, max_steps))))
                emit("progress", job_id, {"fraction": round(frac, 3), "step": step, "total": max_steps})

        sft_kwargs = {
            "output_dir": str(output_dir / "temp_checkpoints"),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "fp16": True,
            "bf16": False,
            "dataset_text_field": "text",
            "optim": "adamw_8bit",
            "seed": 42,
        }
        for length_key in ("max_length", "max_seq_length"):
            try:
                config = SFTConfig(**sft_kwargs, **{length_key: max_seq_length})
                break
            except TypeError:
                continue

        trainer = None
        for extra in (
            {"tokenizer": tokenizer, "processing_class": tokenizer},
            {"processing_class": tokenizer},
            {"tokenizer": tokenizer},
        ):
            try:
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    args=config,
                    callbacks=[ProtocolCallback()],
                    **extra,
                )
                break
            except TypeError:
                continue

        if trainer is None:
            raise RuntimeError("Could not initialize SFTTrainer with installed TRL.")

        emit("log", job_id, {"message": f"Beginning fine-tuning for {max_steps} steps...", "level": "info"})
        start_time = time.time()
        train_result = trainer.train()
        elapsed = round(time.time() - start_time, 1)

        final_loss = float(getattr(train_result, "training_loss", float("nan")))
        emit("log", job_id, {"message": f"Training completed in {elapsed}s with final loss {final_loss:.4f}", "level": "info"})
        emit("progress", job_id, {"fraction": 0.98, "step": max_steps, "total": max_steps})

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        artifacts = [p.name for p in output_dir.iterdir() if p.is_file()]
        emit("result", job_id, {
            "output_dir": str(output_dir),
            "final_loss": final_loss,
            "elapsed_s": elapsed,
            "artifacts": artifacts,
        })
        emit("progress", job_id, {"fraction": 1.0, "step": max_steps, "total": max_steps})
        emit("job_complete", job_id, {"success": True, "exit_code": 0})
        return 0

    except Exception as error:
        tb = traceback.format_exc()
        emit("error", job_id, {"message": f"{error}\n{tb[-1000:]}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unsloth training worker")
    parser.add_argument("--config", help="Path to JSON config or inline JSON string")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        text = args.config.strip()
        if text.startswith("{"):
            cfg = json.loads(text)
        elif Path(text).is_file():
            with open(text, encoding="utf-8") as f:
                cfg = json.load(f)

    if not cfg and not sys.stdin.isatty():
        # Check stdin if available without blocking indefinitely
        with contextlib.suppress(Exception):
            line = sys.stdin.readline().strip()
            if line:
                cfg = json.loads(line)

    return run_worker(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

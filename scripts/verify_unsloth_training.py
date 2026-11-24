"""Real minimal Unsloth LoRA/QLoRA run used as Phase 0 evidence.

This is intentionally small (one tiny model, a handful of steps) so it proves
the engine actually trains on this GPU without exhausting a 6 GB card.

Run inside the isolated engine environment:

    environments\\unsloth\\Scripts\\python.exe scripts\\verify_unsloth_training.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

MODEL = os.environ.get("SHT_UNSLOTH_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")
MAX_STEPS = int(os.environ.get("SHT_UNSLOTH_STEPS", "5"))
MAX_SEQ_LENGTH = int(os.environ.get("SHT_UNSLOTH_SEQ", "256"))
OUTPUT_DIR = Path(os.environ.get("SHT_UNSLOTH_OUT", "docs/evidence/unsloth-lora-run")).resolve()


def build_dataset():
    from datasets import Dataset

    rows = [
        {"text": (f"### Question: What is {value} + {value}?\n### Answer: {value + value}")}
        for value in range(64)
    ]
    return Dataset.from_list(rows)


def make_trainer(model, tokenizer, dataset):
    """Create an SFTTrainer across TRL 0.x/1.x argument differences."""
    from trl import SFTConfig, SFTTrainer

    common = {
        "output_dir": str(OUTPUT_DIR / "trainer-state"),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": MAX_STEPS,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "fp16": True,
        "bf16": False,
        "dataset_text_field": "text",
        "optim": "adamw_8bit",
        "seed": 42,
    }
    config = None
    for length_key in ("max_length", "max_seq_length"):
        try:
            config = SFTConfig(**common, **{length_key: MAX_SEQ_LENGTH})
            break
        except TypeError:
            continue
    if config is None:
        raise RuntimeError("Could not construct SFTConfig with a known max-length argument.")

    attempts = [
        {"tokenizer": tokenizer, "processing_class": tokenizer},
        {"processing_class": tokenizer},
        {"tokenizer": tokenizer},
    ]
    last_error: Exception | None = None
    for extra in attempts:
        try:
            return SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=config,
                **extra,
            )
        except TypeError as error:
            last_error = error
    raise RuntimeError(f"SFTTrainer signature not supported by this TRL version: {last_error}")


def main() -> int:
    report: dict[str, object] = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "model": MODEL,
        "max_steps": MAX_STEPS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "output_dir": str(OUTPUT_DIR),
    }
    started = time.perf_counter()

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["capability"] = list(torch.cuda.get_device_capability(0))
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            report["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )

        import unsloth

        report["unsloth"] = getattr(unsloth, "__version__", "unknown")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        report["load_seconds"] = round(time.perf_counter() - started, 1)

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        dataset = build_dataset()
        trainer = make_trainer(model, tokenizer, dataset)

        train_started = time.perf_counter()
        result = trainer.train()
        report["train_seconds"] = round(time.perf_counter() - train_started, 1)
        report["train_loss"] = float(getattr(result, "training_loss", float("nan")))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        artifacts = sorted(path.name for path in OUTPUT_DIR.iterdir())
        report["artifacts"] = artifacts
        report["adapter_weights"] = [
            name for name in artifacts if name.endswith((".safetensors", ".bin"))
        ]

        if torch.cuda.is_available():
            report["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

        # One generation proves the fine-tuned model loads and produces text.
        from unsloth import FastLanguageModel as FLM

        FLM.for_inference(model)
        prompt = "### Question: What is 21 + 21?\n### Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        report["sample_output"] = tokenizer.decode(output[0], skip_special_tokens=True)[:200]
        report["status"] = "verified"
    except Exception as error:  # noqa: BLE001 - evidence must capture failures
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()[-4000:]
        print(json.dumps(report, indent=2, default=str))
        return 1

    report["total_seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

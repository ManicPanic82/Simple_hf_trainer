"""Unsloth export strategy: merged 16-bit checkpoint vs adapter-only."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

import workers.unsloth_worker as worker_module


def _run_with_export(
    monkeypatch,
    tmp_path: Path,
    strategy: str | None,
    events: list[str],
    config_overrides: dict | None = None,
):
    """Drive run_worker with stubbed engines and capture save calls."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")

    class FakeUnsloth(types.ModuleType):
        def __getattr__(self, name: str):
            if name == "FastLanguageModel":
                return FakeFastLanguageModel
            raise AttributeError(name)

    class FakeModel:
        def save_pretrained(self, path):
            events.append("adapter_saved")

        def save_pretrained_merged(self, path, tokenizer, save_method="merged_16bit"):
            events.append(f"merged:{save_method}")
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "config.json").write_text(
                json.dumps({"model_type": "qwen3"}), encoding="utf-8"
            )

    class FakeTokenizer:
        def save_pretrained(self, path):
            events.append("tokenizer_saved")

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    captured: dict = {}

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeSFTTrainer:
        state = types.SimpleNamespace(global_step=1)

        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self):
            events.append("trained")
            return types.SimpleNamespace(training_loss=0.5)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.TrainerCallback = type("TrainerCallback", (), {})
    fake_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    monkeypatch.setitem(sys.modules, "unsloth", FakeUnsloth("unsloth"))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeSFTTrainer),
    )
    emitted: list[tuple] = []

    def _capture(t, job_id=None, payload=None, seq=None):
        emitted.append((t, job_id, payload or {}))

    monkeypatch.setattr(worker_module, "emit", _capture)
    monkeypatch.setattr(worker_module, "start_cancel_listener", lambda job_id: threading.Event())

    cfg = {
        "job_id": "export-test",
        "model_name": "fake-model",
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "out"),
        "max_steps": 1,
    }
    if strategy is not None:
        cfg["export_strategy"] = strategy
    if config_overrides:
        cfg.update(config_overrides)

    real_stdout = sys.stdout
    try:
        rc = worker_module.run_worker(cfg)
    finally:
        sys.stdout = real_stdout
    return rc, emitted, captured


def test_default_export_is_merged_16bit(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    rc, emitted, _captured = _run_with_export(monkeypatch, tmp_path, None, events)
    assert rc == 0
    assert "merged:merged_16bit" in events
    assert "adapter_saved" not in events
    results = [payload for t, _job, payload in emitted if t == "result"]
    assert results[0]["export_strategy"] == "merged_16bit"
    assert "config.json" in results[0]["artifacts"]


def test_adapter_export_keeps_previous_behaviour(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    rc, emitted, _captured = _run_with_export(monkeypatch, tmp_path, "adapter", events)
    assert rc == 0
    assert "adapter_saved" in events
    assert "tokenizer_saved" in events
    assert not [e for e in events if e.startswith("merged:")]
    results = [payload for t, _job, payload in emitted if t == "result"]
    assert results[0]["export_strategy"] == "adapter"


def test_invalid_export_strategy_fails_cleanly(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    rc, emitted, _captured = _run_with_export(monkeypatch, tmp_path, "gguf", events)
    assert rc == 1
    assert "trained" not in events
    errors = [payload for t, _job, payload in emitted if t == "error"]
    assert "Unsupported export_strategy" in errors[0]["message"]



def test_training_settings_reach_sft_config_and_evaluation_dataset(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    eval_dataset = tmp_path / "eval.jsonl"
    eval_dataset.write_text('{"text":"evaluation sample"}\n', encoding="utf-8")
    rc, emitted, captured = _run_with_export(
        monkeypatch,
        tmp_path,
        "adapter",
        events,
        {
            "max_steps": 0,
            "num_train_epochs": 4,
            "warmup_steps": 12,
            "optim": "adamw_torch",
            "save_strategy": "steps",
            "save_steps": 25,
            "eval_strategy": "steps",
            "eval_steps": 10,
            "eval_dataset_path": str(eval_dataset),
        },
    )
    assert rc == 0
    config = captured["config"]
    assert config["max_steps"] == -1
    assert config["num_train_epochs"] == 4
    assert config["warmup_steps"] == 12
    assert config["optim"] == "adamw_torch"
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 25
    assert config["eval_strategy"] == "steps"
    assert config["eval_steps"] == 10
    assert captured["trainer"]["eval_dataset"] is not None


def test_evaluation_requires_dataset_before_engine_import(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    rc, emitted, _captured = _run_with_export(
        monkeypatch,
        tmp_path,
        "adapter",
        events,
        {"eval_strategy": "epoch", "eval_dataset_path": None},
    )
    assert rc == 1
    assert "trained" not in events
    errors = [payload for kind, _job, payload in emitted if kind == "error"]
    assert "evaluation dataset is required" in errors[0]["message"]

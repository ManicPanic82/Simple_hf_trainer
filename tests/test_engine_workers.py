"""Tests for Unsloth and Heretic worker scripts and protocol adherence."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

from workers.heretic_worker import OutputInterceptor
from workers.unsloth_worker import create_sft_config, load_dataset_file, validate_dataset_path


def test_unsloth_load_dataset_file_jsonl(tmp_path: Path):
    ds_file = tmp_path / "ds.jsonl"
    rows = [
        {"instruction": "What is 5+5?", "output": "10"},
        {"text": "Direct prompt text"},
    ]
    with open(ds_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    dataset = load_dataset_file(ds_file)
    assert len(dataset) == 2
    assert "### Instruction:\nWhat is 5+5?" in dataset[0]["text"]
    assert dataset[1]["text"] == "Direct prompt text"


def test_heretic_output_interceptor_parses_trials_and_metrics(monkeypatch):
    emitted = []

    def mock_emit(msg_type, job_id, payload=None, seq=None):
        emitted.append((msg_type, job_id, payload))

    monkeypatch.setattr("workers.heretic_worker.emit", mock_emit)

    real_stdout = MagicMock()
    interceptor = OutputInterceptor("job-test-1", total_trials=5, real_stdout=real_stdout)

    sample_output = """
Running trial 1 of 5...
Refusals: 4/5
KL divergence: 0.0093
Running trial 2 of 5...
Refusals: 3/5
KL divergence: 0.0125
"""
    interceptor.write(sample_output)

    types = [e[0] for e in emitted]
    assert "progress" in types
    assert "metric" in types
    assert "log" in types

    # Check parsed metrics
    metrics = [e[2] for e in emitted if e[0] == "metric"]
    refusals = [m for m in metrics if m["name"] == "refusal_rate"]
    kls = [m for m in metrics if m["name"] == "kl_divergence"]

    assert len(refusals) >= 2
    assert refusals[0]["value"] == 0.8  # 4/5
    assert refusals[1]["value"] == 0.6  # 3/5

    assert len(kls) >= 2
    assert kls[0]["value"] == 0.0093
    assert kls[1]["value"] == 0.0125


def test_unsloth_requires_an_existing_training_dataset(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="Select a prepared training dataset"):
        validate_dataset_path("")

    with pytest.raises(FileNotFoundError, match="not found"):
        validate_dataset_path(tmp_path / "missing.jsonl")

    unsupported = tmp_path / "dataset.txt"
    unsupported.write_text("not a supported dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported training dataset format"):
        validate_dataset_path(unsupported)

    valid = tmp_path / "dataset.jsonl"
    valid.write_text('{"text":"hello"}\n', encoding="utf-8")
    assert validate_dataset_path(valid) == valid


def test_unsloth_rejects_rows_without_training_fields(tmp_path: Path):
    import pytest

    dataset_file = tmp_path / "events.jsonl"
    dataset_file.write_text('{"timestamp":123,"event":"click"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no supported training fields"):
        load_dataset_file(dataset_file)


def test_unsloth_jsonl_reports_invalid_line_number(tmp_path: Path):
    import pytest

    dataset_file = tmp_path / "broken.jsonl"
    dataset_file.write_text('{"text":"valid"}\n{invalid json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL record at line 2"):
        load_dataset_file(dataset_file)


def test_sft_config_supports_max_length_argument():
    class MaxLengthConfig:
        def __init__(self, **kwargs):
            if "max_length" not in kwargs:
                raise TypeError("unexpected max_seq_length")
            self.values = kwargs

    config = create_sft_config(MaxLengthConfig, {"output_dir": "out"}, 512)
    assert config.values["max_length"] == 512


def test_sft_config_supports_max_seq_length_argument():
    class MaxSeqLengthConfig:
        def __init__(self, **kwargs):
            if "max_length" in kwargs:
                raise TypeError("unexpected max_length")
            self.values = kwargs

    config = create_sft_config(MaxSeqLengthConfig, {"output_dir": "out"}, 256)
    assert config.values["max_seq_length"] == 256


def test_sft_config_reports_unsupported_sequence_length_signature():
    import pytest

    class UnsupportedConfig:
        def __init__(self, **kwargs):
            raise TypeError("unsupported sequence length")

    with pytest.raises(RuntimeError, match="Unable to construct installed TRL SFTConfig"):
        create_sft_config(UnsupportedConfig, {"output_dir": "out"}, 128)


def test_unsloth_cancel_listener_starts_after_engine_import(monkeypatch, tmp_path):
    """The stdin reader must not be active while Unsloth's C extensions load.

    On Windows a daemon thread blocked reading an open QProcess stdin pipe
    deadlocks the numpy/torch imports performed by ``import unsloth``. The
    listener is therefore deferred until after engine/model setup and only
    attached immediately before training, where cancellation is meaningful.
    """
    import workers.unsloth_worker as worker_module

    events: list[str] = []
    emitted: list[str] = []
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")

    class FakeUnsloth(types.ModuleType):
        def __getattr__(self, name: str):
            if name == "FastLanguageModel":
                events.append("engine_import")
                return FakeFastLanguageModel
            raise AttributeError(name)

    class FakeModel:
        def save_pretrained(self, path):  # pragma: no cover - stub
            pass

        def save_pretrained_merged(self, path, tokenizer, save_method="merged_16bit"):
            events.append(f"merge:{save_method}")

    class FakeTokenizer:
        def save_pretrained(self, path):  # pragma: no cover - stub
            pass

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            pass

    class FakeSFTTrainer:
        state = types.SimpleNamespace(global_step=1)

        def __init__(self, **kwargs):
            pass

        def train(self):  # pragma: no cover - called by the worker
            events.append("train")
            return types.SimpleNamespace(training_loss=0.5)

    monkeypatch.setitem(sys.modules, "unsloth", FakeUnsloth("unsloth"))
    # The app venv intentionally has no transformers; datasets still probes for
    # PreTrainedTokenizerBase when fingerprinting, so provide both attributes.
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.TrainerCallback = type("TrainerCallback", (), {})
    fake_transformers.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeSFTTrainer),
    )
    monkeypatch.setattr(
        worker_module,
        "emit",
        lambda msg_type, *args, **kwargs: emitted.append(msg_type),
    )
    monkeypatch.setattr(
        worker_module,
        "start_cancel_listener",
        lambda job_id: events.append("listener") or threading.Event(),
    )

    cfg = {
        "job_id": "test-listener-order",
        "model_name": "fake-model",
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "out"),
        "max_steps": 1,
    }
    real_stdout = sys.stdout
    try:
        rc = worker_module.run_worker(cfg)
    finally:
        # run_worker routes engine stdout chatter to stderr; restore pytest's
        # capture stream so the rest of the suite is unaffected.
        sys.stdout = real_stdout

    assert rc == 0
    assert "engine_import" in events
    assert "listener" in events
    assert "train" in events
    assert "merge:merged_16bit" in events  # default export bakes in the adapter
    assert events.index("engine_import") < events.index("listener") < events.index("train")
    assert {"hello", "ready", "job_complete"}.issubset(set(emitted))

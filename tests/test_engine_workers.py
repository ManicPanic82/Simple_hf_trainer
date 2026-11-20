"""Tests for Unsloth and Heretic worker scripts and protocol adherence."""

from __future__ import annotations

import json
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

"""Tests for DatasetService: inspection, previewing, and training preparation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from services.dataset_service import DatasetService, inspect_dataset_file
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService


def test_inspect_dataset_file(tmp_path: Path):
    jsonl_file = tmp_path / "sample.jsonl"
    rows = [
        {"instruction": "Calculate 2+2", "output": "4"},
        {"instruction": "Say hello", "output": "Hello world!"},
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    rec = inspect_dataset_file(jsonl_file)
    assert rec is not None
    assert rec.name == "sample.jsonl"
    assert rec.format == "jsonl"
    assert rec.row_count == 2
    assert "instruction" in rec.columns
    assert "output" in rec.columns
    assert rec.is_prepared is True


def test_preview_dataset_and_validation(tmp_path: Path):
    service = DatasetService(StorageService(AppSettings()))

    # 1. Prepared dataset
    prep_file = tmp_path / "alpaca_clean.jsonl"
    rows = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(5)]
    with open(prep_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    preview = service.preview_dataset(prep_file, num_rows=3)
    assert preview.total_rows == 5
    assert len(preview.rows) == 3
    assert preview.is_valid_for_training is True
    assert "Ready for fine-tuning" in preview.validation_message

    # 2. Raw unformatted dataset
    raw_file = tmp_path / "raw_log.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": 1234, "event": "click"}) + "\n")

    raw_preview = service.preview_dataset(raw_file)
    assert raw_preview.is_valid_for_training is False
    assert "Needs preparation" in raw_preview.validation_message


def test_prepare_dataset(tmp_path: Path):
    service = DatasetService(StorageService(AppSettings()))

    source = tmp_path / "custom.csv"
    source.write_text("prompt,context,answer\nWhat is Python?,code,A language\n", encoding="utf-8")

    out_file = tmp_path / "prepared_custom.jsonl"
    service.prepare_dataset(
        source_path=source,
        output_path=out_file,
        instruction_col="prompt",
        input_col="context",
        output_col="answer",
    )

    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["instruction"] == "What is Python?"
    assert data["input"] == "code"
    assert data["output"] == "A language"
    assert "### Instruction:\nWhat is Python?" in data["text"]


def test_search_hub_datasets_mocked(tmp_path: Path, monkeypatch):
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "m",
            datasets_dir=tmp_path / "d",
            outputs_dir=tmp_path / "o",
            hf_cache_dir=tmp_path / "c",
        )
    )
    service = DatasetService(StorageService(settings))

    mock_ds = MagicMock()
    mock_ds.id = "tatsu-lab/alpaca"
    mock_ds.downloads = 250000
    mock_ds.likes = 4500
    mock_ds.description = "Alpaca instruction dataset"

    mock_api = MagicMock()
    mock_api.list_datasets.return_value = [mock_ds]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: mock_api)

    results = service.search_hub("alpaca")
    assert len(results) == 1
    assert results[0].repo_id == "tatsu-lab/alpaca"
    assert results[0].author == "tatsu-lab"
    assert results[0].downloads == 250000

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



def test_lightweight_dataset_inspection_skips_full_row_count(tmp_path: Path):
    jsonl_file = tmp_path / "large.jsonl"
    jsonl_file.write_text(
        '{"instruction":"first","output":"one"}\\n'
        '{"instruction":"second","output":"two"}\\n',
        encoding="utf-8",
    )

    record = inspect_dataset_file(jsonl_file, lightweight=True)

    assert record is not None
    assert record.row_count is None
    assert record.columns == ["instruction", "output"]

    csv_file = tmp_path / "large.csv"
    csv_file.write_text("prompt,response\\nfirst,one\\nsecond,two\\n", encoding="utf-8")
    csv_record = inspect_dataset_file(csv_file, lightweight=True)

    assert csv_record is not None
    assert csv_record.row_count is None
    assert csv_record.columns == ["prompt", "response"]

    json_file = tmp_path / "small.json"
    json_file.write_text('[{"text":"ready"}]', encoding="utf-8")
    json_record = inspect_dataset_file(json_file, lightweight=True)

    assert json_record is not None
    assert json_record.row_count == 1
    assert json_record.columns == ["text"]

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


def test_arrow_dataset_inspection_preview_and_preparation(tmp_path: Path):
    pa = __import__("pytest").importorskip("pyarrow")
    import pyarrow.ipc as ipc

    source = tmp_path / "sample.arrow"
    table = pa.Table.from_pylist(
        [
            {"instruction": "What is 2+2?", "input": "", "output": "4"},
            {"instruction": "Name a colour", "input": "", "output": "Blue"},
        ]
    )
    with pa.OSFile(str(source), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)

    record = inspect_dataset_file(source)
    assert record is not None
    assert record.format == "arrow"
    assert record.row_count == 2
    assert "instruction" in record.columns

    service = DatasetService(StorageService(AppSettings()))
    preview = service.preview_dataset(source, num_rows=1)
    assert preview.total_rows == 2
    assert len(preview.rows) == 1
    assert preview.is_valid_for_training is True

    output = tmp_path / "prepared_arrow.jsonl"
    service.prepare_dataset(
        source,
        output_path=output,
        instruction_col="instruction",
        input_col="input",
        output_col="output",
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["output"] == "4"


def test_prepare_dataset_rejects_missing_selected_columns(tmp_path: Path):
    import pytest

    source = tmp_path / "raw.csv"
    source.write_text("prompt,answer\nQuestion,Response\n", encoding="utf-8")
    service = DatasetService(StorageService(AppSettings()))

    with pytest.raises(ValueError, match="Selected dataset columns are unavailable"):
        service.prepare_dataset(
            source,
            output_path=tmp_path / "prepared.jsonl",
            instruction_col="missing_prompt",
            output_col="answer",
        )

"""Tests for ModelService: library scanning, inspection, and VRAM estimation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from services.model_service import (
    ModelService,
    estimate_vram_compatibility,
    inspect_model_directory,
)
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService


def test_estimate_vram_compatibility():
    status, detail = estimate_vram_compatibility(500_000_000, 1000, "safetensors")
    assert status == "ok"
    assert "Estimate only" in detail

    status, detail = estimate_vram_compatibility(2_500_000_000, 4000, "safetensors")
    assert status == "warn"
    assert "short context" in detail

    status, detail = estimate_vram_compatibility(7_000_000_000, 14000, "safetensors")
    assert status == "error"
    assert "high risk" in detail

    # Disk footprint alone must never be presented as runtime VRAM.
    for model_format, size in (("gguf", 1500), ("gguf", 5500), ("safetensors", 1000)):
        status, detail = estimate_vram_compatibility(None, size, model_format)
        assert status == "warn"
        assert "not runtime VRAM" in detail or "cannot predict VRAM" in detail


def test_inspect_model_directory_for_full_model(tmp_path: Path):
    model_dir = tmp_path / "qwen-0.5b"
    model_dir.mkdir()
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "vocab_size": 151936,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    record = inspect_model_directory(model_dir)
    assert record is not None
    assert record.name == "qwen-0.5b"
    assert record.kind == "base"
    assert record.format == "safetensors"
    assert record.architecture == "Qwen2ForCausalLM"
    assert record.parameter_count is not None
    assert record.parameter_count < 1_000_000_000
    assert record.vram_status == "ok"


def test_inspect_model_directory_for_adapter(tmp_path: Path):
    adapter_dir = tmp_path / "lora_out"
    adapter_dir.mkdir()
    adapter_cfg = {
        "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "r": 8,
        "lora_alpha": 16,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg), encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 512)

    record = inspect_model_directory(adapter_dir)
    assert record is not None
    assert record.name == "lora_out"
    assert record.kind == "adapter"
    assert "LoRA Adapter" in record.architecture
    assert record.vram_status == "warn"
    assert "base-model" in record.status_detail


def test_inspect_model_directory_empty(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert inspect_model_directory(empty_dir) is None


def test_scan_local_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"
    models_dir.mkdir()
    outputs_dir.mkdir()

    # Create a model in models_dir
    m1 = models_dir / "model-a"
    m1.mkdir()
    (m1 / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

    # Create an adapter in outputs_dir
    a1 = outputs_dir / "adapter-job-1"
    a1.mkdir()
    (a1 / "adapter_config.json").write_text(json.dumps({"r": 4}), encoding="utf-8")

    settings = AppSettings(
        paths=PathSettings(
            models_dir=models_dir,
            datasets_dir=tmp_path / "datasets",
            outputs_dir=outputs_dir,
            hf_cache_dir=tmp_path / "cache",
        )
    )
    storage = StorageService(settings)
    service = ModelService(storage)

    records = service.scan_local_models()
    names = [r.name for r in records]
    assert "model-a" in names
    assert "adapter-job-1" in names


def test_search_hub_mocked(tmp_path: Path, monkeypatch):
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "m",
            datasets_dir=tmp_path / "d",
            outputs_dir=tmp_path / "o",
            hf_cache_dir=tmp_path / "c",
        )
    )
    storage = StorageService(settings)
    service = ModelService(storage)

    mock_model = MagicMock()
    mock_model.id = "Qwen/Qwen2.5-0.5B-Instruct"
    mock_model.downloads = 50000
    mock_model.likes = 1200
    mock_model.pipeline_tag = "text-generation"

    mock_api = MagicMock()
    mock_api.list_models.return_value = [mock_model]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *args, **kwargs: mock_api)

    results = service.search_hub("Qwen")
    assert len(results) == 1
    assert results[0].repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert results[0].author == "Qwen"
    assert results[0].downloads == 50000


def test_model_size_excludes_archived_training_data(tmp_path: Path):
    model_dir = tmp_path / "packaged-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"\x00" * (2 * 1024 * 1024))
    training_data = model_dir / "training_data" / "train"
    training_data.mkdir(parents=True)
    (training_data / "train.jsonl").write_bytes(b"\x00" * (4 * 1024 * 1024))
    (model_dir / "training_config.json").write_text("{}", encoding="utf-8")

    record = inspect_model_directory(model_dir)

    assert record is not None
    assert record.size_mb == 2.0

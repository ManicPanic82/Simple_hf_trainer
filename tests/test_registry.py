"""Filesystem-only tests for the persistent model/dataset catalog."""

from __future__ import annotations

import json
from pathlib import Path

from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.registry import Registry
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService


def test_add_upsert_and_get(tmp_path: Path) -> None:
    catalog = Registry(path=tmp_path / "registry.json")
    model_path = tmp_path / "models" / "one"

    catalog.add_model({"path": model_path, "name": "first", "kind": "base"})
    catalog.add_model({"path": model_path, "name": "renamed", "kind": "adapter"})
    catalog.add_dataset({"path": tmp_path / "data.jsonl", "name": "data"})

    models = catalog.get_models()
    assert len(models) == 1
    assert models[0]["path"] == str(model_path)
    assert models[0]["name"] == "renamed"
    assert models[0]["kind"] == "adapter"

    datasets = catalog.get_datasets()
    assert len(datasets) == 1
    assert datasets[0]["name"] == "data"
    assert isinstance(datasets[0]["path"], str)

    # Records without a usable path are ignored, and returned lists are copies.
    catalog.add_model({"name": "no path"})
    models[0]["name"] = "mutated"
    assert len(catalog.get_models()) == 1
    assert catalog.get_models()[0]["name"] == "renamed"


def test_persistence_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    first = Registry(path=path)
    first.add_model({"path": tmp_path / "models" / "a", "name": "alpha"})
    first.add_dataset({"path": tmp_path / "data" / "b.jsonl", "name": "beta"})
    assert first.save() is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"models", "datasets"}
    assert len(payload["models"]) == 1
    assert len(payload["datasets"]) == 1

    second = Registry(path=path)
    assert [model["name"] for model in second.get_models()] == ["alpha"]
    assert [dataset["name"] for dataset in second.get_datasets()] == ["beta"]
    assert second.issues == []

    # A second save keeps a single entry per path rather than duplicating it.
    second.add_model({"path": tmp_path / "models" / "a", "name": "alpha-updated"})
    second.save()
    reloaded = Registry(path=path)
    assert [model["name"] for model in reloaded.get_models()] == ["alpha-updated"]


def test_corrupt_or_invalid_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"

    path.write_text("{ this is not JSON", encoding="utf-8")
    corrupt = Registry(path=path)
    assert corrupt.get_models() == []
    assert corrupt.get_datasets() == []
    assert corrupt.issues

    path.write_text("[]", encoding="utf-8")
    assert Registry(path=path).get_models() == []

    path.write_text(
        json.dumps(
            {
                "models": [
                    {"name": "no path"},
                    "not-a-dict",
                    42,
                    {"path": str(tmp_path / "good"), "name": "kept"},
                ],
                "datasets": {"path": str(tmp_path / "wrong-shape")},
            }
        ),
        encoding="utf-8",
    )
    recovered = Registry(path=path)
    assert [model["name"] for model in recovered.get_models()] == ["kept"]
    assert recovered.get_datasets() == []

    recovered.add_dataset({"path": tmp_path / "new.jsonl", "name": "new"})
    recovered.save()
    final = Registry(path=path)
    assert [model["name"] for model in final.get_models()] == ["kept"]
    assert [dataset["name"] for dataset in final.get_datasets()] == ["new"]


def test_scan_services_write_records_into_catalog(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"
    datasets_dir = tmp_path / "datasets"
    for folder in (models_dir, outputs_dir, datasets_dir):
        folder.mkdir(parents=True)

    model_dir = models_dir / "model-a"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    (datasets_dir / "sample.jsonl").write_text(
        json.dumps({"instruction": "hi", "output": "hello"}) + "\n", encoding="utf-8"
    )

    settings = AppSettings(
        paths=PathSettings(
            models_dir=models_dir,
            datasets_dir=datasets_dir,
            outputs_dir=outputs_dir,
            hf_cache_dir=tmp_path / "cache",
        )
    )
    storage = StorageService(settings)
    catalog = Registry(path=tmp_path / "catalog.json")

    model_service = ModelService(storage, catalog=catalog)
    dataset_service = DatasetService(storage, catalog=catalog)
    model_records = model_service.scan_local_models()
    dataset_records = dataset_service.scan_local_datasets()

    assert [record.name for record in model_records] == ["model-a"]
    assert [record.name for record in dataset_records] == ["sample.jsonl"]
    assert [record["name"] for record in catalog.get_models()] == ["model-a"]
    assert [record["name"] for record in catalog.get_datasets()] == ["sample.jsonl"]

    payload = json.loads(catalog.path.read_text(encoding="utf-8"))
    assert payload["models"][0]["path"] == str(model_dir)
    assert payload["datasets"][0]["path"] == str(datasets_dir / "sample.jsonl")

    # Re-scanning updates the same records instead of appending duplicates.
    model_service.scan_local_models()
    dataset_service.scan_local_datasets()
    reloaded = Registry(path=catalog.path)
    assert len(reloaded.get_models()) == 1
    assert len(reloaded.get_datasets()) == 1


def test_scan_without_catalog_does_not_create_file(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "models",
            datasets_dir=datasets_dir,
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    service = DatasetService(StorageService(settings))
    assert service.scan_local_datasets() == []
    assert not (tmp_path / "registry.json").exists()

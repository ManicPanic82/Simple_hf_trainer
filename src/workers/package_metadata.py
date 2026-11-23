"""Portable provenance and dataset archiving for exported model packages."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert values to strict JSON-compatible primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_file(source: Path, destination_dir: Path, package_root: Path, role: str) -> dict:
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Dataset source is not a file: {source}")
    package_root = package_root.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return {
        "role": role,
        "source_path": str(source),
        "archived_path": str(destination.relative_to(package_root)),
        "filename": source.name,
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "archived": True,
    }


def write_training_package(
    *,
    output_dir: Path,
    config: dict,
    metrics: list[dict],
    summary: dict,
    train_dataset: Path,
    eval_dataset: Path | None,
) -> dict:
    """Copy exact local datasets and write reproducibility records beside model weights."""
    output_dir = Path(output_dir).resolve()
    data_root = output_dir / "training_data"
    archived = [
        _archive_file(Path(train_dataset), data_root / "train", output_dir, "training"),
    ]
    if eval_dataset is not None:
        archived.append(
            _archive_file(Path(eval_dataset), data_root / "evaluation", output_dir, "evaluation")
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_type": (
            "lora_adapter" if config.get("export_strategy") == "adapter" else "merged_model"
        ),
        "base_model": config.get("model_name"),
        "export_strategy": config.get("export_strategy"),
        "datasets": archived,
    }
    _write_json(data_root / "dataset_manifest.json", manifest)
    _write_json(output_dir / "training_config.json", config)
    _write_json(output_dir / "training_summary.json", summary)
    _write_json(output_dir / "training_metrics.json", {"schema_version": 1, "metrics": metrics})

    adapter_note = (
        "This package contains a LoRA adapter, not a complete standalone model. "
        "Load it together with the base model recorded in training_config.json."
        if config.get("export_strategy") == "adapter"
        else "This package contains the full merged model weights and tokenizer."
    )
    readme = f"""# Fine-tuned model package

{adapter_note}

## Training provenance

- Base model: `{config.get("model_name", "unknown")}`
- Export strategy: `{config.get("export_strategy", "unknown")}`
- Job ID: `{config.get("job_id", "unknown")}`

## Package contents

- Model or adapter files are stored at this directory's root.
- `training_data/train/` contains the exact training dataset file.
- `training_data/evaluation/` contains the evaluation dataset when one was used.
- `training_data/dataset_manifest.json` records source paths, file sizes and SHA-256 hashes.
- `training_config.json` records the settings passed to the training worker.
- `training_summary.json` records the final run outcome and sample counts.
- `training_metrics.json` contains the metric history emitted by the trainer.

Dataset copies are snapshots of the files used for this run. The original source files are left untouched.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_json(output_dir / "package_manifest.json", manifest)
    return manifest


def _dataset_entries(value: Any, section: str = "root") -> list[dict]:
    entries: list[dict] = []
    if isinstance(value, dict):
        if isinstance(value.get("dataset"), str):
            entries.append(
                {
                    "section": section,
                    "dataset": value["dataset"],
                    "split": value.get("split"),
                    "column": value.get("column"),
                }
            )
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                entries.extend(_dataset_entries(child, f"{section}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(_dataset_entries(child, f"{section}[{index}]"))
    return entries


def write_heretic_package(
    *,
    output_dir: Path,
    job_dir: Path,
    config_toml: str,
    metadata: dict,
    metrics: list[dict] | None = None,
    summary: dict | None = None,
) -> dict:
    """Keep Heretic config and dataset references with the exported modified model.

    Hugging Face dataset IDs are recorded, not downloaded wholesale. Local dataset
    files referenced by the TOML are copied into the package.
    """
    import tomllib

    output_dir = Path(output_dir).resolve()
    job_dir = Path(job_dir)
    data_root = output_dir / "training_data"
    data_root.mkdir(parents=True, exist_ok=True)

    parsed = tomllib.loads(config_toml)
    dataset_entries = _dataset_entries(parsed)
    for index, entry in enumerate(dataset_entries, start=1):
        candidate = Path(str(entry["dataset"])).expanduser()
        if candidate.is_file():
            entry.update(
                _archive_file(
                    candidate,
                    data_root / "local_sources" / f"dataset_{index:03d}",
                    output_dir,
                    "prompt_dataset",
                )
            )
        elif candidate.is_dir():
            entry["archived"] = False
            entry["source_type"] = "local_directory_not_copied"
        else:
            entry["archived"] = False
            entry["source_type"] = "hugging_face_dataset_or_identifier"

    _write_json(
        data_root / "dataset_manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datasets": dataset_entries,
            "note": (
                "Hugging Face dataset IDs are recorded with their configured split and column. "
                "They are not downloaded in full because Heretic may stream or select slices "
                "from large remote datasets."
            ),
        },
    )
    (output_dir / "config.toml").write_text(config_toml, encoding="utf-8")
    provenance = job_dir / "run.json"
    if provenance.is_file():
        shutil.copy2(provenance, output_dir / "run.json")
    _write_json(
        output_dir / "heretic_metrics.json",
        {"schema_version": 1, "metrics": metrics or []},
    )
    _write_json(
        output_dir / "heretic_summary.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(summary or {}),
        },
    )
    _write_json(
        output_dir / "package_manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "package_type": "heretic_modified_model",
            **metadata,
            "datasets": dataset_entries,
        },
    )
    readme = f"""# Heretic model package

This folder contains the exported model produced by Heretic.

## Provenance

- Source model: `{metadata.get("source_model", "unknown")}`
- Heretic version: `{metadata.get("heretic_version", "unknown")}`
- Job ID: `{metadata.get("job_id", "unknown")}`
- Export strategy: `{metadata.get("export_strategy", "unknown")}`

## Package contents

- Model files are stored at this directory's root.
- `config.toml` is the exact configuration used for this run.
- `run.json` contains the application's run provenance when available.
- `training_data/dataset_manifest.json` records prompt dataset IDs, splits and columns.
- `heretic_metrics.json` stores parsed scores from the Heretic run.
- `heretic_summary.json` records the run outcome and trial count.
- Local prompt dataset files are copied under `training_data/local_sources/`.

Remote Hugging Face dataset IDs are recorded rather than downloaded in full. This avoids unexpectedly copying large datasets and preserves the exact dataset references and selected splits used by the configuration.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"datasets": dataset_entries}

"""Tests for portable model-package provenance and dataset archiving."""

from __future__ import annotations

import json
from pathlib import Path

from workers.package_metadata import (
    snapshot_training_datasets,
    write_heretic_package,
    write_training_package,
)


def test_training_package_archives_exact_train_and_eval_files(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    train.write_text('{"text":"train example"}\n', encoding="utf-8")
    evaluation.write_text('{"text":"eval example"}\n', encoding="utf-8")
    output = tmp_path / "model-package"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"model")

    manifest = write_training_package(
        output_dir=output,
        config={
            "job_id": "job-1",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "export_strategy": "merged_16bit",
            "learning_rate": 0.0002,
        },
        metrics=[{"step": 1, "loss": 0.5}],
        summary={"status": "completed", "training_rows": 1, "evaluation_rows": 1},
        train_dataset=train,
        eval_dataset=evaluation,
    )

    archived_train = output / "training_data" / "train" / "train.jsonl"
    archived_eval = output / "training_data" / "evaluation" / "eval.jsonl"
    assert archived_train.read_bytes() == train.read_bytes()
    assert archived_eval.read_bytes() == evaluation.read_bytes()
    assert manifest["package_type"] == "merged_model"
    assert manifest["datasets"][0]["sha256"]
    assert (output / "training_config.json").is_file()
    assert (output / "training_summary.json").is_file()
    assert (output / "training_metrics.json").is_file()
    assert "full merged model" in (output / "README.md").read_text(encoding="utf-8")
    assert json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))


def test_adapter_package_readme_explains_base_model_dependency(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"example"}\n', encoding="utf-8")
    output = tmp_path / "adapter-package"
    output.mkdir()

    write_training_package(
        output_dir=output,
        config={
            "job_id": "job-adapter",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "export_strategy": "adapter",
        },
        metrics=[],
        summary={"status": "completed"},
        train_dataset=dataset,
        eval_dataset=None,
    )

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "not a complete standalone model" in readme
    assert (output / "training_data" / "train" / "train.jsonl").is_file()
    assert not (output / "training_data" / "evaluation").exists()


def test_heretic_package_records_remote_sources_and_copies_local_files(tmp_path: Path) -> None:
    local_prompts = tmp_path / "prompts.jsonl"
    local_prompts.write_text('{"text":"local prompt"}\n', encoding="utf-8")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "run.json").write_text('{"job_id":"heretic-1"}\n', encoding="utf-8")
    output = tmp_path / "heretic-model"
    output.mkdir()

    config = f'''
model = "Qwen/Qwen2.5-1.5B-Instruct"

[scorer.KeywordRate.prompts]
dataset = "org/harmful-prompts"
split = "test[:5]"
column = "text"

[modifier.Abliteration.good_prompts]
dataset = "{local_prompts.as_posix()}"
split = "train"
column = "text"
'''
    result = write_heretic_package(
        output_dir=output,
        job_dir=job_dir,
        config_toml=config,
        metadata={"job_id": "heretic-1", "source_model": "Qwen/Qwen2.5-1.5B-Instruct"},
    )

    assert len(result["datasets"]) == 2
    assert any(not item["archived"] for item in result["datasets"])
    assert any(item["archived"] for item in result["datasets"])
    assert (output / "training_data" / "dataset_manifest.json").is_file()
    assert (output / "config.toml").is_file()
    assert (output / "run.json").is_file()
    assert (output / "heretic_metrics.json").is_file()
    assert (output / "heretic_summary.json").is_file()
    assert (output / "README.md").is_file()


def test_training_uses_a_snapshot_and_records_original_dataset_path(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"original"}\n', encoding="utf-8")
    output = tmp_path / "model"
    output.mkdir()
    train_copy, eval_copy = snapshot_training_datasets(output, source, None)
    source.write_text('{"text":"changed after start"}\n', encoding="utf-8")
    assert train_copy.read_text(encoding="utf-8") == '{"text":"original"}\n'
    manifest = write_training_package(
        output_dir=output,
        config={"model_name": "org/model", "export_strategy": "adapter"},
        metrics=[],
        summary={"status": "completed"},
        train_dataset=train_copy,
        eval_dataset=eval_copy,
        train_dataset_source=source,
    )
    archived = output / "training_data" / "train" / "train.jsonl"
    assert archived.read_text(encoding="utf-8") == '{"text":"original"}\n'
    assert manifest["datasets"][0]["source_path"] == str(source.resolve())


def test_full_finetuning_package_is_identified_as_standalone_model(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"example"}\n', encoding="utf-8")
    output = tmp_path / "full-model-package"
    output.mkdir()

    manifest = write_training_package(
        output_dir=output,
        config={
            "job_id": "job-full",
            "model_name": "Qwen/Qwen3.5-2B",
            "training_method": "full_finetune",
            "precision": "fp16",
            "optim": "adamw_8bit",
            "export_strategy": "full_model",
        },
        metrics=[],
        summary={"status": "completed"},
        train_dataset=dataset,
        eval_dataset=None,
    )

    readme = (output / "README.md").read_text(encoding="utf-8")
    assert manifest["package_type"] == "full_finetuned_model"
    assert "complete model weights updated by full fine-tuning" in readme
    assert "full_finetune" in readme
    assert "fp16" in readme
    assert "adamw_8bit" in readme

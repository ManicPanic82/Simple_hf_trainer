"""Regression tests for the Analytics and Ollama Export workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from app.state import AppState
from services.hardware_monitor import GpuStats, HardwareSnapshot
from ui.analytics_tab import AnalyticsTab
from ui.ollama_export_tab import OllamaExportTab
from workers.job_manager import JobManager


def test_analytics_receives_real_training_metrics(qapp, tmp_path: Path) -> None:
    state = AppState()
    manager = JobManager(history_file=tmp_path / "jobs.json")
    tab = AnalyticsTab(state, manager)

    state.publish_metric("loss", 1, 1.25, "unsloth-1")
    state.publish_metric("eval_loss", 1, 1.4, "unsloth-1")
    state.publish_metric("learning_rate", 1, 0.0002, "unsloth-1")
    state.publish_metric("grad_norm", 1, 0.6, "unsloth-1")
    state.publish_metric("tokens_per_second", 1, 42.0, "unsloth-1")
    state.publish_metric("eval_accuracy", 1, 0.91, "unsloth-1")
    state.publish_hardware(
        HardwareSnapshot(
            gpu=GpuStats(
                name="Test GPU", utilization_percent=72,
                memory_used_mb=4096, memory_total_mb=6144,
            ),
            ram_used_bytes=8 * 1024**3,
            ram_total_bytes=32 * 1024**3,
            captured_at=10.0,
        )
    )
    job = manager.submit("unsloth", "org/model")
    manager.mark_succeeded(job.job_id, message="test complete")
    tab.refresh_history()

    assert tab.loss_chart.points["loss"] == [(1.0, 1.25)]
    assert tab.loss_chart.points["eval_loss"] == [(1.0, 1.4)]
    assert tab.lr_chart.points["learning_rate"] == [(1.0, 0.0002)]
    assert tab.gradient_chart.points["grad_norm"] == [(1.0, 0.6)]
    assert tab.throughput_chart.points["tokens_per_second"] == [(1.0, 42.0)]
    assert tab.evaluation_chart.points["eval_accuracy"] == [(1.0, 0.91)]
    assert tab.gpu_chart.points["GPU %"] == [(10.0, 72.0)]
    assert tab.gpu_chart.points["VRAM %"] == [(10.0, 4096 / 6144 * 100)]
    assert tab.history_table.rowCount() == 1
    assert "unsloth-1" in tab.overview.text()


def test_ollama_export_rejects_adapter_without_starting_process(qapp, tmp_path: Path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")

    tab = OllamaExportTab()
    tab.source_edit.setText(str(source))
    tab.start_export()

    assert not tab.worker_running
    assert tab.status_row.value.text() == "Failed"
    assert "Export a merged 16-bit model" in tab.log.text()


def test_ollama_export_requires_hugging_face_model_directory(qapp, tmp_path: Path) -> None:
    tab = OllamaExportTab()
    tab.source_edit.setText(str(tmp_path / "missing-model"))
    tab.start_export()

    assert not tab.worker_running
    assert tab.status_row.value.text() == "Failed"
    assert "config.json" in tab.log.text()

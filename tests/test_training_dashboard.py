"""Tests for metric widgets and the dedicated Analytics workspace."""

from __future__ import annotations

from pathlib import Path

from app.state import AppState
from services.hardware_monitor import GpuStats, HardwareSnapshot
from services.settings_service import AppSettings, PathSettings
from services.storage_service import StorageService
from ui.analytics_tab import AnalyticsTab
from ui.training_dashboard import ArchitectureView, MetricChart, TrainingDashboard
from ui.training_tab import TrainingTab
from workers.job_manager import JobManager
from workers.protocol import MessageType, WorkerMessage


def test_metric_chart_keeps_only_finite_reported_points(qapp) -> None:
    chart = MetricChart("Loss", ("loss", "eval_loss"))
    chart.add_point("loss", 1, 1.25)
    chart.add_point("loss", 2, 0.85)
    chart.add_point("eval_loss", 2, 1.1)
    chart.add_point("loss", 3, float("nan"))
    chart.add_point("unknown", 4, 0.2)

    assert chart.history["loss"] == [(1, 1.25), (2, 0.85)]
    assert chart.history["eval_loss"] == [(2, 1.1)]


def test_architecture_view_accepts_loaded_model_metadata(qapp) -> None:
    view = ArchitectureView()
    view.set_architecture(
        {
            "model_type": "llama",
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "lora_rank": 8,
            "target_modules": "q_proj, v_proj",
        }
    )

    assert view.metadata["model_type"] == "llama"
    assert view.metadata["num_hidden_layers"] == 24
    assert "target_modules" in view.metadata


def test_dashboard_tracks_metrics_timing_and_hardware(qapp) -> None:
    dashboard = TrainingDashboard()
    dashboard.set_architecture(
        {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584}
    )
    dashboard.add_metric("loss", 1, 1.2)
    dashboard.add_metric("eval_loss", 1, 1.4)
    dashboard.add_metric("learning_rate", 1, 0.0002)
    dashboard.add_metric("epoch", 1, 0.5)
    dashboard.add_metric("tokens_per_second", 1, 42.0)
    dashboard.begin_run()
    dashboard.set_progress(0.5)
    dashboard.set_hardware(
        HardwareSnapshot(
            gpu=GpuStats(
                name="Test GPU",
                utilization_percent=75,
                memory_used_mb=4096,
                memory_total_mb=6144,
            ),
            ram_used_bytes=8 * 1024**3,
            ram_total_bytes=32 * 1024**3,
        )
    )

    assert dashboard.loss_chart.history["loss"] == [(1, 1.2)]
    assert dashboard.loss_chart.history["eval_loss"] == [(1, 1.4)]
    assert dashboard.lr_chart.history["learning_rate"] == [(1, 0.0002)]
    assert dashboard.stat_labels["epoch"].text() == "Epoch: 0.50"
    assert dashboard.stat_labels["tokens_per_second"].text() == "Tokens/sec: 42.0"
    assert dashboard.stat_labels["gpu_utilization"].text() == "GPU utilisation: 75%"
    assert dashboard.stat_labels["gpu_memory"].text() == "GPU memory: 4.00 / 6.00 GB"
    assert dashboard.stat_labels["ram"].text() == "System RAM: 8.0 / 32.0 GB"
    dashboard.finish_run()
    assert dashboard.stat_labels["remaining"].text() == "Est. remaining: Complete"
    dashboard.reset()
    assert dashboard.architecture.metadata == {}
    assert dashboard.loss_chart.history["loss"] == []


def test_training_worker_messages_feed_dedicated_analytics_tab(qapp, tmp_path: Path) -> None:
    settings = AppSettings(
        paths=PathSettings(
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            outputs_dir=tmp_path / "outputs",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    state = AppState()
    tab = TrainingTab(state, StorageService(settings))
    analytics = AnalyticsTab(state, JobManager(history_file=tmp_path / "jobs.json"))
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.ARCHITECTURE,
            job_id="job-1",
            payload={"architecture": {"model_type": "llama", "num_hidden_layers": 12}},
        )
    )
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.METRIC,
            job_id="job-1",
            payload={"name": "loss", "value": 0.75, "step": 3},
        )
    )

    assert analytics.architecture.metadata["model_type"] == "llama"
    assert analytics.loss_chart.points["loss"] == [(3.0, 0.75)]
    assert not hasattr(tab, "dashboard")

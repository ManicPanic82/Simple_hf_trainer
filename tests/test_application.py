"""Startup, tab structure, and shared status strip tests."""

from __future__ import annotations

import pytest

from app.application import create_window
from services.engine_registry import EngineState, EngineStatus
from services.hardware_monitor import GpuStats, HardwareSnapshot
from ui.main_window import MainWindow
from workers.job_manager import JobSnapshot, JobState


@pytest.fixture
def window(context):
    window = create_window(context)
    yield window
    window.close()
    window.deleteLater()


def test_main_window_has_exactly_six_tabs_in_order(window: MainWindow) -> None:
    assert window.tabs.count() == 6
    assert window.tab_names() == [
        "Models", "Datasets", "Training", "Analytics", "Heretic", "Ollama Export"
    ]


def test_tabs_are_navigable(window: MainWindow) -> None:
    for index, name in enumerate(MainWindow.TAB_NAMES):
        window.tabs.setCurrentIndex(index)
        assert window.tabs.currentIndex() == index
        assert window.tabs.tabText(index) == name

    window.tabs.setCurrentWidget(window.heretic_tab)
    assert window.tabs.currentIndex() == 4
    window.tabs.setCurrentWidget(window.models_tab)
    assert window.tabs.currentIndex() == 0


def test_window_shows_and_stays_responsive(window: MainWindow) -> None:
    window.show()
    assert window.isVisible()
    assert window.windowTitle() == "Simple HF Trainer"
    assert window.minimumWidth() > 0


def test_missing_engines_do_not_block_startup(window: MainWindow, context) -> None:
    context.state.publish_engines(
        {
            "unsloth": EngineStatus(
                "unsloth", EngineState.INTERPRETER_MISSING, None, "No engine interpreter found."
            ),
            "heretic": EngineStatus(
                "heretic", EngineState.NOT_INSTALLED, None, "heretic is not installed."
            ),
        }
    )
    assert "set up" in window.unsloth_pill.text().lower()
    assert "not installed" in window.heretic_pill.text().lower()
    assert window.training_tab.start_button.isEnabled() is False
    assert window.heretic_tab.run_button.isEnabled() is False


def test_status_strip_reports_hardware(window: MainWindow, context) -> None:
    snapshot = HardwareSnapshot(
        gpu=GpuStats(
            name="NVIDIA GeForce GTX 1660",
            driver_version="617.42",
            utilization_percent=37.0,
            memory_used_mb=1024.0,
            memory_total_mb=6144.0,
            temperature_c=52.0,
        ),
        ram_used_bytes=8 * 1024**3,
        ram_total_bytes=32 * 1024**3,
        captured_at=0.0,
    )
    context.state.publish_hardware(snapshot)

    strip = window.status_strip
    assert "GTX 1660" in strip.gpu_value.text()
    assert strip.util_value.text() == "37 %"
    assert strip.vram_value.text() == "1.0 GB / 6.0 GB"
    assert strip.vram_bar.value() == 16
    assert strip.ram_bar.value() == 25


def test_status_strip_degrades_when_hardware_is_unavailable(window: MainWindow, context) -> None:
    snapshot = HardwareSnapshot(
        gpu=None,
        gpu_error="nvidia-smi was not found. Install or repair the NVIDIA driver.",
        ram_error="psutil is not installed; RAM usage is unavailable.",
        captured_at=0.0,
    )
    context.state.publish_hardware(snapshot)

    strip = window.status_strip
    assert strip.gpu_value.text() == "Unavailable"
    assert strip.vram_value.text() == "—"
    assert strip.ram_value.text() == "—"
    assert "nvidia-smi" in strip.gpu_value.toolTip()


def test_gpu_lock_state_is_visible(window: MainWindow, context) -> None:
    job = context.job_manager.submit("unsloth", "Qwen/Qwen2.5-0.5B-Instruct")
    assert window.status_strip.lock_pill.text().startswith("GPU busy")
    assert context.job_manager.gpu_owner == job.job_id

    context.job_manager.request_cancel(job.job_id)
    context.job_manager.handle_worker_exit(job.job_id, -1)
    assert window.status_strip.lock_pill.text() == "GPU free"


def test_second_engine_is_queued_while_gpu_is_busy(window: MainWindow, context) -> None:
    started: list[JobSnapshot] = []
    context.job_manager.job_start_requested.connect(started.append)

    first = context.job_manager.submit("unsloth")
    queued = context.job_manager.submit("heretic")

    assert queued.state is JobState.QUEUED
    assert context.job_manager.gpu_owner == first.job_id
    assert [job.job_id for job in started] == [queued.job_id]

    # Completion of the first job promotes the second one automatically.
    context.job_manager.mark_succeeded(first.job_id)
    promoted = context.job_manager.snapshot(queued.job_id)
    assert promoted is not None
    assert promoted.state is JobState.STARTING
    assert context.job_manager.gpu_owner == queued.job_id
    assert [job.state for job in started] == [JobState.QUEUED, JobState.STARTING]

    # Clean up both terminal records.
    context.job_manager.mark_failed(queued.job_id, message="test cleanup")


def test_settings_issue_is_reported_in_banner(context, qapp) -> None:
    context.settings_service.issues = ["Settings file could not be read; using defaults."]
    window = create_window(context)
    try:
        assert window.banner.isVisible() or window.banner.label.text()
    finally:
        window.close()

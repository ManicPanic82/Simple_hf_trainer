"""Hardware monitoring: parsing, degradation, and asynchronous polling."""

from __future__ import annotations

import pytest

from services.hardware_monitor import (
    CommandResult,
    GpuStats,
    HardwareMonitor,
    HardwareProbeError,
    HardwareSnapshot,
    NvidiaSmiProbe,
    PsutilRamProbe,
    collect_snapshot,
    parse_nvidia_smi_csv,
)

TYPICAL = "NVIDIA GeForce GTX 1660, 617.42, 784, 6144, 5, 45"


class FakeGpu:
    def __init__(self, stats: GpuStats | None = None, error: Exception | None = None) -> None:
        self.stats = stats
        self.error = error

    def read(self) -> GpuStats:
        if self.error:
            raise self.error
        assert self.stats is not None
        return self.stats


class FakeRam:
    def __init__(self, used: int = 0, total: int = 0, error: Exception | None = None) -> None:
        self.used = used
        self.total = total
        self.error = error

    def read(self) -> tuple[int, int]:
        if self.error:
            raise self.error
        return self.used, self.total


def test_parse_typical_row() -> None:
    gpu = parse_nvidia_smi_csv(TYPICAL)
    assert gpu.name == "NVIDIA GeForce GTX 1660"
    assert gpu.driver_version == "617.42"
    assert gpu.memory_used_mb == 784
    assert gpu.memory_total_mb == 6144
    assert gpu.utilization_percent == 5
    assert gpu.temperature_c == 45


def test_parse_tolerates_placeholders_and_units() -> None:
    gpu = parse_nvidia_smi_csv(
        "NVIDIA GeForce GTX 1660, 617.42, 1024 MiB, 6144 MiB, [N/A], [Invalid Argument]"
    )
    assert gpu.memory_used_mb == 1024
    assert gpu.utilization_percent is None
    assert gpu.temperature_c is None


def test_parse_first_gpu_of_many() -> None:
    gpu = parse_nvidia_smi_csv(TYPICAL + "\nNVIDIA GeForce RTX 4090, 617.42, 100, 24576, 0, 30")
    assert gpu.name == "NVIDIA GeForce GTX 1660"


def test_parse_empty_output_raises() -> None:
    with pytest.raises(HardwareProbeError):
        parse_nvidia_smi_csv("  \n ")


def test_probe_reports_missing_nvidia_smi() -> None:
    def runner(args, timeout):
        raise FileNotFoundError("nvidia-smi")

    probe = NvidiaSmiProbe(runner=runner)
    with pytest.raises(HardwareProbeError) as caught:
        probe.read()
    assert "nvidia-smi was not found" in str(caught.value)


def test_probe_reports_driver_error() -> None:
    def runner(args, timeout):
        return CommandResult(9, "", "NVIDIA-SMI has failed because it couldn't communicate")

    probe = NvidiaSmiProbe(runner=runner)
    with pytest.raises(HardwareProbeError) as caught:
        probe.read()
    assert "nvidia-smi failed" in str(caught.value)


def test_probe_passes_expected_query() -> None:
    captured: dict[str, list[str]] = {}

    def runner(args, timeout):
        captured["args"] = list(args)
        return CommandResult(0, TYPICAL, "")

    NvidiaSmiProbe(runner=runner).read()
    assert any("--query-gpu=" in arg for arg in captured["args"])
    assert "--format=csv,noheader,nounits" in captured["args"]


def test_snapshot_percentages() -> None:
    snapshot = HardwareSnapshot(
        gpu=GpuStats("GPU", memory_used_mb=1536, memory_total_mb=6144),
        ram_used_bytes=16 * 1024**3,
        ram_total_bytes=32 * 1024**3,
    )
    assert snapshot.vram_percent == 25
    assert snapshot.ram_percent == 50


def test_collect_snapshot_success() -> None:
    gpu = GpuStats("NVIDIA GeForce GTX 1660", memory_used_mb=100, memory_total_mb=6144)
    snapshot = collect_snapshot(FakeGpu(gpu), FakeRam(1024, 2048), clock=lambda: 12.0)
    assert snapshot.gpu == gpu
    assert snapshot.gpu_error is None
    assert snapshot.ram_used_bytes == 1024
    assert snapshot.captured_at == 12.0
    assert snapshot.gpu_available is True


def test_collect_snapshot_handles_probe_errors() -> None:
    snapshot = collect_snapshot(
        FakeGpu(error=HardwareProbeError("driver missing")),
        FakeRam(error=HardwareProbeError("psutil missing")),
    )
    assert snapshot.gpu is None
    assert snapshot.gpu_error == "driver missing"
    assert snapshot.ram_error == "psutil missing"
    assert snapshot.gpu_available is False


def test_collect_snapshot_handles_unexpected_exceptions() -> None:
    snapshot = collect_snapshot(
        FakeGpu(error=RuntimeError("weird driver")), FakeRam(error=ValueError("nope"))
    )
    assert snapshot.gpu_error is not None
    assert "Unexpected GPU probe failure" in snapshot.gpu_error
    assert snapshot.ram_error is not None


def test_monitor_emits_off_thread(qapp, wait_for) -> None:
    gpu = GpuStats("NVIDIA GeForce GTX 1660", memory_total_mb=6144)
    monitor = HardwareMonitor(FakeGpu(gpu), FakeRam(1, 2), interval_ms=500)
    received: list[HardwareSnapshot] = []
    monitor.snapshot_ready.connect(received.append)
    try:
        monitor.refresh()
        assert wait_for(lambda: bool(received))
        assert received[-1].gpu == gpu
        assert monitor.last_snapshot is received[-1]
    finally:
        monitor.stop()


def test_monitor_skips_overlapping_refreshes(qapp, wait_for) -> None:
    monitor = HardwareMonitor(FakeGpu(GpuStats("GPU")), FakeRam(1, 2))
    received: list[HardwareSnapshot] = []
    monitor.snapshot_ready.connect(received.append)
    try:
        monitor.refresh()
        monitor.refresh()
        assert wait_for(lambda: bool(received))
    finally:
        monitor.stop()


def test_psutil_probe_returns_numbers() -> None:
    try:
        used, total = PsutilRamProbe().read()
    except HardwareProbeError:
        pytest.skip("psutil is not installed in this interpreter")
    assert total > 0
    assert 0 <= used <= total

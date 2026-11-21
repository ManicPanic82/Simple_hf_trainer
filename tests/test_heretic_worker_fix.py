"""Regression tests for the Heretic worker protocol stream."""

from __future__ import annotations

import io
import sys
import types

import workers.heretic_worker as heretic_worker


def test_heretic_emit_uses_captured_stream_when_stdout_is_intercepted(monkeypatch):
    """emit() must not recurse through OutputInterceptor into itself."""
    protocol_stream = io.StringIO()
    monkeypatch.setattr(heretic_worker, "_PROTOCOL_STREAM", protocol_stream)
    real_stdout = sys.stdout
    interceptor = heretic_worker.OutputInterceptor("job-1", 1, real_stdout)
    try:
        sys.stdout = interceptor
        heretic_worker.emit("log", "job-1", {"message": "hello"})
    finally:
        sys.stdout = real_stdout

    output = protocol_stream.getvalue()
    assert output.count("\n") == 1
    assert '"type": "log"' in output
    assert '"message": "hello"' in output


def test_heretic_interceptor_parsing_does_not_recurse(monkeypatch):
    """Intercepted engine output is parsed into protocol lines, not re-emitted."""
    protocol_stream = io.StringIO()
    monkeypatch.setattr(heretic_worker, "_PROTOCOL_STREAM", protocol_stream)
    real_stdout = sys.stdout
    interceptor = heretic_worker.OutputInterceptor("job-1", 5, real_stdout)
    try:
        sys.stdout = interceptor
        interceptor.write("Running trial 2 of 5\nRefusals: 2/5\nKL divergence: 0.01\n")
    finally:
        sys.stdout = real_stdout

    output = protocol_stream.getvalue()
    assert '"type": "progress"' in output
    assert '"type": "metric"' in output
    assert output.count("\n") == 6  # each parsed line emits a metric/progress + a log


def test_heretic_cancel_listener_starts_after_engine_import(monkeypatch, tmp_path):
    """Heretic must not block stdin while its C extensions are still loading."""
    events: list[str] = []

    class FakeHeretic(types.ModuleType):
        def __getattr__(self, name: str):
            if name == "__version__":
                events.append("engine_import")
                return "test"
            raise AttributeError(name)

    fake_main = types.ModuleType("heretic.main")

    def fake_heretic_main():
        events.append("engine_main")
        return None

    fake_main.main = fake_heretic_main
    monkeypatch.setitem(sys.modules, "heretic", FakeHeretic("heretic"))
    monkeypatch.setitem(sys.modules, "heretic.main", fake_main)
    monkeypatch.setattr(heretic_worker, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(heretic_worker, "install_dummy_prompt_output", lambda: None)
    monkeypatch.setattr(
        heretic_worker, "start_cancel_listener", lambda job_id: events.append("listener")
    )

    rc = heretic_worker.run_worker(
        {
            "job_id": "order-test",
            "model_name": "fake-model",
            "trials": 2,
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert rc == 0
    assert events == ["engine_import", "listener", "engine_main"]

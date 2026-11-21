"""Worker behaviour with a complete GUI-generated Heretic 2.x config."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import workers.heretic_worker as heretic_worker
from services.heretic_config import HereticConfig


def _patch_engine(monkeypatch, seen: dict) -> None:
    fake_package = types.ModuleType("heretic")
    fake_package.__version__ = "2.1.0"
    fake_main = types.ModuleType("heretic.main")

    def fake_run() -> None:
        path = Path(os.environ["HERETIC_CONFIG_FILE"])
        seen["text"] = path.read_text(encoding="utf-8")
        return None

    fake_main.main = fake_run
    monkeypatch.setitem(sys.modules, "heretic", fake_package)
    monkeypatch.setitem(sys.modules, "heretic.main", fake_main)
    monkeypatch.setattr(heretic_worker, "install_dummy_prompt_output", lambda: None)
    monkeypatch.setattr(heretic_worker, "start_cancel_listener", lambda job_id: None)


def _collect(monkeypatch) -> list[tuple]:
    emitted: list[tuple] = []
    monkeypatch.setattr(
        heretic_worker,
        "emit",
        lambda msg_type, job_id=None, payload=None, seq=None: emitted.append(
            (msg_type, job_id, payload, seq)
        ),
    )
    return emitted


def test_generated_toml_is_passed_verbatim_to_engine(monkeypatch, tmp_path: Path) -> None:
    cfg = HereticConfig.compact()
    cfg.model = "org/model"
    text = cfg.headless_copy(tmp_path / "out", tmp_path / "checkpoints").to_toml()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "model.safetensors").write_text("weights", encoding="utf-8")
    seen: dict = {}
    _patch_engine(monkeypatch, seen)
    emitted = _collect(monkeypatch)

    rc = heretic_worker.run_worker(
        {"job_id": "full-1", "toml": text, "output_dir": str(tmp_path / "out")}
    )

    assert rc == 0
    assert seen["text"] == text
    complete = [entry for entry in emitted if entry[0] == "job_complete"]
    assert complete[-1][2]["success"] is True
    result = [entry for entry in emitted if entry[0] == "result"]
    assert result[-1][2]["artifacts"] == ["model.safetensors"]


def test_missing_saved_model_is_reported_as_failure(monkeypatch, tmp_path: Path) -> None:
    cfg = HereticConfig.compact()
    cfg.model = "org/model"
    text = cfg.headless_copy(tmp_path / "empty", tmp_path / "checkpoints").to_toml()
    (tmp_path / "empty").mkdir()
    seen: dict = {}
    _patch_engine(monkeypatch, seen)
    emitted = _collect(monkeypatch)

    rc = heretic_worker.run_worker(
        {"job_id": "full-2", "toml": text, "output_dir": str(tmp_path / "empty")}
    )

    assert rc == 1
    complete = [entry for entry in emitted if entry[0] == "job_complete"]
    assert complete[-1][2]["success"] is False
    assert "no saved model" in complete[-1][2]["message"]


def test_generic_score_lines_become_metrics(monkeypatch) -> None:
    emitted = _collect(monkeypatch)
    interceptor = heretic_worker.OutputInterceptor("j", 3, sys.stdout)
    interceptor.write("  * Minimize: 3/10\n  * PIQA acc_norm: 0.7123\n")
    metrics = {entry[2]["name"]: entry[2]["value"] for entry in emitted if entry[0] == "metric"}
    assert metrics["minimize"] == 0.3
    assert metrics["piqa_acc_norm"] == 0.7123

    emitted.clear()
    interceptor.write("* CUDA Version: 12.6\n* Driver Version: 617.42\n")
    assert not [entry for entry in emitted if entry[0] == "metric"]

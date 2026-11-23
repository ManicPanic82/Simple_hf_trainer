"""Training -> Models hand-off and non-blocking compatibility advisory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.engine_registry import UNSLOTH, EngineState, EngineStatus
from services.hardware_monitor import GpuStats, HardwareSnapshot
from ui.training_tab import TrainingTab
from workers.job_manager import JobState
from workers.protocol import MessageType, WorkerMessage


def _make_tab(
    tmp_path: Path, *, job_manager: MagicMock, model_service: MagicMock, state: MagicMock
):
    """Build a TrainingTab with mocked services and an isolated real storage path."""
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    storage = MagicMock()
    storage.validate.return_value = SimpleNamespace(path=outputs, issues=[], ok=True, exists=True)
    storage.path_for.return_value = outputs

    registry = MagicMock()
    registry.status.return_value = EngineStatus(
        UNSLOTH, EngineState.AVAILABLE, None, "unsloth 1.0", "1.0"
    )
    state.engines = {}
    state.models = []
    state.hardware = None

    return TrainingTab(
        state,
        storage,
        job_manager=job_manager,
        registry=registry,
        model_service=model_service,
        dataset_service=MagicMock(),
    )


def _worker_messages(job_id: str, output_dir: Path) -> list[WorkerMessage]:
    return [
        WorkerMessage(type=MessageType.HELLO, payload={"pid": 1234}),
        WorkerMessage(type=MessageType.READY, payload={"engine": "unsloth", "version": "1.0"}),
        WorkerMessage(type=MessageType.LOG, payload={"message": "LoRA training started"}),
        WorkerMessage(type=MessageType.PROGRESS, payload={"fraction": 0.5, "step": 3, "total": 5}),
        WorkerMessage(type=MessageType.RESULT, payload={"output_dir": str(output_dir)}),
        WorkerMessage(
            type=MessageType.JOB_COMPLETE,
            payload={"success": True, "message": "Training finished."},
            job_id=job_id,
        ),
    ]


def test_successful_job_publishes_scanned_models(qapp, tmp_path: Path) -> None:
    state = MagicMock()
    model_service = MagicMock()
    job_manager = MagicMock()
    scanned = [SimpleNamespace(name="unsloth_2026_adapter", kind="adapter")]
    model_service.scan_local_models.return_value = scanned

    tab = _make_tab(tmp_path, job_manager=job_manager, model_service=model_service, state=state)

    job_id = "job-1"
    tab._active_job_id = job_id
    output_dir = tmp_path / "outputs" / "unsloth_job-1" / "adapter"
    for message in _worker_messages(job_id, output_dir):
        tab._on_worker_message(message)

    # The RESULT message still drives the "Last adapter" row.
    assert tab.last_adapter_row.value.text() == str(output_dir)

    tab._on_worker_exited(0)

    model_service.scan_local_models.assert_called_once_with()
    state.publish_models.assert_called_once_with(scanned)
    state.notify.assert_any_call("info", "LoRA fine-tuning completed successfully!")
    assert tab._active_job_id is None
    assert tab.start_button.isEnabled()


def test_unknown_hardware_advisory_does_not_block_start(qapp, tmp_path: Path, monkeypatch) -> None:
    state = MagicMock()
    model_service = MagicMock()
    job_manager = MagicMock()
    job_manager.submit.return_value = SimpleNamespace(job_id="job-2", state=JobState.STARTING)

    tab = _make_tab(tmp_path, job_manager=job_manager, model_service=model_service, state=state)

    # Engine availability gating is intact: an available engine enables Start.
    tab.apply_engines({UNSLOTH: EngineStatus(UNSLOTH, EngineState.AVAILABLE, None, "ok")})
    assert tab.start_button.isEnabled()

    # Unknown hardware and no locally verified model: advisory only.
    state.hardware = None
    state.models = []

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    interpreter = tmp_path / "python.exe"
    interpreter.write_text("", encoding="utf-8")
    tab.dataset_combo.setEditText(str(dataset))
    tab.model_combo.setEditText("unsloth/Qwen2.5-0.5B-Instruct")
    tab._registry.status.return_value = EngineStatus(
        UNSLOTH, EngineState.AVAILABLE, interpreter, "unsloth 1.0", "1.0"
    )

    worker = MagicMock()
    worker.running = False
    worker.pid = 1234
    monkeypatch.setattr("ui.training_tab.WorkerProcess", lambda **kwargs: worker)

    tab._start_training()

    # The job was submitted: unknown hardware never blocked Start.
    job_manager.submit.assert_called_once()
    log = tab.log_view.toPlainText()
    assert "Compatibility not verified for this model/GPU" in log
    assert "Unsloth will report incompatibilities at load time" in log
    assert "Compatibility not verified" in tab.start_button.toolTip()

    # The advisory leaves no terminal verdict and does not permanently disable Start.
    tab._active_job_id = "job-2"
    tab._on_worker_message(
        WorkerMessage(
            type=MessageType.JOB_COMPLETE,
            payload={"success": True, "message": "Training finished."},
            job_id="job-2",
        )
    )
    tab._on_worker_exited(0)
    assert tab.start_button.isEnabled()
    assert "incompatible" not in log.lower()


def test_no_advisory_when_model_and_gpu_are_known(qapp, tmp_path: Path) -> None:
    state = MagicMock()
    state.hardware = HardwareSnapshot(gpu=GpuStats(name="NVIDIA GeForce RTX 4090"))
    state.models = [
        SimpleNamespace(
            path=tmp_path / "models" / "qwen",
            name="qwen",
            architecture="Qwen2ForCausalLM",
        )
    ]

    known_model = state.models[0]
    tab = _make_tab(
        tmp_path,
        job_manager=MagicMock(),
        model_service=MagicMock(),
        state=state,
    )
    state.hardware = HardwareSnapshot(gpu=GpuStats(name="NVIDIA GeForce RTX 4090"))
    state.models = [known_model]

    assert tab._compatibility_advisory(str(known_model.path)) is None


def test_export_combo_feeds_config(qapp, tmp_path: Path) -> None:
    tab = _make_tab(
        tmp_path,
        job_manager=MagicMock(),
        model_service=MagicMock(),
        state=MagicMock(),
    )
    assert tab.export_combo.count() == 2
    assert tab.export_combo.currentData() == "merged_16bit"

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text":"hi"}\n', encoding="utf-8")
    cfg = tab._build_config("job-x", "a-model", str(dataset))
    assert cfg["export_strategy"] == "merged_16bit"
    assert cfg["eval_dataset_path"] is None

    tab.export_combo.setCurrentIndex(1)
    cfg = tab._build_config("job-y", "a-model", str(dataset))
    assert cfg["export_strategy"] == "adapter"



def test_all_training_controls_are_serialized(qapp, tmp_path: Path) -> None:
    tab = _make_tab(
        tmp_path,
        job_manager=MagicMock(),
        model_service=MagicMock(),
        state=MagicMock(),
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text('{"text":"train"}\n', encoding="utf-8")
    eval_path.write_text('{"text":"eval"}\n', encoding="utf-8")
    tab.steps_spin.setValue(0)
    tab.epochs_spin.setValue(4)
    tab.warmup_spin.setValue(12)
    tab.optimizer_combo.setCurrentIndex(1)
    tab.save_strategy_combo.setCurrentIndex(1)
    tab.save_steps_spin.setValue(25)
    tab.eval_strategy_combo.setCurrentIndex(1)
    tab.eval_steps_spin.setValue(10)
    tab.eval_dataset_combo.addItem("evaluation", str(eval_path))
    tab.eval_dataset_combo.setCurrentIndex(tab.eval_dataset_combo.count() - 1)

    cfg = tab._build_config("job-config", "a-model", str(train_path))
    assert cfg["max_steps"] == 0
    assert cfg["num_train_epochs"] == 4
    assert cfg["warmup_steps"] == 12
    assert cfg["optim"] == "adamw_torch"
    assert cfg["save_strategy"] == "steps"
    assert cfg["save_steps"] == 25
    assert cfg["eval_strategy"] == "steps"
    assert cfg["eval_steps"] == 10
    assert cfg["eval_dataset_path"] == str(eval_path)

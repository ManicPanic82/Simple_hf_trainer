"""Heretic abliteration worker speaking the Simple HF Trainer JSON-lines protocol.

Runs in the isolated Heretic environment and drives unattended abliteration.
Includes the prompt_toolkit DummyOutput patch to avoid Windows NoConsoleScreenBufferError.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import sys
import tempfile
import threading
import traceback
from pathlib import Path

# stdout is reserved for the JSON-lines protocol. OutputInterceptor replaces
# sys.stdout while Heretic runs, so keep a private handle to the real stream;
# emitting through the interceptor would recurse back into emit() forever.
_PROTOCOL_STREAM = sys.stdout


def emit(
    msg_type: str,
    job_id: str | None = None,
    payload: dict | None = None,
    seq: int | None = None,
) -> None:
    """Emit one protocol line to stdout."""
    data = {
        "protocol": 1,
        "type": msg_type,
        "payload": payload or {},
    }
    if job_id is not None:
        data["job_id"] = job_id
    if seq is not None:
        data["seq"] = seq
    line = json.dumps(data, ensure_ascii=False)
    _PROTOCOL_STREAM.write(line + "\n")
    _PROTOCOL_STREAM.flush()


def start_cancel_listener(job_id: str) -> None:
    """Convert a protocol cancel message into an interrupt of the Heretic main thread."""

    def listen() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                continue
            if message.get("type") == "cancel" and message.get("job_id") == job_id:
                # Python delivers the signal handler on the main thread. This
                # lets Heretic unwind through its normal finally blocks.
                signal.raise_signal(signal.SIGINT)
                return

    threading.Thread(target=listen, name="heretic-cancel-listener", daemon=True).start()


def install_dummy_prompt_output() -> None:
    """Force prompt_toolkit to use headless output."""
    try:
        from prompt_toolkit.output import DummyOutput
        from prompt_toolkit.output import defaults as output_defaults

        def create_output(*args: object, **kwargs: object) -> DummyOutput:
            return DummyOutput()

        output_defaults.create_output = create_output  # type: ignore[assignment]
    except Exception:
        pass


class OutputInterceptor:
    """Intercept stdout/stderr from Heretic and transform into protocol messages."""

    def __init__(self, job_id: str, total_trials: int, real_stdout):
        self.job_id = job_id
        self.total_trials = max(1, total_trials)
        self.real_stdout = real_stdout
        self.current_trial = 0

    def write(self, text: str) -> int:
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            # Parse trials
            trial_match = re.search(r"Running trial (\d+) of (\d+)", line_str)
            if trial_match:
                self.current_trial = int(trial_match.group(1))
                frac = min(0.95, 0.1 + 0.8 * (self.current_trial / self.total_trials))
                emit(
                    "progress",
                    self.job_id,
                    {
                        "fraction": round(frac, 3),
                        "step": self.current_trial,
                        "total": self.total_trials,
                    },
                )
                emit("log", self.job_id, {"message": line_str, "level": "info"})
                continue

            # Parse refusals
            refusals_match = re.search(r"Refusals:\s*(\d+)/(\d+)", line_str)
            if refusals_match:
                num = int(refusals_match.group(1))
                denom = int(refusals_match.group(2))
                rate = round(num / max(1, denom), 4)
                emit(
                    "metric",
                    self.job_id,
                    {"name": "refusal_rate", "value": rate, "step": self.current_trial},
                )

            # Parse KL divergence
            kl_match = re.search(r"KL divergence:\s*([\d\.]+)", line_str)
            if kl_match:
                kl_val = float(kl_match.group(1))
                emit(
                    "metric",
                    self.job_id,
                    {"name": "kl_divergence", "value": kl_val, "step": self.current_trial},
                )

            emit("log", self.job_id, {"message": line_str, "level": "info"})

        return len(text)

    def flush(self) -> None:
        pass


def run_worker(cfg: dict) -> int:
    job_id = cfg.get("job_id", "heretic-job")
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    trials = int(cfg.get("trials", 2))
    export_strategy = cfg.get("export_strategy", "merge")
    output_dir = Path(cfg.get("output_dir", "outputs/heretic_model")).resolve()
    seed = int(cfg.get("seed", 42))
    if export_strategy not in {"merge", "adapter"}:
        emit("error", job_id, {"message": "Export strategy must be 'merge' or 'adapter'."})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    emit("hello", payload={"role": "heretic-worker", "pid": os.getpid()})

    install_dummy_prompt_output()

    try:
        import heretic

        version = getattr(heretic, "__version__", "unknown")
        emit("ready", payload={"engine": "heretic", "version": version})
    except Exception as error:
        emit("error", job_id, {"message": f"Failed to import Heretic: {error}"})
        emit("job_complete", job_id, {"success": False, "exit_code": 1})
        return 1

    emit(
        "log",
        job_id,
        {
            "message": f"Configuring abliteration for {model_name} ({trials} trials)...",
            "level": "info",
        },
    )
    emit("progress", job_id, {"fraction": 0.05, "step": 0, "total": trials})

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    toml_content = f"""
model = {json.dumps(model_name)}
quantization = "bnb_4bit"
batch_size = 2
max_response_length = 32
offload_outputs_to_cpu = true
seed = {seed}
n_trials = {trials}
n_startup_trials = 1
modified_model_type = "abliterated"
modifiers = [
    {{ plugin = "heretic.modifiers.abliteration.Abliteration" }},
]
export_strategy = {json.dumps(export_strategy)}
checkpoint_action = "restart"
trial_index = 0
model_action = "save"
save_directory = {json.dumps(output_dir.as_posix())}
study_checkpoint_dir = {json.dumps(checkpoints_dir.as_posix())}

[[response_prefix_test_prompts]]
dataset = "mlabonne/harmless_alpaca"
split = "train[:5]"
column = "text"

[scorer.KeywordRate.prompts]
dataset = "mlabonne/harmful_behaviors"
split = "test[:5]"
column = "text"

[scorer.KLDivergence.prompts]
dataset = "mlabonne/harmless_alpaca"
split = "test[:5]"
column = "text"

[modifier.Abliteration.good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:5]"
column = "text"

[modifier.Abliteration.bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:5]"
column = "text"
"""

    with tempfile.NamedTemporaryFile(
        "w", suffix=".toml", delete=False, encoding="utf-8"
    ) as temp_cfg:
        temp_cfg.write(toml_content)
        temp_cfg_path = temp_cfg.name

    os.environ["HERETIC_CONFIG_FILE"] = temp_cfg_path

    real_stdout = sys.stdout
    interceptor = OutputInterceptor(job_id, trials, real_stdout)

    try:
        from heretic.main import main as heretic_main

        emit(
            "log",
            job_id,
            {"message": "Starting Heretic abliteration optimization...", "level": "info"},
        )
        sys.stdout = interceptor  # type: ignore[assignment]
        sys.argv = ["heretic"]
        # Defer the stdin reader until after all engine imports. On Windows a
        # daemon thread blocked reading an open QProcess stdin pipe deadlocks
        # the numpy/torch C-extension imports performed by Heretic. Cancellation
        # messages sent during startup stay buffered in the pipe until here.
        start_cancel_listener(job_id)
        result = heretic_main()
        rc = 0 if result is None else int(result)
    except KeyboardInterrupt:
        emit(
            "log",
            job_id,
            {"message": "Cancellation acknowledged; Heretic is stopping.", "level": "warning"},
        )
        rc = 130
    except Exception as error:
        tb = traceback.format_exc()
        emit("error", job_id, {"message": f"{error}\n{tb[-1000:]}"})
        rc = 1
    finally:
        sys.stdout = real_stdout
        with contextlib.suppress(OSError):
            os.remove(temp_cfg_path)

    if rc == 0:
        emit("progress", job_id, {"fraction": 1.0, "step": trials, "total": trials})
        artifacts = (
            [p.name for p in output_dir.iterdir() if p.is_file()] if output_dir.is_dir() else []
        )
        emit(
            "result",
            job_id,
            {"output_dir": str(output_dir), "model_saved": True, "artifacts": artifacts},
        )
        emit("job_complete", job_id, {"success": True, "exit_code": 0})
    else:
        emit("job_complete", job_id, {"success": False, "exit_code": rc})

    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heretic abliteration worker")
    parser.add_argument("--config", help="Path to JSON config or inline JSON string")
    args = parser.parse_args(argv)

    cfg = {}
    if args.config:
        text = args.config.strip()
        if text.startswith("{"):
            cfg = json.loads(text)
        elif Path(text).is_file():
            with open(text, encoding="utf-8") as f:
                cfg = json.load(f)

    if not cfg and not sys.stdin.isatty():
        try:
            line = sys.stdin.readline().strip()
            if line:
                cfg = json.loads(line)
        except Exception:
            pass

    return run_worker(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

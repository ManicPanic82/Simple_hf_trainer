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

from package_metadata import write_heretic_package

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


def _score_name(name: str) -> str:
    """Map a Heretic score name to the metric names used by the GUI."""
    lowered = name.lower()
    if "refus" in lowered or "keyword" in lowered:
        return "refusal_rate"
    if "kl divergence" in lowered:
        return "kl_divergence"
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "score"


_NON_SCORE_WORDS = ("version", "memory", "driver", "cuda", "time", "parameter", "ram", "gpu")


def _is_score_line(name: str) -> bool:
    """System information is printed with the same bullet style as scores."""
    lowered = name.lower()
    return not any(word in lowered for word in _NON_SCORE_WORDS)


def _score_value(name: str, raw: str) -> float | None:
    """Convert a displayed score such as ``2/5`` or ``0.0123`` to a float."""
    fraction = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", raw)
    if fraction:
        denom = float(fraction.group(2))
        return round(float(fraction.group(1)) / denom, 6) if denom else None
    number = re.match(r"^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", raw, re.IGNORECASE)
    if number:
        return float(number.group(0))
    if "by definition" in raw.lower():
        return 0.0
    return None


class OutputInterceptor:
    """Intercept stdout/stderr from Heretic and transform into protocol messages."""

    def __init__(self, job_id: str, total_trials: int, real_stdout):
        self.job_id = job_id
        self.total_trials = max(1, total_trials)
        self.real_stdout = real_stdout
        self.current_trial = 0
        # Heretic writes through several buffered streams. A write can end in
        # the middle of a line, so complete lines are assembled here instead of
        # treating every write as a whole line.
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(raw_line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            raw_line, self._buffer = self._buffer, ""
            self._process_line(raw_line.rstrip("\r"))

    def _process_line(self, line: str) -> None:
        line_str = line.strip()
        if not line_str:
            return

        # Parse trials
        trial_match = re.search(r"Running trial (\d+) of (\d+)", line_str)
        if trial_match:
            self.current_trial = int(trial_match.group(1))
            self.total_trials = max(1, int(trial_match.group(2)))
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
            emit(
                "log",
                self.job_id,
                {"message": line, "level": "info", "stream": "stdout"},
            )
            return

        # Every Heretic score is printed as "  * <score name>: <value>".
        # Support bare "Refusals: 2/5" lines as well for older releases.
        score_match = re.match(r"^(?:\*\s+)?([^:]{2,60}):\s*(.+)$", line_str)
        if score_match:
            name = score_match.group(1).strip()
            raw_value = score_match.group(2).strip()
            value = _score_value(name, raw_value)
            if value is not None and _is_score_line(name):
                emit(
                    "metric",
                    self.job_id,
                    {
                        "name": _score_name(name),
                        "value": value,
                        "step": self.current_trial,
                    },
                )

        # Preserve the original line text; the GUI adds timestamps and colours.
        emit("log", self.job_id, {"message": line, "level": "info", "stream": "stdout"})


def _config_value(toml_text: str, key: str, default: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\"([^\"]*)\"", toml_text)
    return match.group(1) if match else default


def _config_int(toml_text: str, key: str, default: int) -> int:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*(\d+)", toml_text)
    return int(match.group(1)) if match else default


def run_worker(cfg: dict) -> int:
    job_id = cfg.get("job_id", "heretic-job")
    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = Path(cfg.get("output_dir", "outputs/heretic_model")).resolve()
    full_config = cfg.get("toml")
    has_full_config = isinstance(full_config, str) and bool(full_config.strip())
    if has_full_config:
        # The GUI generated a complete Heretic 2.x config.toml. Only read the
        # few keys needed for progress/metadata; Heretic validates the rest.
        trials = _config_int(full_config, "n_trials", 2)
        export_strategy = _config_value(full_config, "export_strategy", "merge")
        model_action = _config_value(full_config, "model_action", "save")
    else:
        trials = int(cfg.get("trials", 2))
        export_strategy = cfg.get("export_strategy", "merge")
        model_action = "save"
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
            "stream": "system",
        },
    )
    emit("progress", job_id, {"fraction": 0.05, "step": 0, "total": trials})

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir.parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if has_full_config:
        toml_content = full_config
    else:
        # Legacy fallback for jobs queued or submitted by older app versions
        # that only supply the compact abliteration options.
        seed = int(cfg.get("seed", 42))
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
            {
                "message": "Starting Heretic abliteration optimization...",
                "level": "info",
                "stream": "system",
            },
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
            {
                "message": "Cancellation acknowledged; Heretic is stopping.",
                "level": "warning",
                "stream": "system",
            },
        )
        rc = 130
    except Exception as error:
        tb = traceback.format_exc()
        emit("error", job_id, {"message": f"{error}\n{tb[-1000:]}"})
        rc = 1
    finally:
        # Emit any complete line that was still buffered when the engine
        # stopped, then restore the real stream for the protocol messages.
        interceptor.flush()
        sys.stdout = real_stdout
        with contextlib.suppress(OSError):
            os.remove(temp_cfg_path)

    if rc == 0:
        emit("progress", job_id, {"fraction": 1.0, "step": trials, "total": trials})
        artifacts = (
            [p.name for p in output_dir.iterdir() if p.is_file()] if output_dir.is_dir() else []
        )
        if has_full_config and model_action == "save" and not artifacts:
            # Heretic can print a validation error and return normally without
            # exporting anything. Never report that as a successful job.
            message = (
                "Heretic finished but produced no saved model. "
                "Check the engine log for configuration or objective errors."
            )
            emit("error", job_id, {"message": message})
            emit("job_complete", job_id, {"success": False, "exit_code": 1, "message": message})
            return 1
        if has_full_config and model_action == "save":
            write_heretic_package(
                output_dir=output_dir,
                job_dir=Path(cfg.get("job_dir", output_dir.parent)),
                config_toml=toml_content,
                metadata={
                    "application": "Simple HF Trainer",
                    "job_id": job_id,
                    "source_model": model_name,
                    "heretic_version": version,
                    "export_strategy": export_strategy,
                    "trials": trials,
                },
            )
            artifacts = [p.name for p in output_dir.iterdir() if p.is_file()]
            emit(
                "log",
                job_id,
                {
                    "message": "Archived Heretic configuration and dataset provenance with the model.",
                    "level": "info",
                    "stream": "system",
                },
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

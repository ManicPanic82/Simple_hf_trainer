"""QProcess-backed worker runner speaking the JSON-lines protocol.

Malformed output is reported and skipped; it never raises into the GUI thread.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from workers.protocol import (
    MAX_MESSAGE_BYTES,
    MessageType,
    ProtocolError,
    WorkerMessage,
    decode_message,
    encode_message,
)

MAX_BUFFERED_BYTES = MAX_MESSAGE_BYTES * 4


def split_complete_lines(buffer: bytes, chunk: bytes) -> tuple[list[str], bytes]:
    """Split ``buffer + chunk`` into complete UTF-8 lines plus a remainder.

    Engine stderr is not line-buffered by QProcess, so a read can end in the
    middle of a line. Keeping the remainder prevents two half lines from being
    shown as two unrelated log entries.
    """
    data = buffer + chunk
    if not data:
        return [], b""
    parts = data.split(b"\n")
    remainder = parts.pop()
    lines = [part.decode("utf-8", errors="replace").rstrip("\r") for part in parts]
    return lines, remainder


class WorkerProcess(QObject):
    """Runs one engine worker and validates everything it prints."""

    message_received = Signal(object)  # WorkerMessage
    protocol_error = Signal(object)  # ProtocolError
    log_line = Signal(str)
    started = Signal()
    exited = Signal(int)
    failed_to_start = Signal(str)

    def __init__(
        self,
        interpreter: str | Path,
        script: str | Path,
        *,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._interpreter = str(interpreter)
        self._script = str(script)
        self._args = list(args or [])
        self._buffer = b""
        self._stderr_buffer = b""
        self._stopping = False
        self._cwd = str(cwd) if cwd is not None else ""

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.setProgram(self._interpreter)
        self._process.setArguments([self._script, *self._args])
        if cwd is not None:
            # Engines write scratch files next to the process (e.g. Unsloth's
            # ``unsloth_compiled_cache``), so give them a controlled directory.
            self._process.setWorkingDirectory(str(cwd))
        if env:
            environment: QProcessEnvironment = self._process.processEnvironment()
            for key, value in env.items():
                environment.insert(key, value)
            self._process.setProcessEnvironment(environment)

        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.started.connect(self.started.emit)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._buffer = b""
        self._stderr_buffer = b""
        self._stopping = False
        # Make the actual launch target visible in the UI log. Engine imports can
        # silently use the wrong interpreter if settings or environment variables
        # resolve to a different Python, so record exactly what QProcess runs.
        self.log_line.emit(
            f"[launch] interpreter={self._interpreter} "
            f"script={self._script} args={len(self._args)} cwd={self._cwd or '<inherited>'}"
        )
        self._process.start()

    @property
    def running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def pid(self) -> int:
        return int(self._process.processId())

    def stop(self, grace_ms: int = 3000) -> None:
        """Ask the worker to exit, then terminate it if it does not comply."""
        if not self.running:
            return
        self._stopping = True
        self._process.terminate()
        if not self._process.waitForFinished(grace_ms):
            self._process.kill()
            self._process.waitForFinished(2000)

    def send(self, message: WorkerMessage) -> bool:
        text = encode_message(
            message.type,
            job_id=message.job_id,
            payload=message.payload,
            seq=message.seq,
        )
        return self.send_raw(text)

    def send_message(
        self,
        message_type: MessageType | str,
        *,
        job_id: str | None = None,
        payload: dict | None = None,
    ) -> bool:
        return self.send_raw(encode_message(message_type, job_id=job_id, payload=payload))

    def request_cancel(
        self,
        job_id: str,
        *,
        reason: str = "Cancelled by user.",
        grace_ms: int = 10_000,
    ) -> bool:
        """Request graceful cancellation and schedule a non-blocking hard-stop fallback.

        The GUI thread is never held while waiting for an engine. Workers that
        honour the protocol exit themselves; an unresponsive engine is
        terminated after the grace period and killed shortly afterwards.
        """
        sent = self.send_message(MessageType.CANCEL, job_id=job_id, payload={"reason": reason})
        # Even if stdin has already closed or the write fails, cancellation
        # must not strand a GPU-owning process indefinitely.
        if self.running:
            QTimer.singleShot(max(0, grace_ms), self._terminate_after_cancel)
        return sent

    def _terminate_after_cancel(self) -> None:
        if not self.running:
            return
        self._stopping = True
        self._process.terminate()
        QTimer.singleShot(2_000, self._kill_after_cancel)

    def _kill_after_cancel(self) -> None:
        if self.running:
            self._process.kill()

    def send_raw(self, text: str) -> bool:
        if not self.running:
            return False
        data = (text.rstrip("\r\n") + "\n").encode("utf-8")
        return self._process.write(data) == len(data)

    # -- output handling -------------------------------------------------

    def _read_stdout(self) -> None:
        chunk = bytes(self._process.readAllStandardOutput())
        if not chunk:
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split(b"\n", 1)
            self._handle_line(raw_line)
        if len(self._buffer) > MAX_BUFFERED_BYTES:
            self.protocol_error.emit(
                ProtocolError(
                    "Worker produced an oversized line without a newline; buffer dropped.",
                    code="message_too_large",
                )
            )
            self._buffer = b""

    def _read_stderr(self) -> None:
        chunk = bytes(self._process.readAllStandardError())
        if not chunk:
            return
        lines, remainder = split_complete_lines(self._stderr_buffer, chunk)
        self._stderr_buffer = remainder
        for line in lines:
            if line.strip():
                self.log_line.emit(line)
        if len(self._stderr_buffer) > MAX_BUFFERED_BYTES:
            self.log_line.emit(
                "[stderr] unterminated output line exceeded "
                f"{MAX_BUFFERED_BYTES} bytes; remainder discarded."
            )
            self._stderr_buffer = b""

    def _handle_line(self, raw_line: bytes) -> None:
        text = raw_line.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            message = decode_message(text)
        except ProtocolError as error:
            self.protocol_error.emit(error)
            return
        self.message_received.emit(message)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._buffer.strip():
            self._handle_line(self._buffer)
            self._buffer = b""
        if self._stderr_buffer.strip():
            remainder = self._stderr_buffer.decode("utf-8", errors="replace").strip()
            self._stderr_buffer = b""
            if remainder:
                self.log_line.emit(remainder)
        self.exited.emit(int(exit_code))

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.failed_to_start.emit(f"Worker process could not be started: {self._interpreter}")
        elif not self._stopping:
            self.log_line.emit(f"Worker process error: {error.name}")

"""Structured worker protocol (JSON lines over stdin/stdout).

Every worker message is a single UTF-8 JSON object on one line:

    {"protocol": 1, "type": "progress", "job_id": "job-1", "seq": 7,
     "payload": {"fraction": 0.42, "step": 21, "total": 50}}

Parsing is strict about the envelope and the documented required payload
fields, permissive about extra payload keys (forward compatibility), and never
raises anything except :class:`ProtocolError`. The GUI treats a malformed line
as a visible protocol warning rather than a crash.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024
MAX_ERROR_DETAIL = 400


class MessageType(StrEnum):
    HELLO = "hello"
    READY = "ready"
    JOB_START = "job_start"
    PROGRESS = "progress"
    LOG = "log"
    METRIC = "metric"
    ARCHITECTURE = "architecture"
    RESULT = "result"
    ERROR = "error"
    CANCEL = "cancel"
    JOB_COMPLETE = "job_complete"


JOB_SCOPED_TYPES = frozenset(
    {
        MessageType.JOB_START,
        MessageType.PROGRESS,
        MessageType.LOG,
        MessageType.METRIC,
        MessageType.ARCHITECTURE,
        MessageType.RESULT,
        MessageType.ERROR,
        MessageType.CANCEL,
        MessageType.JOB_COMPLETE,
    }
)

LOG_LEVELS = frozenset({"debug", "info", "warning", "error"})


class ProtocolError(ValueError):
    """A worker line that could not be accepted."""

    def __init__(self, message: str, *, code: str, line: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.line = (line or "")[:MAX_ERROR_DETAIL]

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.code}: {super().__str__()}"


@dataclass(frozen=True)
class WorkerMessage:
    type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    seq: int | None = None
    protocol: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "protocol": self.protocol,
            "type": self.type.value,
            "payload": self.payload,
        }
        if self.job_id is not None:
            data["job_id"] = self.job_id
        if self.seq is not None:
            data["seq"] = self.seq
        return data


def encode_message(
    message_type: MessageType | str,
    *,
    job_id: str | None = None,
    payload: dict[str, Any] | None = None,
    seq: int | None = None,
) -> str:
    """Serialise a message, validating it with the same rules as decoding."""
    if isinstance(message_type, str):
        try:
            message_type = MessageType(message_type)
        except ValueError as error:
            raise ProtocolError(
                f"Unknown message type '{message_type}'", code="unknown_type"
            ) from error

    data: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "type": message_type.value,
        "payload": {} if payload is None else payload,
    }
    if job_id is not None:
        data["job_id"] = job_id
    if seq is not None:
        data["seq"] = seq

    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    decode_message(text)  # fail fast on programmer error
    return text


def decode_message(line: str, *, max_bytes: int = MAX_MESSAGE_BYTES) -> WorkerMessage:
    """Parse and validate one protocol line.

    Raises :class:`ProtocolError` with a stable ``code`` for every rejection.
    """
    if not isinstance(line, str):
        raise ProtocolError("Message must be text, not bytes.", code="not_text")

    stripped = line.strip()
    if not stripped:
        raise ProtocolError("Empty message line.", code="empty_message")
    if len(stripped.encode("utf-8")) > max_bytes:
        raise ProtocolError(
            f"Message exceeds the {max_bytes} byte limit.", code="message_too_large"
        )

    try:
        data = json.loads(stripped)
    except ValueError as error:
        raise ProtocolError(f"Invalid JSON: {error}", code="invalid_json", line=stripped) from error

    if not isinstance(data, dict):
        raise ProtocolError("Message must be a JSON object.", code="not_an_object", line=stripped)

    version = data.get("protocol", PROTOCOL_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProtocolError(
            "'protocol' must be an integer.", code="invalid_protocol", line=stripped
        )
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported protocol version {version}.", code="unsupported_protocol", line=stripped
        )

    raw_type = data.get("type")
    if raw_type is None:
        raise ProtocolError("Message is missing 'type'.", code="missing_type", line=stripped)
    if not isinstance(raw_type, str):
        raise ProtocolError("'type' must be a string.", code="invalid_type", line=stripped)
    try:
        message_type = MessageType(raw_type)
    except ValueError as error:
        raise ProtocolError(
            f"Unknown message type '{raw_type}'.", code="unknown_type", line=stripped
        ) from error

    job_id: str | None = None
    raw_job_id = data.get("job_id")
    if raw_job_id is not None:
        if not isinstance(raw_job_id, str) or not raw_job_id.strip():
            raise ProtocolError(
                "'job_id' must be a non-empty string.", code="invalid_job_id", line=stripped
            )
        job_id = raw_job_id
    elif message_type in JOB_SCOPED_TYPES:
        raise ProtocolError(
            f"'{message_type.value}' requires a 'job_id'.", code="missing_job_id", line=stripped
        )

    raw_seq = data.get("seq")
    invalid_seq = raw_seq is not None and (
        isinstance(raw_seq, bool) or not isinstance(raw_seq, int) or raw_seq < 0
    )
    if invalid_seq:
        raise ProtocolError(
            "'seq' must be a non-negative integer.", code="invalid_sequence", line=stripped
        )

    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError(
            "'payload' must be a JSON object.", code="invalid_payload", line=stripped
        )
    _validate_payload(message_type, payload, stripped)

    return WorkerMessage(
        type=message_type,
        payload=payload,
        job_id=job_id,
        seq=raw_seq,
        protocol=version,
    )


def _validate_payload(message_type: MessageType, payload: dict[str, Any], line: str) -> None:
    if message_type is MessageType.PROGRESS:
        fraction = payload.get("fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ProtocolError(
                "Progress messages require a numeric 'fraction'.",
                code="invalid_progress",
                line=line,
            )
        if not 0.0 <= float(fraction) <= 1.0:
            raise ProtocolError(
                "'fraction' must be between 0 and 1.", code="invalid_progress", line=line
            )
        for key in ("step", "total"):
            value = payload.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ProtocolError(
                    f"'{key}' must be a non-negative integer.", code="invalid_progress", line=line
                )

    elif message_type is MessageType.LOG:
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ProtocolError(
                "Log messages require a non-empty 'message'.", code="invalid_log", line=line
            )
        level = payload.get("level", "info")
        if not isinstance(level, str) or level not in LOG_LEVELS:
            raise ProtocolError(
                f"'level' must be one of {sorted(LOG_LEVELS)}.", code="invalid_log", line=line
            )

    elif message_type is MessageType.METRIC:
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProtocolError(
                "Metric messages require a non-empty 'name'.", code="invalid_metric", line=line
            )
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(
                "Metric messages require a numeric 'value'.", code="invalid_metric", line=line
            )

    elif message_type is MessageType.ERROR:
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ProtocolError(
                "Error messages require a non-empty 'message'.", code="invalid_error", line=line
            )
        recoverable = payload.get("recoverable")
        if recoverable is not None and not isinstance(recoverable, bool):
            raise ProtocolError("'recoverable' must be a boolean.", code="invalid_error", line=line)

    elif message_type is MessageType.JOB_COMPLETE:
        success = payload.get("success")
        if success is not None and not isinstance(success, bool):
            raise ProtocolError(
                "'success' must be a boolean.", code="invalid_completion", line=line
            )
        exit_code = payload.get("exit_code")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            raise ProtocolError(
                "'exit_code' must be an integer.", code="invalid_completion", line=line
            )


def parse_stream(
    lines: Iterable[str], *, max_bytes: int = MAX_MESSAGE_BYTES
) -> tuple[list[WorkerMessage], list[ProtocolError]]:
    """Decode many lines, collecting valid messages and protocol failures."""
    messages: list[WorkerMessage] = []
    errors: list[ProtocolError] = []
    for line in lines:
        try:
            messages.append(decode_message(line, max_bytes=max_bytes))
        except ProtocolError as error:
            errors.append(error)
    return messages, errors

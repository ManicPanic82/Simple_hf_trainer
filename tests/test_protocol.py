"""Worker protocol: valid messages, malformed input, and forward compatibility."""

from __future__ import annotations

import json

import pytest

from workers.protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    MessageType,
    ProtocolError,
    decode_message,
    encode_message,
    parse_stream,
)


def test_progress_roundtrip() -> None:
    text = encode_message(
        MessageType.PROGRESS,
        job_id="job-1",
        payload={"fraction": 0.42, "step": 21, "total": 50},
        seq=7,
    )
    message = decode_message(text)

    assert message.type is MessageType.PROGRESS
    assert message.job_id == "job-1"
    assert message.seq == 7
    assert message.protocol == PROTOCOL_VERSION
    assert message.payload == {"fraction": 0.42, "step": 21, "total": 50}


def test_hello_does_not_need_job_id() -> None:
    message = decode_message(
        json.dumps({"protocol": 1, "type": "hello", "payload": {"role": "worker"}})
    )
    assert message.type is MessageType.HELLO
    assert message.job_id is None


def test_every_documented_type_has_a_valid_example() -> None:
    examples = {
        MessageType.HELLO: ({"role": "worker"}, None),
        MessageType.READY: ({"engine": "unsloth", "version": "1"}, None),
        MessageType.JOB_START: ({"model": "tiny"}, "job-2"),
        MessageType.PROGRESS: ({"fraction": 0.0}, "job-2"),
        MessageType.LOG: ({"message": "starting", "level": "info"}, "job-2"),
        MessageType.METRIC: ({"name": "loss", "value": 1.5, "step": 3}, "job-2"),
        MessageType.RESULT: ({"output_dir": "C:\\out"}, "job-2"),
        MessageType.ERROR: ({"message": "boom", "recoverable": False}, "job-2"),
        MessageType.CANCEL: ({"reason": "user"}, "job-2"),
        MessageType.JOB_COMPLETE: ({"success": True, "exit_code": 0}, "job-2"),
    }
    for message_type, (payload, job_id) in examples.items():
        text = encode_message(message_type, job_id=job_id, payload=payload)
        decoded = decode_message(text)
        assert decoded.type is message_type
        assert decoded.payload == payload


def test_job_scoped_type_requires_job_id() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"type": "progress", "payload": {"fraction": 0.1}}))
    assert caught.value.code == "missing_job_id"


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message("{oops")
    assert caught.value.code == "invalid_json"
    assert caught.value.line


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message("[1, 2, 3]")
    assert caught.value.code == "not_an_object"


def test_empty_line_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message("   \n")
    assert caught.value.code == "empty_message"


def test_missing_type_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"payload": {}}))
    assert caught.value.code == "missing_type"


def test_unknown_type_is_rejected_without_crashing() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"type": "teleport"}))
    assert caught.value.code == "unknown_type"


def test_payload_must_be_object() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"type": "hello", "payload": [1, 2]}))
    assert caught.value.code == "invalid_payload"


def test_progress_fraction_bounds_are_enforced() -> None:
    for bad in (-0.1, 1.5, "half", None, True):
        with pytest.raises(ProtocolError) as caught:
            decode_message(
                json.dumps({"type": "progress", "job_id": "j", "payload": {"fraction": bad}})
            )
        assert caught.value.code == "invalid_progress"


def test_invalid_step_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(
            json.dumps(
                {"type": "progress", "job_id": "j", "payload": {"fraction": 0.1, "step": -1}}
            )
        )
    assert caught.value.code == "invalid_progress"


def test_log_requires_message_and_known_level() -> None:
    with pytest.raises(ProtocolError):
        decode_message(json.dumps({"type": "log", "job_id": "j", "payload": {}}))
    with pytest.raises(ProtocolError):
        decode_message(
            json.dumps({"type": "log", "job_id": "j", "payload": {"message": "x", "level": "loud"}})
        )


def test_metric_requires_numeric_value() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(
            json.dumps(
                {"type": "metric", "job_id": "j", "payload": {"name": "loss", "value": "low"}}
            )
        )
    assert caught.value.code == "invalid_metric"


def test_error_requires_message() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"type": "error", "job_id": "j", "payload": {}}))
    assert caught.value.code == "invalid_error"


def test_sequence_must_be_non_negative_integer() -> None:
    for bad in (-1, "3", True, 1.5):
        with pytest.raises(ProtocolError) as caught:
            decode_message(json.dumps({"type": "hello", "seq": bad}))
        assert caught.value.code == "invalid_sequence"


def test_unsupported_protocol_version_is_rejected() -> None:
    with pytest.raises(ProtocolError) as caught:
        decode_message(json.dumps({"protocol": 99, "type": "hello"}))
    assert caught.value.code == "unsupported_protocol"


def test_oversized_message_is_rejected() -> None:
    huge = json.dumps({"type": "hello", "payload": {"pad": "x" * (MAX_MESSAGE_BYTES + 10)}})
    with pytest.raises(ProtocolError) as caught:
        decode_message(huge)
    assert caught.value.code == "message_too_large"


def test_extra_payload_keys_are_allowed() -> None:
    message = decode_message(
        json.dumps(
            {
                "type": "metric",
                "job_id": "j",
                "payload": {"name": "lr", "value": 0.0002, "future_field": {"nested": True}},
            }
        )
    )
    assert message.payload["future_field"] == {"nested": True}


def test_encode_rejects_unknown_type() -> None:
    with pytest.raises(ProtocolError):
        encode_message("explode", payload={})


def test_parse_stream_separates_good_and_bad_lines() -> None:
    lines = [
        json.dumps({"type": "hello"}),
        "garbage",
        json.dumps({"type": "progress", "job_id": "j", "payload": {"fraction": 0.5}}),
        json.dumps({"type": "nope"}),
    ]
    messages, errors = parse_stream(lines)
    assert [message.type for message in messages] == [MessageType.HELLO, MessageType.PROGRESS]
    assert [error.code for error in errors] == ["invalid_json", "unknown_type"]


def test_unicode_payload_is_preserved() -> None:
    text = encode_message(MessageType.LOG, job_id="j", payload={"message": "café ☕"})
    assert decode_message(text).payload["message"] == "café ☕"
